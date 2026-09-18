"""Standalone PyTorch mirror of FlashInfer's SM120 CP GDN implementation.

FlashInfer source correspondence:

* Public adapter:
  ``flashinfer/gdn_prefill.py::chunk_gated_delta_rule``
* Complete SM120 CP pipeline:
  ``flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py::
  cp_delta_rule_dsl_sm120``
* Varlen workspace/chunk helpers:
  ``flashinfer/gdn_kernels/delta_rule_dsl/varlen_helper.py``

This module preserves the algorithm, packed-varlen indexing, 64-token blocks,
and public tensor layouts. It replaces CuTe-DSL kernels with regular PyTorch
operations and is therefore a correctness reference rather than a performance
implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch


BLOCK_SIZE = 64
HEAD_SIZE = 128
CP_CHUNK_LEN_GRANULARITY = 512
_STATE_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


@dataclass
class TorchGDNCPIntermediates:
    block_size: int
    cp_chunk_len: int
    t: torch.Tensor
    local_transfer: torch.Tensor
    local_state: torch.Tensor
    fixed_state: torch.Tensor


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_up(a: int, b: int) -> int:
    return _ceil_div(a, b) * b


def _chunk_bound(num_items: int, total: int, chunk_size: int) -> int:
    """Mirror FlashInfer ``varlen_helper.py::chunk_bound_host``."""
    m = min(num_items, total)
    return m + (total - m) // chunk_size


def _workspace_num_chunks(num_seqs: int, total: int, chunk_size: int) -> int:
    """Mirror FlashInfer ``varlen_helper.py::workspace_num_chunks_host``."""
    return _chunk_bound(num_seqs, total, chunk_size)


def _seq_metadata(cu_seqlens: torch.Tensor, total_seqlen: int) -> tuple[list[int], list[int]]:
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be 1D, got {tuple(cu_seqlens.shape)}")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")

    offsets = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    if not offsets:
        raise ValueError("cu_seqlens must contain at least one element")
    if offsets[0] != 0 or offsets[-1] != total_seqlen:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at {total_seqlen}, got "
            f"{offsets[0]} and {offsets[-1]}"
        )
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be nondecreasing")
    return offsets, [right - left for left, right in zip(offsets, offsets[1:])]


def _choose_cp_chunk_len(
    max_seqlen: int,
    num_heads: int,
    total_seqlen: int,
    num_seqs: int,
    device: torch.device,
    granularity: int,
) -> int:
    """PyTorch equivalent of ``varlen_helper.py::choose_cp_chunk_len_host``.

    The same one-wave MN-precompute target and SM120 short-workload balancing
    rule are retained. CPU execution uses a one-SM stand-in solely so this
    reference remains independently runnable.
    """
    if granularity <= 0 or granularity % BLOCK_SIZE != 0:
        raise ValueError(
            f"cp_chunk_len_granularity must be a positive multiple of {BLOCK_SIZE}"
        )

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count
        device_name = props.name.lower()
        major, _ = torch.cuda.get_device_capability(device)
    else:
        num_sms = 1
        device_name = str(device).lower()
        major = 12

    is_gddr = any(marker in device_name for marker in ("geforce", "rtx", "workstation"))
    threshold_num, threshold_den = ((1, 3) if is_gddr else (1, 2))

    if major == 12 and num_heads <= 16:
        approx_ctas = _ceil_div(total_seqlen, granularity) * num_heads
        if approx_ctas * threshold_den < num_sms * threshold_num:
            square = _ceil_div(max_seqlen * BLOCK_SIZE, 2)
            balanced = math.isqrt(square)
            if balanced * balanced < square:
                balanced += 1
            return max(BLOCK_SIZE, _round_up(balanced, BLOCK_SIZE))

    target_chunks = max(1, num_sms // num_heads)
    remaining_seqlen = max(0, total_seqlen - max_seqlen)
    remaining_seqs = max(0, num_seqs - 1)

    def chunk_bound_for_len(chunk_len: int) -> int:
        return _ceil_div(max_seqlen, chunk_len) + _chunk_bound(
            remaining_seqs, remaining_seqlen, chunk_len
        )

    lo = 1
    hi = max(1, _ceil_div(max_seqlen, granularity))
    while lo < hi:
        mid = (lo + hi) // 2
        if chunk_bound_for_len(mid * granularity) <= target_chunks:
            hi = mid
        else:
            lo = mid + 1
    return lo * granularity


def _validate_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    o: torch.Tensor,
) -> tuple[int, int, int, int]:
    tensors = (q, k, v, alpha, beta, o)
    if any(t.device != q.device for t in tensors):
        raise ValueError("q/k/v/alpha/beta/o must be on the same device")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or o.ndim != 3:
        raise ValueError("q/k/v/o must be rank-3 packed tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"q must be float16 or bfloat16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype or o.dtype != q.dtype:
        raise ValueError("q/k/v/o dtypes must match")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("q/k/v/alpha/beta/o must be contiguous")

    total_seqlen, num_q_heads, d = q.shape
    if d != HEAD_SIZE:
        raise ValueError(f"SM120-compatible GDN requires head_size={HEAD_SIZE}, got {d}")
    if k.shape[0] != total_seqlen or v.shape[0] != total_seqlen:
        raise ValueError("q/k/v must have the same total token count")
    if k.shape[2] != d or v.shape[2] != d:
        raise ValueError("q/k/v must have the same head dimension")

    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    if num_sab_heads % num_q_heads != 0:
        raise ValueError("num_sab_heads must be a multiple of num_q_heads")
    if num_sab_heads % num_k_heads != 0:
        raise ValueError("num_sab_heads must be a multiple of num_k_heads")
    if num_sab_heads % num_v_heads != 0:
        raise ValueError("num_sab_heads must be a multiple of num_v_heads")
    if alpha.shape != (total_seqlen, num_sab_heads):
        raise ValueError(
            f"alpha must have shape {(total_seqlen, num_sab_heads)}, "
            f"got {tuple(alpha.shape)}"
        )
    if beta.shape != alpha.shape:
        raise ValueError(f"beta must have shape {tuple(alpha.shape)}, got {tuple(beta.shape)}")
    if alpha.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("alpha and beta must be float32")
    if o.shape != (total_seqlen, num_sab_heads, d):
        raise ValueError(
            f"o must have shape {(total_seqlen, num_sab_heads, d)}, got {tuple(o.shape)}"
        )
    return total_seqlen, num_q_heads, num_k_heads, num_v_heads


def _expand_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_sab_heads = max(q.shape[1], v.shape[1])
    if q.shape[1] != num_sab_heads:
        q = q.repeat_interleave(num_sab_heads // q.shape[1], dim=1)
    if k.shape[1] != num_sab_heads:
        k = k.repeat_interleave(num_sab_heads // k.shape[1], dim=1)
    if v.shape[1] != num_sab_heads:
        v = v.repeat_interleave(num_sab_heads // v.shape[1], dim=1)
    return q, k, v


def _pad_block(
    tensor: torch.Tensor,
    valid_len: int,
    *,
    fill_value: float,
) -> torch.Tensor:
    if valid_len == BLOCK_SIZE:
        return tensor
    shape = (BLOCK_SIZE, *tensor.shape[1:])
    padded = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    if valid_len:
        padded[:valid_len].copy_(tensor)
    return padded


def _alpha_terms(alpha_sh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_hs = alpha_sh.transpose(0, 1)
    cumulative_log = torch.cumsum(torch.log(alpha_hs + 1e-10), dim=-1)
    gamma_hs1 = torch.exp(cumulative_log).unsqueeze(-1)
    gamma_hss = torch.exp(
        cumulative_log.unsqueeze(2) - cumulative_log.unsqueeze(1)
    )
    return gamma_hss, gamma_hs1


def _identity_add_strict_lower(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[-1]
    strict_lower = torch.tril(matrix, diagonal=-1)
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    return strict_lower + eye


def _compute_t_block(
    k_shk: torch.Tensor,
    beta_sh: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Build one 64-token T block in FlashInfer's stored orientation.

    Corresponds to ``CPDeltaRuleTPrecomputeSm120``. FlashInfer stores the
    transposed negative beta-folded inverse in the Q/K dtype; retaining that
    representation is important for matching the later MN and prefill stages.
    """
    k_hsk = k_shk.transpose(0, 1).float()
    beta_hs1 = beta_sh.transpose(0, 1).unsqueeze(-1).float()
    ikk = _identity_add_strict_lower(
        beta_hs1 * torch.matmul(k_hsk, k_hsk.transpose(-1, -2))
    )
    eye = torch.eye(BLOCK_SIZE, dtype=torch.float32, device=k_shk.device)
    eye = eye.expand(ikk.shape[0], BLOCK_SIZE, BLOCK_SIZE)
    inverse = torch.linalg.solve_triangular(
        ikk, eye, upper=False, unitriangular=True
    )
    clean = inverse * beta_hs1.transpose(-1, -2)
    return (-clean.transpose(-1, -2)).to(output_dtype)


@torch.inference_mode()
def torch_cp_delta_rule_t_precompute_sm120(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    max_seqlen: int,
    *,
    num_sab_heads: Optional[int] = None,
) -> torch.Tensor:
    """PyTorch stage 1 matching ``CPDeltaRuleTPrecomputeSm120``.

    The returned workspace uses FlashInfer's compact packed-varlen block slots:
    ``[workspace_64_blocks, num_sab_heads, 64, 64]``.
    """
    del max_seqlen
    offsets, seq_lens = _seq_metadata(cu_seqlens, total_seqlen)
    if k.ndim != 3 or k.shape[0] != total_seqlen or k.shape[2] != HEAD_SIZE:
        raise ValueError("k must have shape [total_seqlen, num_k_heads, 128]")
    if k.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("k must be float16 or bfloat16")
    if num_sab_heads is None:
        num_sab_heads = beta.shape[1]
    if num_sab_heads % k.shape[1] != 0:
        raise ValueError("num_sab_heads must be a multiple of num_k_heads")
    if beta.shape != (total_seqlen, num_sab_heads) or beta.dtype != torch.float32:
        raise ValueError("beta must be float32 [total_seqlen, num_sab_heads]")

    k_expanded = (
        k
        if k.shape[1] == num_sab_heads
        else k.repeat_interleave(num_sab_heads // k.shape[1], dim=1)
    )
    total_blocks = _workspace_num_chunks(
        len(seq_lens), total_seqlen, BLOCK_SIZE
    )
    result = torch.zeros(
        total_blocks,
        num_sab_heads,
        BLOCK_SIZE,
        BLOCK_SIZE,
        dtype=k.dtype,
        device=k.device,
    )
    for seq_idx, (seq_start, seq_len) in enumerate(zip(offsets[:-1], seq_lens)):
        block_base = _chunk_bound(seq_idx, seq_start, BLOCK_SIZE)
        for block_idx, block_offset in enumerate(range(0, seq_len, BLOCK_SIZE)):
            valid_len = min(BLOCK_SIZE, seq_len - block_offset)
            start = seq_start + block_offset
            k_block = _pad_block(k_expanded[start : start + valid_len], valid_len, fill_value=0)
            beta_block = _pad_block(beta[start : start + valid_len], valid_len, fill_value=0)
            result[block_base + block_idx] = _compute_t_block(
                k_block, beta_block, k.dtype
            )
    return result


def _compute_cp_chunk_affine_transposed(
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    t_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose 64-token transforms as ``H_out = H_in @ M + N``.

    ``M`` and ``N`` are kept in the transposed ``[V, K]`` workspace orientation
    used by FlashInfer's SM120 MN and fixup kernels.
    """
    num_heads = k.shape[1]
    eye = torch.eye(HEAD_SIZE, dtype=torch.float32, device=k.device).expand(
        num_heads, HEAD_SIZE, HEAD_SIZE
    )
    transfer_t = eye.clone()
    state_t = torch.zeros_like(transfer_t)

    for block_idx, block_offset in enumerate(range(0, k.shape[0], BLOCK_SIZE)):
        valid_len = min(BLOCK_SIZE, k.shape[0] - block_offset)
        k_block = _pad_block(k[block_offset : block_offset + valid_len], valid_len, fill_value=0)
        v_block = _pad_block(v[block_offset : block_offset + valid_len], valid_len, fill_value=0)
        alpha_block = _pad_block(
            alpha[block_offset : block_offset + valid_len], valid_len, fill_value=1
        )

        _, gamma_hs1 = _alpha_terms(alpha_block)
        gamma_hs1 = gamma_hs1.float()
        inv_gamma_h1s = torch.reciprocal(gamma_hs1.transpose(-1, -2))
        block_gamma = gamma_hs1[:, [valid_len - 1], :]

        k_hsk = k_block.transpose(0, 1).float()
        v_hsv = v_block.transpose(0, 1).float()
        t_hss = t_blocks[block_idx].float()
        v_inv_hvs = v_hsv.transpose(-1, -2) * inv_gamma_h1s
        kt_hks = k_hsk.transpose(-1, -2)

        block_state_t = -block_gamma * (
            torch.matmul(torch.matmul(v_inv_hvs, t_hss), k_hsk)
        )
        block_transfer_t = block_gamma * eye + block_gamma * (
            torch.matmul(torch.matmul(kt_hks, t_hss), k_hsk)
        )
        state_t = torch.matmul(state_t, block_transfer_t) + block_state_t
        transfer_t = torch.matmul(transfer_t, block_transfer_t)

    return transfer_t, state_t


@torch.inference_mode()
def torch_cp_delta_rule_mn_precompute_sm120(
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    alpha: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int,
    max_seqlen: int,
    *,
    num_sab_heads: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch stage 2 matching ``CPDeltaRuleMNPrecomputeSm120``.

    Multiple mathematical 64-token blocks are composed into one affine
    transform for each ``cp_chunk_len`` chunk. The two returned FP32 tensors
    correspond to FlashInfer's local transfer ``M`` and local state ``N``
    workspaces.
    """
    del max_seqlen
    if cp_chunk_len <= 0 or cp_chunk_len % BLOCK_SIZE != 0:
        raise ValueError(f"cp_chunk_len must be a positive multiple of {BLOCK_SIZE}")
    offsets, seq_lens = _seq_metadata(cu_seqlens, total_seqlen)
    if num_sab_heads is None:
        num_sab_heads = alpha.shape[1]
    if num_sab_heads % k.shape[1] != 0 or num_sab_heads % v.shape[1] != 0:
        raise ValueError("num_sab_heads must be a multiple of K and V head counts")
    k_expanded = (
        k
        if k.shape[1] == num_sab_heads
        else k.repeat_interleave(num_sab_heads // k.shape[1], dim=1)
    )
    v_expanded = (
        v
        if v.shape[1] == num_sab_heads
        else v.repeat_interleave(num_sab_heads // v.shape[1], dim=1)
    )

    total_cp_chunks = _workspace_num_chunks(
        len(seq_lens), total_seqlen, cp_chunk_len
    )
    shape = (total_cp_chunks, num_sab_heads, HEAD_SIZE, HEAD_SIZE)
    local_transfer = torch.zeros(shape, dtype=torch.float32, device=k.device)
    local_state = torch.zeros_like(local_transfer)

    for seq_idx, (seq_start, seq_len) in enumerate(zip(offsets[:-1], seq_lens)):
        cp_base = _chunk_bound(seq_idx, seq_start, cp_chunk_len)
        t_base = _chunk_bound(seq_idx, seq_start, BLOCK_SIZE)
        for cp_idx, cp_offset in enumerate(range(0, seq_len, cp_chunk_len)):
            chunk_len = min(cp_chunk_len, seq_len - cp_offset)
            start = seq_start + cp_offset
            first_t = cp_offset // BLOCK_SIZE
            num_t = _ceil_div(chunk_len, BLOCK_SIZE)
            transfer, state = _compute_cp_chunk_affine_transposed(
                k_expanded[start : start + chunk_len],
                v_expanded[start : start + chunk_len],
                alpha[start : start + chunk_len],
                t[t_base + first_t : t_base + first_t + num_t],
            )
            local_transfer[cp_base + cp_idx] = transfer
            local_state[cp_base + cp_idx] = state
    return local_transfer, local_state


def _gather_initial_states(
    initial_state: Optional[torch.Tensor],
    state_indices: Optional[torch.Tensor],
    num_seqs: int,
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    if initial_state is None:
        return torch.zeros(
            num_seqs,
            num_heads,
            HEAD_SIZE,
            HEAD_SIZE,
            dtype=torch.float32,
            device=device,
        )
    if initial_state.dtype not in _STATE_DTYPES:
        raise ValueError(f"unsupported initial_state dtype: {initial_state.dtype}")
    if tuple(initial_state.shape[1:]) != (num_heads, HEAD_SIZE, HEAD_SIZE):
        raise ValueError("initial_state must end with [num_heads, 128, 128]")
    if state_indices is None:
        if initial_state.shape[0] != num_seqs:
            raise ValueError("packed initial_state first dimension must equal num_seqs")
        return initial_state.float().clone()
    if state_indices.shape != (num_seqs,) or state_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("state_indices must be int32/int64 [num_seqs]")
    return initial_state.index_select(0, state_indices.long()).float().clone()


@torch.inference_mode()
def torch_cp_delta_rule_fixup_sm120(
    local_transfer: torch.Tensor,
    local_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int,
    *,
    initial_state: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch stage 3 matching FlashInfer's SM120 fixup kernels.

    This is the algorithmic equivalent of both
    ``CPDeltaRuleFixupHmmaSm120`` and ``CPDeltaRuleFixupSimtSm120``. PyTorch
    FP32 matmul replaces the kernel's HMMA/TF32 or SIMT implementation.
    ``fixed_state[slot]`` is the state after that CP chunk; stage 4 reads the
    previous slot as the next chunk's input state.
    """
    offsets, seq_lens = _seq_metadata(cu_seqlens, total_seqlen)
    num_seqs = len(seq_lens)
    num_heads = local_transfer.shape[1]
    initial = _gather_initial_states(
        initial_state, state_indices, num_seqs, num_heads, local_transfer.device
    )
    fixed_state = torch.zeros_like(local_state)
    final_states = initial.clone()

    for seq_idx, (seq_start, seq_len) in enumerate(zip(offsets[:-1], seq_lens)):
        state = initial[seq_idx]
        cp_base = _chunk_bound(seq_idx, seq_start, cp_chunk_len)
        for cp_idx in range(_ceil_div(seq_len, cp_chunk_len)):
            slot = cp_base + cp_idx
            state = (
                torch.matmul(state, local_transfer[slot]) + local_state[slot]
            )
            fixed_state[slot] = state
        final_states[seq_idx] = state
    return fixed_state, final_states


def _run_main_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    t: torch.Tensor,
    state_v_k: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute one 64-token recurrence block from precomputed T.

    This follows the block mathematics fused into
    ``CPDeltaRulePrefillSm120`` rather than replaying the token recurrence.
    """
    valid_len = q.shape[0]
    q_block = _pad_block(q, valid_len, fill_value=0)
    k_block = _pad_block(k, valid_len, fill_value=0)
    v_block = _pad_block(v, valid_len, fill_value=0)
    alpha_block = _pad_block(alpha, valid_len, fill_value=1)

    gamma_hss, gamma_hs1 = _alpha_terms(alpha_block)
    gamma_hss = gamma_hss.float()
    gamma_hs1 = gamma_hs1.float()
    block_gamma = gamma_hs1[:, [valid_len - 1], :]

    q_hsk = q_block.transpose(0, 1).float() * scale
    k_hsk = k_block.transpose(0, 1).float()
    v_hsv = v_block.transpose(0, 1).float()
    clean_beta_inverse = -t.float().transpose(-1, -2)

    u_hsv = torch.matmul(clean_beta_inverse, v_hsv)
    w_hsk = torch.matmul(clean_beta_inverse, gamma_hs1 * k_hsk)
    logical_state_kv = state_v_k.transpose(-1, -2)
    new_v = u_hsv - torch.matmul(w_hsk, logical_state_kv)

    causal = torch.tril(
        torch.ones(BLOCK_SIZE, BLOCK_SIZE, dtype=torch.float32, device=q.device)
    )
    scores = torch.matmul(q_hsk, k_hsk.transpose(-1, -2)) * gamma_hss * causal
    output = (
        torch.matmul(gamma_hs1 * q_hsk, logical_state_kv)
        + torch.matmul(scores, new_v)
    )

    k_decay = torch.exp(
        torch.log(block_gamma) - torch.log(gamma_hs1)
    ) * k_hsk
    logical_state_kv = (
        block_gamma * logical_state_kv
        + torch.matmul(k_decay.transpose(-1, -2), new_v)
    )
    return output[:, :valid_len].transpose(0, 1), logical_state_kv.transpose(-1, -2)


@torch.inference_mode()
def torch_cp_delta_rule_prefill_sm120(
    o: torch.Tensor,
    state: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    fixed_state: torch.Tensor,
    alpha: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int,
    max_seqlen: int,
    *,
    initial_state: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
) -> torch.Tensor:
    """PyTorch stage 4 matching ``CPDeltaRulePrefillSm120``.

    It consumes stage-1 ``T`` and stage-3 fixed boundary states, reconstructs
    outputs blockwise, writes optional checkpoints, and stores final states in
    FlashInfer's public ``[N, H, V, K]`` layout.
    """
    del max_seqlen
    offsets, seq_lens = _seq_metadata(cu_seqlens, total_seqlen)
    num_heads = o.shape[1]
    q_expanded, k_expanded, v_expanded = _expand_heads(q, k, v)
    initial = _gather_initial_states(
        initial_state, state_indices, len(seq_lens), num_heads, q.device
    )
    final_states = initial.clone()

    needs_checkpoints = checkpoint_every_n_tokens > 0
    if needs_checkpoints:
        if checkpoint_every_n_tokens % BLOCK_SIZE != 0:
            raise ValueError(
                f"checkpoint_every_n_tokens must be a multiple of {BLOCK_SIZE}"
            )
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError(
                "state_checkpoints and checkpoint_cu_starts are required"
            )
        checkpoint_offsets, _ = _seq_metadata(
            checkpoint_cu_starts, state_checkpoints.shape[0]
        )
    else:
        if state_checkpoints is not None or checkpoint_cu_starts is not None:
            raise ValueError(
                "checkpoint tensors must be None when checkpointing is disabled"
            )
        checkpoint_offsets = []

    for seq_idx, (seq_start, seq_len) in enumerate(zip(offsets[:-1], seq_lens)):
        cp_base = _chunk_bound(seq_idx, seq_start, cp_chunk_len)
        t_base = _chunk_bound(seq_idx, seq_start, BLOCK_SIZE)
        seq_final = initial[seq_idx]
        for cp_idx, cp_offset in enumerate(range(0, seq_len, cp_chunk_len)):
            cp_valid_len = min(cp_chunk_len, seq_len - cp_offset)
            chunk_state = initial[seq_idx] if cp_idx == 0 else fixed_state[cp_base + cp_idx - 1]
            for local_block, block_offset in enumerate(
                range(0, cp_valid_len, BLOCK_SIZE)
            ):
                valid_len = min(BLOCK_SIZE, cp_valid_len - block_offset)
                token_start = seq_start + cp_offset + block_offset
                t_slot = t_base + (cp_offset + block_offset) // BLOCK_SIZE
                block_output, chunk_state = _run_main_block(
                    q_expanded[token_start : token_start + valid_len],
                    k_expanded[token_start : token_start + valid_len],
                    v_expanded[token_start : token_start + valid_len],
                    alpha[token_start : token_start + valid_len],
                    t[t_slot],
                    chunk_state,
                    scale,
                )
                o[token_start : token_start + valid_len].copy_(block_output.to(o.dtype))

                block_end = cp_offset + block_offset + valid_len
                if (
                    needs_checkpoints
                    and valid_len == BLOCK_SIZE
                    and block_end % checkpoint_every_n_tokens == 0
                ):
                    checkpoint_idx = (
                        checkpoint_offsets[seq_idx]
                        + block_end // checkpoint_every_n_tokens
                        - 1
                    )
                    state_checkpoints[checkpoint_idx].copy_(
                        chunk_state.to(state_checkpoints.dtype)
                    )
            seq_final = chunk_state
        final_states[seq_idx] = seq_final

    if state is not None:
        if state.dtype not in _STATE_DTYPES:
            raise ValueError(f"unsupported state dtype: {state.dtype}")
        if tuple(state.shape[1:]) != (num_heads, HEAD_SIZE, HEAD_SIZE):
            raise ValueError("state must end with [num_heads, 128, 128]")
        if state_indices is None:
            if state.shape[0] != len(seq_lens):
                raise ValueError("packed state first dimension must equal num_seqs")
            state.copy_(final_states.to(state.dtype))
        else:
            state.index_copy_(0, state_indices.long(), final_states.to(state.dtype))
    return final_states


@torch.inference_mode()
def torch_cp_delta_rule_sm120(
    o: torch.Tensor,
    state: Optional[torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    initial_state: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    max_seqlen: Optional[int] = None,
    cp_chunk_len: Optional[int] = None,
    cp_chunk_len_granularity: int = CP_CHUNK_LEN_GRANULARITY,
    return_intermediates: bool = False,
) -> Optional[TorchGDNCPIntermediates]:
    """Run the four-stage PyTorch mirror of ``cp_delta_rule_dsl_sm120``.

    The argument order and tensor contracts intentionally track FlashInfer's
    low-level SM120 entry point so callers can execute both implementations
    with the same inputs. ``return_intermediates`` is reference-only and
    exposes the workspaces for stage-by-stage comparison.
    """
    total_seqlen, _, _, _ = _validate_qkv(q, k, v, alpha, beta, o)
    offsets, seq_lens = _seq_metadata(cu_seqlens, total_seqlen)
    num_seqs = len(seq_lens)
    num_heads = max(q.shape[1], v.shape[1])
    if max_seqlen is None:
        if num_seqs != 1:
            raise ValueError("max_seqlen is required when num_seqs != 1")
        max_seqlen = total_seqlen
    if max_seqlen < max(seq_lens, default=0):
        raise ValueError("max_seqlen must be an upper bound on sequence length")
    if cp_chunk_len is None:
        cp_chunk_len = _choose_cp_chunk_len(
            max_seqlen,
            num_heads,
            total_seqlen,
            num_seqs,
            q.device,
            cp_chunk_len_granularity,
        )
    if cp_chunk_len <= 0 or cp_chunk_len % BLOCK_SIZE != 0:
        raise ValueError(f"cp_chunk_len must be a positive multiple of {BLOCK_SIZE}")

    t = torch_cp_delta_rule_t_precompute_sm120(
        k,
        beta,
        cu_seqlens,
        total_seqlen,
        max_seqlen,
        num_sab_heads=num_heads,
    )
    local_transfer, local_state = torch_cp_delta_rule_mn_precompute_sm120(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len,
        max_seqlen,
        num_sab_heads=num_heads,
    )
    fixed_state, _ = torch_cp_delta_rule_fixup_sm120(
        local_transfer,
        local_state,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len,
        initial_state=initial_state,
        state_indices=state_indices,
    )
    torch_cp_delta_rule_prefill_sm120(
        o,
        state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len,
        max_seqlen,
        initial_state=initial_state,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )
    if not return_intermediates:
        return None
    return TorchGDNCPIntermediates(
        block_size=BLOCK_SIZE,
        cp_chunk_len=cp_chunk_len,
        t=t,
        local_transfer=local_transfer,
        local_state=local_state,
        fixed_state=fixed_state,
    )


@torch.inference_mode()
def torch_chunk_gated_delta_rule_cp_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_cp: Literal["auto"] | bool = True,
    state_indices: Optional[torch.Tensor] = None,
    _cp_chunk_len: Optional[int] = None,
    backend: Literal["auto", "flashinfer"] = "flashinfer",
):
    """Public adapter corresponding to ``gdn_prefill.py::chunk_gated_delta_rule``.

    This standalone variant intentionally implements only the forced SM120
    context-parallel FlashInfer route. Parameter names, defaults, packed
    Q/K/V layout, indexed state-pool behavior, checkpoints, and return shape
    follow the FlashInfer public API. Here ``g`` is FlashInfer's decay factor
    ``alpha`` rather than a model-level log-decay value.
    """
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required")
    if use_cp is not True:
        raise ValueError("this standalone function implements only the forced CP path")
    if backend not in ("auto", "flashinfer"):
        raise ValueError("this standalone function implements only the FlashInfer CP path")
    if use_qk_l2norm_in_kernel:
        raise NotImplementedError(
            "the current SM120 CP kernel consumes caller-preprocessed Q/K; "
            "normalize Q/K before both the reference and kernel calls"
        )

    total_seqlen = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1
    num_heads = max(q.shape[1], v.shape[1])
    _, seq_lens = _seq_metadata(cu_seqlens, total_seqlen)
    max_seqlen = max(seq_lens, default=0)
    if g is None:
        g = torch.ones(
            total_seqlen, num_heads, dtype=torch.float32, device=q.device
        )
    if beta is None:
        beta = torch.ones_like(g)
    if scale is None or scale == 0:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if output is None:
        output = torch.empty(
            total_seqlen, num_heads, q.shape[-1], dtype=q.dtype, device=q.device
        )
    if output_final_state and output_state is None:
        if state_indices is not None:
            raise ValueError(
                "state_indices requires an explicit output_state pool when "
                "output_final_state=True"
            )
        output_state = torch.empty(
            num_seqs,
            num_heads,
            q.shape[-1],
            q.shape[-1],
            dtype=torch.float32,
            device=q.device,
        )

    torch_cp_delta_rule_sm120(
        output,
        output_state,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        float(scale),
        initial_state=initial_state,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        max_seqlen=max_seqlen,
        cp_chunk_len=_cp_chunk_len,
    )
    if output_final_state:
        assert output_state is not None
        return output, output_state
    return output
