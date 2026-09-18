import math

import torch
import torch.nn.functional as F

from torch_gdn import (
    torch_chunk_gated_delta_rule_cp_sm120,
    torch_cp_delta_rule_sm120,
)


def _make_inputs(seq_lens, hq=1, hk=1, hv=1, dtype=torch.bfloat16):
    torch.manual_seed(0)
    total = sum(seq_lens)
    q = F.normalize(torch.randn(total, hq, 128), dim=-1).to(dtype).contiguous()
    k = F.normalize(torch.randn(total, hk, 128), dim=-1).to(dtype).contiguous()
    v = torch.randn(total, hv, 128).to(dtype).contiguous()
    heads = max(hq, hv)
    alpha = (0.97 + 0.03 * torch.rand(total, heads)).float().contiguous()
    beta = torch.rand(total, heads).float().contiguous()
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64)
    return q, k, v, alpha, beta, cu_seqlens


def _expand_heads(q, k, v):
    heads = max(q.shape[1], v.shape[1])
    if q.shape[1] != heads:
        q = q.repeat_interleave(heads // q.shape[1], dim=1)
    if k.shape[1] != heads:
        k = k.repeat_interleave(heads // k.shape[1], dim=1)
    if v.shape[1] != heads:
        v = v.repeat_interleave(heads // v.shape[1], dim=1)
    return q.float(), k.float(), v.float()


def _recurrent_reference(q, k, v, alpha, beta, cu_seqlens, scale, initial_state=None):
    q, k, v = _expand_heads(q, k, v)
    heads = q.shape[1]
    output = torch.zeros_like(q)
    states = torch.zeros(len(cu_seqlens) - 1, heads, 128, 128)
    if initial_state is not None:
        states.copy_(initial_state.float())
    offsets = cu_seqlens.tolist()
    for seq_idx, (start, end) in enumerate(zip(offsets, offsets[1:])):
        state = states[seq_idx].transpose(-1, -2)
        for token in range(start, end):
            state = alpha[token, :, None, None] * state
            memory = torch.matmul(k[token].unsqueeze(1), state).squeeze(1)
            delta = beta[token, :, None] * (v[token] - memory)
            state = state + k[token].unsqueeze(-1) * delta.unsqueeze(-2)
            output[token] = (
                torch.matmul(q[token].unsqueeze(1), state).squeeze(1) * scale
            )
        states[seq_idx] = state.transpose(-1, -2)
    return output, states


def test_varlen_cp_matches_recurrent_reference():
    q, k, v, alpha, beta, cu_seqlens = _make_inputs([70, 33])
    scale = 1.0 / math.sqrt(128)
    output = torch.empty(103, 1, 128, dtype=q.dtype)
    state = torch.empty(2, 1, 128, 128, dtype=torch.float32)

    debug = torch_cp_delta_rule_sm120(
        output,
        state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        scale,
        max_seqlen=70,
        cp_chunk_len=64,
        return_intermediates=True,
    )
    ref_output, ref_state = _recurrent_reference(
        q, k, v, alpha, beta, cu_seqlens, scale
    )

    assert debug is not None
    assert debug.block_size == 64
    assert debug.cp_chunk_len == 64
    torch.testing.assert_close(output.float(), ref_output, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(state, ref_state, atol=6e-2, rtol=5e-2)


def test_gqa_state_pool_and_checkpoints():
    q, k, v, alpha, beta, cu_seqlens = _make_inputs(
        [128, 64], hq=2, hk=1, hv=1
    )
    pool = torch.randn(4, 2, 128, 128, dtype=torch.float32) * 0.01
    original = pool.clone()
    state_indices = torch.tensor([3, 1], dtype=torch.int32)
    checkpoints = torch.empty(3, 2, 128, 128, dtype=torch.float32)
    checkpoint_starts = torch.tensor([0, 2, 3], dtype=torch.int64)

    output, returned_pool = torch_chunk_gated_delta_rule_cp_sm120(
        q,
        k,
        v,
        g=alpha,
        beta=beta,
        scale=1.0,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        output_state=pool,
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=checkpoint_starts,
        checkpoint_every_n_tokens=64,
        use_cp=True,
        state_indices=state_indices,
        _cp_chunk_len=128,
    )

    assert returned_pool is pool
    assert output.shape == (192, 2, 128)
    torch.testing.assert_close(pool[0], original[0])
    torch.testing.assert_close(pool[2], original[2])
    assert torch.isfinite(pool[1]).all()
    assert torch.isfinite(pool[3]).all()
    assert torch.isfinite(checkpoints).all()


def test_zero_length_sequence():
    q, k, v, alpha, beta, cu_seqlens = _make_inputs([64, 0, 64])
    initial = torch.randn(3, 1, 128, 128, dtype=torch.float32) * 0.01
    output = torch.empty_like(q)
    state = torch.empty_like(initial)

    torch_cp_delta_rule_sm120(
        output,
        state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        1.0,
        initial_state=initial,
        max_seqlen=64,
        cp_chunk_len=64,
    )

    torch.testing.assert_close(state[1], initial[1])
    assert torch.isfinite(output).all()


def test_public_adapter_uses_true_max_seqlen():
    q, k, v, alpha, beta, cu_seqlens = _make_inputs([64, 128])

    output = torch_chunk_gated_delta_rule_cp_sm120(
        q,
        k,
        v,
        g=alpha,
        beta=beta,
        cu_seqlens=cu_seqlens,
        use_cp=True,
    )

    assert output.shape == (192, 1, 128)
    assert torch.isfinite(output).all()


def test_all_sequences_empty_preserves_state():
    q, k, v, alpha, beta, cu_seqlens = _make_inputs([0, 0])
    initial = torch.randn(2, 1, 128, 128, dtype=torch.float32)
    output = torch.empty_like(q)
    state = torch.empty_like(initial)

    torch_cp_delta_rule_sm120(
        output,
        state,
        q,
        k,
        v,
        alpha,
        beta,
        cu_seqlens,
        1.0,
        initial_state=initial,
        max_seqlen=0,
    )

    assert output.numel() == 0
    torch.testing.assert_close(state, initial)
