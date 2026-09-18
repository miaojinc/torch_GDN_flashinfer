# torch-GDN-flashinfer

Standalone PyTorch reference for the FlashInfer SM120 context-parallel GDN
prefill algorithm. It reproduces the same high-level pipeline:

1. 64-token signed, beta-folded triangular `T` precompute.
2. Composition of 64-token transforms into `cp_chunk_len` affine maps.
3. Cross-chunk state fixup.
4. Main prefill from each fixed chunk-boundary state.

The package does not import FlashInfer, CUTLASS, CuTe DSL, TVM-FFI, or
`cuda-python`. It is intended for correctness and debugging, not as a
performance replacement for the SM120 kernel.

## Install

```bash
pip install -e .
```

## Example

```python
import torch
from torch_gdn import torch_chunk_gated_delta_rule_cp_sm120

device = "cuda"
dtype = torch.bfloat16
seq_len = 1024
heads = 8

q = torch.randn(seq_len, heads, 128, device=device, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn_like(q)
alpha = torch.full((seq_len, heads), 0.99, device=device, dtype=torch.float32)
beta = torch.ones_like(alpha)
cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)

output, final_state = torch_chunk_gated_delta_rule_cp_sm120(
    q,
    k,
    v,
    g=alpha,
    beta=beta,
    cu_seqlens=cu_seqlens,
    output_final_state=True,
    use_cp=True,
    _cp_chunk_len=192,
)
```

`g` follows FlashInfer's prefill API and is the decay factor `alpha`, not the
log-decay tensor used by some model-level GDN implementations. The current
SM120 CP kernel consumes caller-preprocessed Q/K, so this reference requires
`use_qk_l2norm_in_kernel=False`; normalize Q/K before both implementations
when comparing them.
