import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Scatter key/value vectors into the paged KV cache.

    key, value: [N, num_kv_heads, head_dim]
    k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping: [N]  -- flat slot index = block_id * block_size + block_offset
                          value -1 means skip (prefix-cached token)
    """
    if slot_mapping.numel() == 0:
        return
    valid = slot_mapping != -1
    if not valid.any():
        return
    num_kv_heads, head_dim = key.shape[1], key.shape[2]
    k_flat = k_cache.view(-1, num_kv_heads, head_dim)
    v_flat = v_cache.view(-1, num_kv_heads, head_dim)
    valid_slots = slot_mapping[valid]
    k_flat[valid_slots] = key[valid]
    v_flat[valid_slots] = value[valid]


def _gather_kv_from_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather seqlen tokens of K and V from the paged cache for one sequence.

    k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: [max_blocks]  -- physical block indices (-1 = unused)
    Returns: ki, vi each [seqlen, num_kv_heads, head_dim]
    """
    block_size = k_cache.shape[1]
    ki_list = []
    vi_list = []
    remaining = seqlen
    for blk_idx in range(block_table.shape[0]):
        if remaining <= 0:
            break
        blk_id = block_table[blk_idx].item()
        if blk_id < 0:
            break
        take = min(block_size, remaining)
        ki_list.append(k_cache[blk_id, :take])
        vi_list.append(v_cache[blk_id, :take])
        remaining -= take
    return torch.cat(ki_list, dim=0), torch.cat(vi_list, dim=0)


def _expand_kv(kv: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Expand KV heads to match Q heads for GQA via repeat_interleave."""
    num_kv_heads = kv.shape[1]
    if num_kv_heads == num_q_heads:
        return kv
    repeat_factor = num_q_heads // num_kv_heads
    return kv.repeat_interleave(repeat_factor, dim=1)


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """
    Run scaled dot-product attention. Inputs/outputs are
    [seqlen, num_heads, head_dim]; internally reshaped for SDPA.
    Casts to float32 for broad dtype compatibility on CPU.
    """
    dtype = q.dtype
    # [1, num_heads, seqlen, head_dim]
    q4 = q.permute(1, 0, 2).unsqueeze(0).float()
    k4 = k.permute(1, 0, 2).unsqueeze(0).float()
    v4 = v.permute(1, 0, 2).unsqueeze(0).float()
    o4 = F.scaled_dot_product_attention(q4, k4, v4, scale=scale, is_causal=is_causal)
    # [seqlen, num_heads, head_dim]
    return o4.squeeze(0).permute(1, 0, 2).to(dtype)


def _prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_tables: torch.Tensor | None,
) -> torch.Tensor:
    """
    Variable-length prefill attention. q/k/v are packed (all sequences
    concatenated along dim 0). Process each sequence independently.
    """
    num_seqs = cu_seqlens_q.shape[0] - 1
    outputs = []
    for i in range(num_seqs):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i + 1].item()
        seqlen_k = k_end - k_start

        qi = q[q_start:q_end]  # [seqlen_q, num_q_heads, head_dim]

        if block_tables is not None:
            # Prefix cache hit: full K/V context is in the paged cache.
            # store_kvcache already wrote the new tokens; gather everything.
            ki, vi = _gather_kv_from_cache(k_cache, v_cache, block_tables[i], seqlen_k)
        else:
            ki = k[k_start:k_end]
            vi = v[k_start:k_end]

        oi = _sdpa(qi, _expand_kv(ki, qi.shape[1]), _expand_kv(vi, qi.shape[1]),
                   scale, is_causal=True)
        outputs.append(oi)

    return torch.cat(outputs, dim=0)


def _decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
) -> torch.Tensor:
    """
    Decode attention: one new token per sequence attends to its full K/V history.

    q: [num_seqs, num_q_heads, head_dim]
    Returns: [num_seqs, num_q_heads, head_dim]
    """
    num_seqs = q.shape[0]
    outputs = []
    for i in range(num_seqs):
        seqlen_k = context_lens[i].item()
        ki, vi = _gather_kv_from_cache(k_cache, v_cache, block_tables[i], seqlen_k)
        # qi: [1, num_q_heads, head_dim]
        oi = _sdpa(q[i:i + 1], _expand_kv(ki, q.shape[1]),
                   _expand_kv(vi, q.shape[1]), scale, is_causal=False)
        outputs.append(oi)

    return torch.cat(outputs, dim=0)  # [num_seqs, num_q_heads, head_dim]


class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            o = _prefill_attention(
                q, k, v, k_cache, v_cache,
                scale=self.scale,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                block_tables=context.block_tables,
            )
        else:
            o = _decode_attention(
                q, k_cache, v_cache,
                scale=self.scale,
                context_lens=context.context_lens,
                block_tables=context.block_tables,
            )
        return o
