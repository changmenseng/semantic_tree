import torch
import torch.nn as nn
from typing import Optional
from transformer_embedder.utils import batched_span_select

def keep_keys(input_dict, keys=set()):
    output_dict = dict()
    for key in keys:
        try: output_dict[key] = input_dict[key]
        except KeyError: pass
    return output_dict

def triu_add(
    src: torch.tensor, # (batch_size, n, n, m)
    v: torch.tensor, # (batch_size, k<=n, m)
    dim: int # left dim of the diag mat.
):
    assert src.shape[dim] == src.shape[dim + 1]
    n = src.shape[dim]
    k = v.shape[dim]
    # import pdb; pdb.set_trace()
    for i in range(src.ndim - dim - 2): 
        v = torch.movedim(v, -1, 0)
    triu_v = torch.diag_embed(v, n - k)
    for i in range(src.ndim - dim - 2): 
        triu_v = torch.movedim(triu_v, 0, -1)
    return src + triu_v

def triu_product(
    src: torch.tensor, # (batch_size, n, n, m)
    v: torch.tensor, # (batch_size, k<=n, m)
    dim: int # left dim of the diag mat.
):
    assert src.shape[dim] == src.shape[dim + 1]
    n = src.shape[dim]
    k = v.shape[dim]
    # import pdb; pdb.set_trace()
    for i in range(src.ndim - dim - 2): 
        v = torch.movedim(v, -1, 0)
    triu_v = torch.diag_embed(v, n - k)
    for i in range(src.ndim - dim - 2): 
        triu_v = torch.movedim(triu_v, 0, -1)
    triu_v += (triu_v == 0).float()
    return src * triu_v

def logsumexp_lastdims(
    x: torch.tensor,
    n: int
):
    for i in range(n):
        x = x.logsumexp(-1)
    return x

def max_lastdims(
    x: torch.tensor,
    n: int
):
    flattened_x = x.flatten(-n, -1)
    values, indices = flattened_x.max(-1) # (batch_size, seq_len - i, n_nonterminals)
    coordinates = []
    for i in range(n):
        coordinates_ = (indices % x.shape[- i - 1]).long() # (batch_size, seq_len - i, n_nonterminals)
        indices = ((indices - coordinates_) / x.shape[- i - 1]).long()
        coordinates.append(coordinates_)
    coordinates.reverse()
    coordinates = torch.stack(coordinates, -1)
    return values, coordinates

def get_parameters_by_prefix(
    module: nn.Module,
    prefix: Optional[str] = '',
    mode: Optional[str] = 'include' # 'include' or 'exclude' or 'all'
):
    if mode == 'include':
        for key, parameter in module.named_parameters():
            if key.startswith(prefix): yield parameter
    elif mode == 'exclude':
        for key, parameter in module.named_parameters():
            if not key.startswith(prefix): yield parameter

def subtoken_pooling(token_seq_embeddings, offsets=None, op='mean'):
    if offsets is None: return token_seq_embeddings
    span_embeddings, span_mask = batched_span_select(token_seq_embeddings.contiguous(), offsets)
    # # (batch_size, max_word_len, max_span_len, embedding_dim), # (batch_size, max_word_len, max_span_len)
    span_mask = span_mask.unsqueeze(-1) # (batch_size, max_word_len, max_span_len, 1)
    span_embeddings *= span_mask
    
    if op == 'mean':
        span_lens = span_mask.sum(2) # (batch_size, max_word_len, 1)
        word_seq_embeddings = span_embeddings.sum(2) / torch.clamp_min(span_lens, 1) # # (batch_size, max_word_len, embedding_dim)
        word_seq_embeddings *= (span_lens != 0).float()
    elif op == 'left':
        word_seq_embeddings = span_embeddings[:, :, 0, :]

    # word_seq_embeddings = TransformerEmbedder.merge_subtoken_embeddings(span_embeddings, span_mask)
    return word_seq_embeddings