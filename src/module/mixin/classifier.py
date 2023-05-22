import torch
import torch.nn as nn
from typing import Optional, List, Union

from ..utils import triu_add, logsumexp_lastdims, max_lastdims, keep_keys

class ClassifierMixin(nn.Module):

    def __init__(
        self,
        hidden_dim,
        n_classes,
        sent_labels: Optional[List[int]] = None, # sometimes, you might need that some labels are never predicted.
        **kwargs
    ):
        self.classify_head = nn.Linear(hidden_dim, n_classes)
        
        if sent_labels is None:
            sent_label_mask = torch.ones(n_classes)
        else:
            sent_label_mask = torch.zeros(n_classes)
            sent_label_mask[sent_labels] = 1
        self.register_buffer('sent_label_mask', sent_label_mask)

    def forward(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        output_keys: Optional[set] = {'logits'},
        **kwargs
    ):
        outputs = super().forward(
            token_seqs=token_seqs,
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs,
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            output_keys={'seq_feats'} | output_keys,
            **kwargs)
        logits = self.classify_head(outputs['seq_feats']) # (batch_size, n_classes)
        logits = logits + (self.sent_label_mask.to(pos_seqs.device)[None, :] - 1) * 1e10
        outputs['logits'] = logits
        return keep_keys(outputs, output_keys)

    @torch.no_grad()
    def predict(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        outputs = self.forward(
            token_seqs=token_seqs,
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs,
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            **kwargs)
        return outputs