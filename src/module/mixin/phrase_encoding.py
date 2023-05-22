import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..utils import triu_add, keep_keys

class PhraseEncodingMixin(nn.Module):

    def __init__(
        self, 
        hidden_dim: int,
        pooling_op: Optional[str] = 'mean',
        direction: Optional[str] = None, # only valid for lstm,
        keep_seq_feats: Optional[bool] = False,
        **kwargs
    ):
        # `pooling_op`` can be `sum`, `mean`, `max`, `substract`, `disloc_substract`
        self.pooling_op = pooling_op
        self.direction = direction
        self.keep_seq_feats = keep_seq_feats
    
    def sum(self, seq_hiddens, *args, **kwargs):
        seq_cumsum_hiddens = seq_hiddens.cumsum(1) # (batch_size, max_len, hidden_dim)
        return self.disloc_substract(seq_cumsum_hiddens, 'forward')
    
    def mean(self, seq_hiddens, *args, **kwargs):
        max_len = seq_hiddens.shape[1]
        phrase_hiddens = self.sum(seq_hiddens) # (batch_size, seq_len, seq_len, hidden_dim)
        tmp = torch.arange(max_len, device=phrase_hiddens.device)
        # import pdb; pdb.set_trace()
        return phrase_hiddens / ((tmp[None, :] - tmp[:, None]).abs()[None, :, :, None] + 1)
    
    def substract(self, seq_hiddens, direction):
        if direction == 'forward':
            return seq_hiddens.unsqueeze(1) - seq_hiddens.unsqueeze(2)
        elif direction == 'backward':
            return seq_hiddens.unsqueeze(2) - seq_hiddens.unsqueeze(1)
        
    def disloc_substract(self, seq_hiddens, direction):
        batch_size, _, hidden_dim = seq_hiddens.shape
        pad_hiddens = torch.zeros(batch_size, 1, hidden_dim, device=seq_hiddens.device)
        if direction == 'forward':
            seq_hiddens_ = torch.cat([pad_hiddens, seq_hiddens[:, :-1, :]], 1) # (batch_size, seq_len, hidden_dim)
            return seq_hiddens.unsqueeze(1) - seq_hiddens_.unsqueeze(2) # (batch_size, seq_len, seq_len, hidden_dim)
        elif direction == 'backward':
            seq_hiddens_ = torch.cat([seq_hiddens[:, 1:, :], pad_hiddens], 1) # (batch_size, seq_len, hidden_dim)
            return seq_hiddens.unsqueeze(2) - seq_hiddens_.unsqueeze(1) # (batch_size, seq_len, seq_len, hidden_dim)

    def max(self, seq_hiddens, *args, **kwargs):
        batch_size, max_len, seq_len = seq_hiddens.shape
        phrase_hiddens = torch.zeros(batch_size, seq_len, seq_len, hidden_dim, device=seq_hiddens.device)
        phrase_hiddens = triu_add(phrase_hiddens, seq_hiddens, 1) # (batch_size, seq_len, seq_len, hidden_dim)
        for i in range(1, max_len):
            triu_indices = [list(range(max_len  - i)), list(range(i, max_len))]
            compared_hiddens = []
            for row_id, col_id in zip(*triu_indices): # has `max_len - i` items in the layer
                compared_hiddens_ = torch.stack([
                    phrase_hiddens[:, row_id, col_id - 1, :],
                    phrase_hiddens[:, row_id + 1, col_id, :]
                ], 1) # (batch_size, 2, hidden_dim)
                compared_hiddens.append(compared_hiddens_)
            compared_hiddens = torch.stack(compared_hiddens, 1) # (batch_size, max_len - i, 2, hidden_dim)
            layer_phrase_hiddens = compared_hiddens.max(2).values # (batch_size, max_len - i, hidden_dim)
            phrase_hiddens = triu_add(phrase_hiddens, layer_phrase_hiddens, 1)
        return phrase_hiddens


    def forward(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        output_keys: Optional[set] = {'phrase_hiddens'},
        **kwargs
    ):
        outputs = super().forward(
            token_seqs=token_seqs,
            pos_seqs=pos_seqs,
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            output_keys={'seq_hiddens', 'seq_feats'} | output_keys,
            **kwargs)
        seq_hiddens = outputs['seq_hiddens']
        op_func = getattr(self, self.pooling_op)
        if self.direction == 'both' and self.pooling_op in {'substract', 'disloc_substract'}:
            seq_forward_hiddens = seq_hiddens[:, :, :int(self.hidden_dim / 2)]
            phrase_forward_hiddens = op_func(seq_forward_hiddens, 'forward')
            seq_backward_hiddens = seq_hiddens[:, :, int(self.hidden_dim / 2):]
            phrase_backward_hiddens = op_func(seq_backward_hiddens, 'backward')
            phrase_hiddens = torch.cat([phrase_forward_hiddens, phrase_backward_hiddens], -1)
        else:
            phrase_hiddens = op_func(seq_hiddens, self.direction) # (batch_size, max_seq_len, max_seq_len, hidden_dim)

        if self.keep_seq_feats:
            seq_feats = outputs['seq_feats'] # (batch_size, hidden_dim)
            seq_end_masks = (seq_masks - F.pad(seq_masks, (1,1))[:,2:])[..., None] # (batch_size, max_seq_len, 1)
            phrase_hiddens[:, 0, ...] = phrase_hiddens[:, 0, ...] * (1 - seq_end_masks) + seq_feats[:, None, :] * seq_end_masks

        outputs['phrase_hiddens'] = phrase_hiddens
        return keep_keys(outputs, output_keys)