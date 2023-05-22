import torch
import torch.nn as nn
from typing import Optional, List

from ..utils import keep_keys

class WithLexiconMixin(nn.Module): # on top of ClassifierMixin and CRFCFGMixin

    def __init__(
        self,
        n_classes: int, # total len of label set, including word labels and sent labels
        hidden_dim: int,
        embedding_dim: int,
        word_labels: Optional[List[int]] = None,
        contextual: Optional[bool] = False,
    ):
        self.n_classes = n_classes
        self.contextual = contextual
        self.word_labels = list(range(n_classes)) if word_labels is None else word_labels
        self.word_label_mask = torch.zeros(n_classes)
        self.word_label_mask[word_labels] = 1

        word_classify_head_layers = []
        if (not contextual) and embedding_dim != hidden_dim:
            word_classify_head_layers.append(nn.Linear(embedding_dim, hidden_dim))
            word_classify_head_layers.append(nn.Tanh())
        try:
            classify_head = getattr(self, 'classify_head')
        except AttributeError:
            classify_head = getattr(self, 'node_head')
        word_classify_head_layers.append(classify_head)
        self.word_classify_head = nn.Sequential(*word_classify_head_layers)
    
    def forward(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        output_keys: Optional[set] = {'logits', 'logit_seqs'},
        **kwargs
    ):
        if self.contextual:
            required_key = 'seq_hiddens'
        else:
            required_key = 'seq_embeddings'
        outputs = super().forward(
            token_seqs=token_seqs,
            pos_seqs=pos_seqs,
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            output_keys={required_key} | output_keys,
            **kwargs)
        logit_seqs = self.word_classify_head(outputs[required_key]) # (batch_size, max_seq_len, n_classes)
        logit_seqs = logit_seqs + (self.word_label_mask.to(pos_seqs.device)[None, None, :] - 1) * 1e10

        outputs['logit_seqs'] = logit_seqs
        return keep_keys(outputs, output_keys)