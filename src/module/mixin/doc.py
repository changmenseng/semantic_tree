import torch
import torch.nn as nn
from typing import List, Optional

class DocMixin(nn.Module):
    
    def __init__(
        self,
        n_classes,
        hidden_dim: Optional[int] = None,
        doc_labels: Optional[List[int]] = None,
        aggregation_op: Optional[str] = 'attn',
        **kwargs
    ):
        self.aggregation_op = aggregation_op
        if aggregation_op == 'attn':
            self.sent_attn_head = nn.Linear(hidden_dim, 1)

        if doc_labels is None:
            doc_label_mask = torch.ones(n_classes)
        else:
            doc_label_mask = torch.zeros(n_classes)
            doc_label_mask[doc_labels] = 1
        self.register_buffer('doc_label_mask', doc_label_mask)
    
    def forward(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len),
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        doc_sizes: List[int],
        offsets: Optional[torch.LongTensor] = None,
        output_keys: Optional[set] = {'logits'},
        **kwargs
    ):
        outputs = super().forward( # sent_classifier
            token_seqs=token_seqs,
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs,
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            output_keys={'seq_feats', 'logits'} | output_keys,
            **kwargs)
        outputs['seq_logits'] = outputs['logits']
        if max(doc_sizes) == 1:
            doc_logits = outputs['logits']
        else:
            doc_seq_logits = outputs['logits'].split(doc_sizes)
            doc_logits = []
            if self.aggregation_op == 'attn':
                attn_scores = self.sent_attn_head(outputs['seq_feats']) # (batch_size, 1)
                doc_attn_scores = attn_scores.split(doc_sizes, 0)
                for attn_scores_in_a_doc, seq_logits_in_a_doc in zip(doc_attn_scores, doc_seq_logits):
                    attn_weights_in_a_doc = torch.softmax(attn_scores_in_a_doc, 0) # (doc_size, 1)
                    logits_of_a_doc = (attn_weights_in_a_doc * seq_logits_in_a_doc).sum(0) # (n_classes)
                    doc_logits.append(logits_of_a_doc)
            elif self.aggregation_op == 'sum':
                for seq_logits_in_a_doc in doc_seq_logits:
                    logits_of_a_doc = seq_logits_in_a_doc.sum(0) # (n_classes)
                    doc_logits.append(logits_of_a_doc)
            doc_logits = torch.stack(doc_logits, 0) # (n_docs, n_classes)

        doc_logits = doc_logits + (self.doc_label_mask.to(pos_seqs.device)[None, :] - 1) * 1e10
        outputs['logits'] = doc_logits

        return outputs