import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from transformers import AutoModel, BertModel, BertConfig

from ..registry import registry
from ..mixin import PhraseEncodingMixin
from ...embedding import get_embeddings
from ..utils import keep_keys, subtoken_pooling

@registry.register('transformer_encoder')
class TransformerEncoder(nn.Module):

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        embeddings: Optional[Union[nn.Embedding, dict]] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = 6,
        pooling_op: Optional[str] = 'cls', # sum, mean, max
        hidden_drop_p: Optional[float] = 0.4,
        subtoken_pooling_op: Optional[str] = 'mean',
        low_hidden_layer: Optional[int] = 1,
    ):
        super().__init__()
        if pretrained_path is None:
            if isinstance(embeddings, dict):
                embeddings = get_embeddings(**embeddings)
            vocab_size, hidden_dim = embeddings.weight.shape
            self.transformer = BertModel(BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_dim,
                num_hidden_layers=n_layers,
                num_attention_heads=n_heads,
                intermediate_size=4 * hidden_dim,
                max_position_embeddings=1024))
            self.transformer.embeddings.word_embeddings = embeddings
        else:
            self.transformer = AutoModel.from_pretrained(pretrained_path)

        self.pooling_op = pooling_op
        self.subtoken_pooling_op = subtoken_pooling_op
        self.low_hidden_layer = low_hidden_layer
        self.hidden_dropout = nn.Dropout(hidden_drop_p)

    def forward(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        output_keys: Optional[set] = {'seq_hiddens', 'seq_feats'},
        **kwargs
    ):
        outputs = self.transformer(
            input_ids=token_seqs,
            attention_mask=token_seq_masks,
            output_hidden_states=True)
        token_seq_hiddens = outputs.last_hidden_state # (batch_size, max_seq_len, hidden_dim)
        seq_hiddens = subtoken_pooling(token_seq_hiddens, offsets, self.subtoken_pooling_op)
        token_seq_low_hiddens = outputs.hidden_states[self.low_hidden_layer]
        seq_low_hiddens = subtoken_pooling(token_seq_low_hiddens, offsets, self.subtoken_pooling_op)

        # if kwargs['data_args']['add_special_tokens']:
        #     seq_hiddens = seq_hiddens[:, 1:-1, :]
        #     seq_low_hiddens = seq_low_hiddens[:, 1:-1, :]
        seq_hiddens *= seq_masks[..., None].float()
        seq_low_hiddens *= seq_masks[..., None].float()
        seq_hiddens = self.hidden_dropout(seq_hiddens)

        if self.pooling_op == 'cls':
            seq_feats = outputs.pooler_output
        elif self.pooling_op == 'mean':
            seq_feats = (seq_hiddens * seq_masks.unsqueeze(-1)).sum(1)
            seq_feats /= seq_masks.sum(-1, keepdims=True)
        elif self.pooling_op == 'max':
            seq_feats = torch.max(seq_hiddens + (seq_masks.unsqueeze(-1) - 1) * 1e10, 1).values
        
        outputs = {
            'seq_low_hiddens': seq_low_hiddens,
            'seq_hiddens': seq_hiddens,
            'seq_feats': seq_feats}
        return outputs

@registry.register('transformer_phrase_encoder')
class TransformerPhraseEncoder(PhraseEncodingMixin, TransformerEncoder):

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        embeddings: Optional[Union[nn.Embedding, dict]] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = 6,
        pooling_op: Optional[str] = 'mean', # sum, mean, max
        hidden_drop_p: Optional[float] = 0.4,
        subtoken_pooling_op: Optional[str] = 'mean',
        low_hidden_layer: Optional[int] = 1,
        keep_seq_feats: Optional[bool] = False
    ):
        TransformerEncoder.__init__(
            self=self,
            pretrained_path=pretrained_path,
            embeddings=embeddings,
            n_heads=n_heads,
            n_layers=n_layers,
            pooling_op=pooling_op,
            hidden_drop_p=hidden_drop_p,
            subtoken_pooling_op=subtoken_pooling_op,
            low_hidden_layer=low_hidden_layer)
        PhraseEncodingMixin.__init__(
            self=self,
            hidden_dim=self.transformer.config.hidden_size,
            pooling_op='mean',
            keep_seq_feats=keep_seq_feats)