import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM
from typing import Optional, Union, Dict

from ..registry import registry
from ..mixin import PhraseEncodingMixin
from ...embedding import get_embeddings
from ..utils import triu_add, keep_keys, subtoken_pooling

@registry.register('lstm_encoder')
class LSTMEncoder(nn.Module):

    def __init__(
        self,
        embeddings: Union[nn.Embedding, Dict],
        n_postags: Optional[int] = 46,
        bidirectional: Optional[bool] = True,
        hidden_dim: Optional[int] = 256,
        emb_drop_p: Optional[float] = 0.2,
        hidden_drop_p: Optional[float] = 0.4,
        pooling_op: Optional[float] = 'last'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pooling_op = pooling_op
        if isinstance(embeddings, nn.Embedding):
            self.embeddings = embeddings
        elif isinstance(embeddings, dict):
            self.embeddings = get_embeddings(**embeddings)
        vocab_size, embedding_dim = self.embeddings.weight.shape
        self.pos_embeddings = nn.Embedding.from_pretrained(
            torch.zeros(n_postags, embedding_dim), False)
        if bidirectional:
            hidden_dim = int(hidden_dim / 2)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=bidirectional)
        self.emb_dropout = nn.Dropout(emb_drop_p)
        self.hidden_dropout = nn.Dropout(hidden_drop_p)

    def forward(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        output_keys: Optional[set] = {'seq_hiddens', 'seq_feats'},
        **kwargs
    ):
        assert kwargs['data_args']['add_special_tokens'] is False # tired to support special tokens...
        batch_size = token_seqs.shape[0]
        token_seq_embeddings = self.embeddings(token_seqs) # (batch_size, max_token_seq_len, embedding_dim)
        word_seq_embeddings = subtoken_pooling(token_seq_embeddings, offsets) * seq_masks[..., None].float() # (batch_size, max_seq_len, embedding_dim)
        pos_seq_embeddings = self.pos_embeddings(pos_seqs) * pos_seq_masks[..., None].float() # (batch_size, max_seq_len, embedding_dim)
        seq_embeddings = word_seq_embeddings + pos_seq_embeddings
        seq_embeddings = self.emb_dropout(seq_embeddings)
        
        packed_seqs = nn.utils.rnn.pack_padded_sequence(
            input=seq_embeddings,
            lengths=seq_masks.sum(-1).cpu(),
            batch_first=True,
            enforce_sorted=False) # packed_seqs: PackedSequence object
        packed_seqs, (seq_final_hiddens, _) = self.lstm(packed_seqs)
        seq_final_hiddens = seq_final_hiddens.transpose(0,1).reshape(batch_size, -1) # (batch_size, hidden_dim)
        seq_hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=packed_seqs,
            batch_first=True)
        seq_hiddens = self.hidden_dropout(seq_hiddens)

        if self.pooling_op == 'max':
            seq_feats = torch.max(seq_hiddens + (seq_masks.unsqueeze(-1) - 1) * 1e10, 1).values
        elif self.pooling_op == 'mean':
            seq_feats = (seq_hiddens * seq_masks.unsqueeze(-1)).sum(1)
            seq_feats /= seq_masks.sum(-1, keepdims=True)
        else: # by default the feature of the sentence is the final hidden states
            seq_feats = seq_final_hiddens

        outputs = {
            'seq_hiddens': seq_hiddens,
            'seq_feats': seq_feats
        }
        return keep_keys(outputs, output_keys)


@registry.register('lstm_phrase_encoder')
class LSTMPhraseEncoder(PhraseEncodingMixin, LSTMEncoder):

    def __init__(
        self,
        embeddings: Union[nn.Embedding, Dict],
        n_postags: Optional[int] = 46,
        bidirectional: Optional[bool] = True,
        hidden_dim: Optional[int] = 256,
        emb_drop_p: Optional[float] = 0.2,
        hidden_drop_p: Optional[float] = 0.4,
        pooling_op: Optional[str] = 'disloc_substract',
        keep_seq_feats: Optional[bool] = False
    ):
        LSTMEncoder.__init__(
            self=self,
            embeddings=embeddings,
            n_postags=n_postags,
            bidirectional=bidirectional,
            hidden_dim=hidden_dim,
            emb_drop_p=emb_drop_p,
            hidden_drop_p=hidden_drop_p,
            pooling_op=pooling_op)
        direction = 'both' if bidirectional else 'forward'
        PhraseEncodingMixin.__init__(
            self=self,
            hidden_dim=hidden_dim,
            pooling_op=pooling_op,
            direction=direction,
            keep_seq_feats=keep_seq_feats)