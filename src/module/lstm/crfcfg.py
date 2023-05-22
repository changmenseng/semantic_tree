import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union

from .encoder import LSTMPhraseEncoder
from ..mixin import CRFCFGMixin, DocMixin
from ..registry import registry

@registry.register('lstm_crfcfg_classifier')
class LSTMCRFCFGClassifier(CRFCFGMixin, LSTMPhraseEncoder):
    def __init__(
        self,
        n_nodes: int,
        roots: List[List[int]],
        prenodes: Union[List[List[int]], str],
        posnodes: List[int], 
        rules: List[List[int]],
        pos_unary_rules: List[List[int]],
        embeddings: Union[nn.Embedding, Dict],
        n_postags: Optional[int] = 46,
        bidirectional: Optional[bool] = True,
        hidden_dim: Optional[int] = 256,
        emb_drop_p: Optional[float] = 0.2,
        hidden_drop_p: Optional[float] = 0.4,
        pooling_op: Optional[str] = 'disloc_substract',
        keep_seq_feats: Optional[bool] = False,
        node_score_layers: Optional[List] = [1, 2, -1],
        has_rule_score: Optional[bool] = True,
        has_span_score: Optional[bool] = True,
        has_root_score: Optional[bool] = True,
        has_children_score: Optional[bool] = True
    ):
        LSTMPhraseEncoder.__init__(
            self=self,
            embeddings=embeddings,
            n_postags=n_postags,
            bidirectional=bidirectional,
            hidden_dim=hidden_dim,
            emb_drop_p=emb_drop_p,
            hidden_drop_p=hidden_drop_p,
            pooling_op=pooling_op,
            keep_seq_feats=keep_seq_feats)
        CRFCFGMixin.__init__(
            self=self,
            hidden_dim=hidden_dim,
            n_nodes=n_nodes,
            roots=roots,
            prenodes=prenodes,
            posnodes=posnodes,
            rules=rules,
            pos_unary_rules=pos_unary_rules,
            node_score_layers=node_score_layers,
            has_rule_score=has_rule_score,
            has_span_score=has_span_score,
            has_root_score=has_root_score,
            has_children_score=has_children_score)

@registry.register('lstm_doc_crfcfg_classifier')
class LSTMDocCRFCFGClassifier(DocMixin, LSTMCRFCFGClassifier):

    def __init__(
        self,
        n_nodes: int,
        roots: List[List[int]],
        prenodes: Union[List[List[int]], str],
        posnodes: List[int], 
        rules: List[List[int]],
        pos_unary_rules: List[List[int]],
        embeddings: Union[nn.Embedding, Dict],
        n_postags: Optional[int] = 46,
        bidirectional: Optional[bool] = True,
        hidden_dim: Optional[int] = 256,
        emb_drop_p: Optional[float] = 0.2,
        hidden_drop_p: Optional[float] = 0.4,
        pooling_op: Optional[str] = 'disloc_substract',
        keep_seq_feats: Optional[bool] = False,
        node_score_layers: Optional[List] = [1, 2, -1],
        has_rule_score: Optional[bool] = True,
        has_span_score: Optional[bool] = True,
        has_root_score: Optional[bool] = True,
        has_children_score: Optional[bool] = True,
        doc_labels: Optional[List[int]] = None,
        aggregation_op: Optional[str] = 'sum'
    ):
        LSTMCRFCFGClassifier.__init__(
            self=self,
            n_nodes=n_nodes,
            roots=roots,
            prenodes=prenodes,
            posnodes=posnodes,
            rules=rules,
            pos_unary_rules=pos_unary_rules,
            embeddings=embeddings,
            n_postags=n_postags,
            bidirectional=bidirectional,
            hidden_dim=hidden_dim,
            emb_drop_p=emb_drop_p,
            hidden_drop_p=hidden_drop_p,
            pooling_op=pooling_op,
            keep_seq_feats=keep_seq_feats,
            node_score_layers=node_score_layers,
            has_rule_score=has_rule_score,
            has_span_score=has_span_score,
            has_root_score=has_root_score,
            has_children_score=has_children_score)
        DocMixin.__init__(
            self=self,
            hidden_dim=hidden_dim,
            n_classes=n_nodes,
            doc_labels=doc_labels,
            aggregation_op=aggregation_op)