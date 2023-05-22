import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union

from .encoder import TransformerPhraseEncoder
from ..mixin import CRFCFGMixin, DocMixin
from ..registry import registry

@registry.register('transformer_crfcfg_classifier')
class TransformerCRFCFGClassifier(CRFCFGMixin, TransformerPhraseEncoder):

    def __init__(
        self,
        n_nodes: int,
        roots: List[List[int]],
        prenodes: Union[List[List[int]], str],
        posnodes: List[int], # None means all
        rules: List[List[int]],
        pos_unary_rules: Optional[List[List[int]]] = None, # pair indicating head and one child
        pretrained_path: Optional[str] = None,
        embeddings: Optional[Union[nn.Embedding, dict]] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = 6,
        pooling_op: Optional[str] = 'cls', # sum, mean, max
        keep_seq_feats: Optional[bool] = False,
        hidden_drop_p: Optional[float] = 0.4,
        subtoken_pooling_op: Optional[str] = 'mean',
        low_hidden_layer: Optional[int] = 1,
        node_score_layers: Optional[List] = None, # None means all
        has_rule_score: Optional[bool] = True,
        has_span_score: Optional[bool] = True,
        has_root_score: Optional[bool] = True,
        has_children_score: Optional[bool] = False,
        **kwargs
    ):

        TransformerPhraseEncoder.__init__(
            self=self,
            pretrained_path=pretrained_path,
            embeddings=embeddings,
            n_heads=n_heads,
            n_layers=n_layers,
            pooling_op=pooling_op,
            hidden_drop_p=hidden_drop_p,
            subtoken_pooling_op=subtoken_pooling_op,
            low_hidden_layer=low_hidden_layer,
            keep_seq_feats=keep_seq_feats)
        CRFCFGMixin.__init__(
            self=self,
            hidden_dim=self.transformer.config.hidden_size,
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

@registry.register('transformer_doc_crfcfg_classifier')
class TransformerDocCRFCFGClassifier(DocMixin, TransformerCRFCFGClassifier):

    def __init__(
        self,
        n_nodes: int,
        roots: List[List[int]],
        prenodes: Union[List[List[int]], str],
        posnodes: List[int], # None means all
        rules: List[List[int]],
        pos_unary_rules: Optional[List[List[int]]] = None, # pair indicating head and one child
        pretrained_path: Optional[str] = None,
        embeddings: Optional[Union[nn.Embedding, dict]] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = 6,
        pooling_op: Optional[str] = 'cls', # sum, mean, max
        keep_seq_feats: Optional[bool] = False,
        hidden_drop_p: Optional[float] = 0.4,
        subtoken_pooling_op: Optional[str] = 'mean',
        low_hidden_layer: Optional[int] = 1,
        node_score_layers: Optional[List] = None, # None means all
        has_rule_score: Optional[bool] = True,
        has_span_score: Optional[bool] = True,
        has_root_score: Optional[bool] = True,
        has_children_score: Optional[bool] = False,
        doc_labels: Optional[List[int]] = None,
        aggregation_op: Optional[str] = 'sum',
        **kwargs
    ):
        TransformerCRFCFGClassifier.__init__(
            self=self,
            n_nodes=n_nodes,
            roots=roots,
            prenodes=prenodes,
            posnodes=posnodes,
            rules=rules,
            pos_unary_rules=pos_unary_rules,
            pretrained_path=pretrained_path,
            embeddings=embeddings,
            n_heads=n_heads,
            n_layers=n_layers,
            pooling_op=pooling_op,
            keep_seq_feats=keep_seq_feats,
            hidden_drop_p=hidden_drop_p,
            subtoken_pooling_op=subtoken_pooling_op,
            low_hidden_layer=low_hidden_layer,
            node_score_layers=node_score_layers,
            has_rule_score=has_rule_score,
            has_span_score=has_span_score,
            has_root_score=has_root_score,
            has_children_score=has_children_score)
        DocMixin.__init__(
            self=self,
            hidden_dim=self.transformer.config.hidden_size,
            n_classes=n_nodes,
            doc_labels=doc_labels,
            aggregation_op=aggregation_op)