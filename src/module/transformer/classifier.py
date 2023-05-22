import torch
import torch.nn as nn
from typing import Optional, Union, List

from .encoder import TransformerEncoder, TransformerPhraseEncoder
from ..registry import registry
from ..mixin import ClassifierMixin, DocMixin

@registry.register('transformer_classifier')
class TransformerClassifier(ClassifierMixin, TransformerEncoder):

    def __init__(
        self,
        n_classes: int,
        pretrained_path: Optional[str] = None,
        embeddings: Optional[Union[nn.Embedding, dict]] = None,
        sent_labels: Optional[List[int]] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = 6,
        pooling_op: Optional[str] = 'cls', # sum, mean, max,
        subtoken_pooling_op: Optional[str] = 'mean',
        hidden_drop_p: Optional[float] = 0.4,
        **kwargs
    ):
        TransformerEncoder.__init__(
            self=self,
            pretrained_path=pretrained_path,
            embeddings=embeddings,
            n_heads=n_heads,
            n_layers=n_layers,
            pooling_op=pooling_op,
            hidden_drop_p=hidden_drop_p,
            subtoken_pooling_op=subtoken_pooling_op)
        ClassifierMixin.__init__(
            self=self,
            hidden_dim=self.transformer.config.hidden_size,
            n_classes=n_classes,
            sent_labels=sent_labels)

@registry.register('transformer_doc_classifier')
class TransformerDocClassifier(DocMixin, TransformerClassifier):
    def __init__(
        self,
        n_classes: int,
        pretrained_path: Optional[str] = None,
        embeddings: Optional[Union[nn.Embedding, dict]] = None,
        sent_labels: Optional[List[int]] = None,
        doc_labels: Optional[List[int]] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = 6,
        pooling_op: Optional[str] = 'cls', # sum, mean, max
        hidden_drop_p: Optional[float] = 0.4,
        subtoken_pooling_op: Optional[str] = 'mean',
        aggregation_op: Optional[str] = 'attn',
        **kwargs
    ):
        TransformerClassifier.__init__(
            self=self,
            n_classes=n_classes,
            pretrained_path=pretrained_path,
            embeddings=embeddings,
            sent_labels=sent_labels,
            n_heads=n_heads,
            n_layers=n_layers,
            pooling_op=pooling_op,
            hidden_drop_p=hidden_drop_p,
            subtoken_pooling_op=subtoken_pooling_op)
        DocMixin.__init__(
            self=self,
            hidden_dim=self.transformer.config.hidden_size,
            n_classes=n_classes,
            doc_labels=doc_labels,
            aggregation_op=aggregation_op)