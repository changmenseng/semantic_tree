import torch
import torch.nn as nn
from typing import Optional, List, Dict, Union

from .encoder import LSTMEncoder
from ..mixin import ClassifierMixin, DocMixin
from ..registry import registry

@registry.register('lstm_classifier')
class LSTMClassifier(ClassifierMixin, LSTMEncoder):

    def __init__(
        self,
        n_classes: int,
        embeddings: Union[nn.Embedding, Dict],
        sent_labels: Optional[List[int]] = None,
        n_postags: Optional[int] = 48,
        bidirectional: Optional[bool] = True,
        hidden_dim: Optional[int] = 256,
        emb_drop_p: Optional[float] = 0.2,
        hidden_drop_p: Optional[float] = 0.4,
        pooling_op: Optional[float] = 'last'
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
        ClassifierMixin.__init__(
            self=self,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            sent_labels=sent_labels)

@registry.register('lstm_doc_classifier')
class LSTMDocClassifier(DocMixin, LSTMClassifier):

    def __init__(
        self,
        n_classes: int,
        embeddings: Union[nn.Embedding, Dict],
        sent_labels: Optional[List[int]] = None,
        doc_labels: Optional[List[int]] = None,
        n_postags: Optional[int] = 46,
        bidirectional: Optional[bool] = True,
        hidden_dim: Optional[int] = 256,
        emb_drop_p: Optional[float] = 0.2,
        hidden_drop_p: Optional[float] = 0.4,
        pooling_op: Optional[float] = 'last',
        aggregation_op: Optional[float] = 'sum'
    ):
        LSTMClassifier.__init__(
            self=self,
            n_classes=n_classes,
            embeddings=embeddings,
            sent_labels=sent_labels,
            n_postags=n_postags,
            bidirectional=bidirectional,
            hidden_dim=hidden_dim,
            emb_drop_p=emb_drop_p,
            hidden_drop_p=hidden_drop_p,
            pooling_op=pooling_op)
        DocMixin.__init__(
            self=self,
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            doc_labels=doc_labels,
            aggregation_op=aggregation_op)