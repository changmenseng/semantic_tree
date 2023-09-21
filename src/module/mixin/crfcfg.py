import torch
import torch.nn as nn
from typing import Optional, List, Union

from ..utils import triu_add, logsumexp_lastdims, max_lastdims, keep_keys

def get_triu_lr_scores(scores, i, triu_left_indices, triu_right_indices):
    batch_size, _, _, n_nodes = scores.shape
    triu_left_scores = scores.gather(2, triu_left_indices[None,:,:,None].repeat(batch_size,1,1,n_nodes)) # (batch_size, max_len-i, i, n_nodes)
    triu_right_scores = scores[:, :, i:, :].gather(1, triu_right_indices[None,:,:,None].repeat(batch_size,1,1,n_nodes)).transpose(1,2) # (batch_size, max_len-i, i, n_nodes)
    return (triu_left_scores, triu_right_scores)

def get_triu_scores(scores, triu_indices):
    batch_size, _, _, n_nodes = scores.shape
    triu_scores = scores.gather(2, triu_indices[None,:,:,None].repeat(batch_size,1,1,n_nodes)).squeeze(2)
    return triu_scores

class CRFCFGMixin(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_nodes: int,
        roots: Optional[List[int]] = None, # None means all
        prenodes: Optional[List[int]] = None, # None means all
        posnodes: Optional[List[int]] = None, # None means all
        rules: Optional[List[List[int]]] = None, # triple indicating head and two children
        pos_unary_rules: Optional[List[List[int]]] = None, # pair indicating head and one child
        node_score_layers: Optional[List] = None, # None means all
        has_rule_score: Optional[bool] = True,
        has_span_score: Optional[bool] = True,
        has_root_score: Optional[bool] = True,
        has_children_score: Optional[bool] = True,
        **kwargs
    ):

        # flags to specify a potential function
        self.has_rule_score = has_rule_score
        self.has_span_score = has_span_score
        self.has_root_score = has_root_score
        self.has_children_score = has_children_score

        self.n_nodes = n_nodes
        self.node_score_layers = node_score_layers
        
        self.posnode_head = nn.Linear(hidden_dim, n_nodes)
        self.node_head = nn.Linear(hidden_dim, n_nodes)

        if has_span_score:
            self.span_head = nn.Linear(hidden_dim, 1)

        if roots is None:
            root_mask = torch.ones(n_nodes)
        else:
            root_mask = torch.zeros(n_nodes)
            root_mask[roots] = 1
        self.register_buffer('root_mask', root_mask)

        if posnodes is None:
            posnode_mask = torch.ones(n_nodes)
        else:
            posnode_mask = torch.zeros(n_nodes)
            posnode_mask[posnodes] = 1
        self.register_buffer('posnode_mask', posnode_mask)

        if rules is None:
            rule_mask = torch.ones(n_nodes, n_nodes, n_nodes)
        else:
            rule_mask = torch.zeros(n_nodes, n_nodes, n_nodes)
            rule_mask[torch.tensor(rules).T.tolist()] = 1
        self.register_buffer('rule_mask', rule_mask)

        self.rule_scores = nn.Parameter(
            torch.zeros(n_nodes, n_nodes, n_nodes), 
            requires_grad=has_rule_score)
        
        if pos_unary_rules is None:
            pos_unary_rule_mask = torch.ones(n_nodes, n_nodes)
        else:
            pos_unary_rule_mask = torch.zeros(n_nodes, n_nodes)
            pos_unary_rule_mask[torch.tensor(pos_unary_rules).T.tolist()] = 1
        self.register_buffer('pos_unary_rule_mask', pos_unary_rule_mask)

        self.pos_unary_rule_scores = nn.Parameter(
            torch.zeros(n_nodes, n_nodes),
            requires_grad=has_rule_score)

    def get_node_scores(
        self,
        phrase_hiddens: torch.tensor, # (batch_size, max_len, max_len, hidden_dim)
        seq_hiddens: torch.tensor, # (batch_size, max_len, hidden_dim)
        seq_masks: torch.LongTensor # (batch_size, max_len)
    ):
        device = seq_masks.device
        batch_size, max_len = seq_masks.shape

        posnode_scores = self.posnode_head(seq_hiddens) # (batch_size, max_len, n_nodes)
        posnode_scores += (self.posnode_mask[None, None, :] - 1) * 1e10
        node_scores = self.node_head(phrase_hiddens) # (batch_size, max_len, max_len, n_nodes)

        if self.node_score_layers is not None:
            seq_lens = seq_masks.sum(-1) # (batch_size,)
            node_score_masks = torch.zeros_like(node_scores[:, :, :, 0]) # (batch_size, max_len, max_len)
            for i in self.node_score_layers:
                if i >= 0:
                    node_score_masks += torch.diag(torch.ones(max_len - i, device=device), i).unsqueeze(0) # (1, max_len, max_len)
                else:
                    torch.use_deterministic_algorithms(False)
                    for j in range(-i):
                        node_score_masks[range(batch_size), [j] * batch_size, (seq_lens + i + j).tolist()] = 1
            node_scores *= node_score_masks.unsqueeze(-1)
        # node_scores = triu_add(
        #     src=node_scores, dim=1,
        #     v=(self.prenode_mask[None, None, :].repeat(batch_size, max_len, 1) - 1) * 1e10)
        return posnode_scores, node_scores

    def get_scores(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        output_keys: Optional[set] = {"posnode_scores", "node_scores", "span_scores"},
        **kwargs
    ):
        outputs = super().forward(
            token_seqs=token_seqs, 
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs, 
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            output_keys={'seq_low_hiddens', 'seq_hiddens', 'phrase_hiddens'} | output_keys,
            **kwargs) # (batch_size, max_len, max_len, hidden_dim)
        
        seq_low_hiddens = outputs.get('seq_low_hiddens', outputs['seq_hiddens'])
        phrase_hiddens = outputs['phrase_hiddens']
        posnode_scores, node_scores = self.get_node_scores(
            seq_masks=seq_masks,
            phrase_hiddens=phrase_hiddens,
            seq_hiddens=seq_low_hiddens) # (batch_size, max_len, max_len, n_nodes)
        if self.has_span_score:
            span_scores = self.span_head(phrase_hiddens).squeeze(-1) # (batch_size, max_len, max_len)
        else:
            span_scores = None

        outputs['posnode_scores'] = posnode_scores
        outputs['node_scores'] = node_scores
        outputs['span_scores'] = span_scores

        return keep_keys(outputs, output_keys)

    def inner_looper(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        output_keys: Optional[set] = set(),
        greedy_on_words: Optional[bool] = False,
        **kwargs
    ):
        batch_size, max_len = pos_seqs.shape
        device = pos_seqs.device

        outputs = self.get_scores(
            token_seqs=token_seqs, 
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs, 
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            output_keys={"node_scores", "span_scores", "posnode_scores"} | output_keys,
            **kwargs)
        
        posnode_scores = outputs['posnode_scores'] # (batch_size, max_len, n_nodes)
        node_scores = outputs['node_scores'] # (batch_size, max_len, max_len, n_nodes)
        span_scores = outputs['span_scores'] # (batch_size, max_len, max_len)

        if greedy_on_words and not self.training:
            posnode_scores = (torch.softmax(posnode_scores / 1e-10, -1) - 1) * 1e20
            # This is very tricky. 1e20 is bigger than 1e10 which is used to simulate infinity in other places of the code. 
            # Here we use 1e20 to allow rules not in the rule list to handle some extrem cases.
            # non-greedy prediction of words to a slight extent to handle some extrem cases.
        prenode_scores = torch.diagonal(node_scores, 0, 1, 2).transpose(1, 2) # (batch_size, max_len, n_nodes)
        
        pos_unary_rule_scores = self.pos_unary_rule_scores + (self.pos_unary_rule_mask - 1) * 1e15 # (n_nodes, n_nodes)
        first_layer_summed_scores = pos_unary_rule_scores[None, None, :, :] + prenode_scores[:, :, :, None] + posnode_scores[:, :, None, :]
        first_layer_inner_scores = yield first_layer_summed_scores
        # (batch_size, max_len, n_nodes)

        inner_scores = triu_add(torch.zeros_like(node_scores), first_layer_inner_scores, 1)
        # inner_scores = node_scores * torch.eye(max_len, device=device)[None,:,:,None] # (batch_size, max_len, max_len, n_nodes)
        compute_skeleton_logZ = 'skeleton_logZ' in output_keys and self.has_span_score and self.training
        if compute_skeleton_logZ:
            skeleton_inner_scores = span_scores * torch.eye(max_len, device=device)[None,...] # (batch_size, max_len, max_len)

        for i in range(1, max_len):
            # triu_indices_ = [list(range(max_len  - i)), list(range(i, max_len))]
            n_trius = max_len - i
            triu_indices = torch.arange(i, max_len, device=device)[:, None]
            triu_left_indices = torch.arange(i, device=device)[None, :].repeat(n_trius, 1) + torch.arange(n_trius, device=device)[:, None]
            triu_right_indices = torch.arange(n_trius, max_len, device=device)[None, :].repeat(n_trius, 1) - torch.arange(n_trius, 0, -1, device=device)[:, None] + 1
            triu_right_indices = triu_right_indices.T

            summed_scores = torch.zeros(batch_size, max_len - i, self.n_nodes, i, self.n_nodes, self.n_nodes, device=device)
            rule_scores = self.rule_scores + (self.rule_mask - 1) * 1e10
            summed_scores += rule_scores[None, None, :, None, :, :]

            triu_pair_scores = get_triu_lr_scores(inner_scores, i, triu_left_indices, triu_right_indices)  # (batch_size, max_len-i, i, n_nodes) each
            triu_pair_scores = triu_pair_scores[0].unsqueeze(-1) + triu_pair_scores[1].unsqueeze(-2) # (batch_size, max_len - i, i, n_nodes, n_nodes)

            summed_scores += triu_pair_scores[:, :, None, :, :, :] # (batch_size, max_len - i, 1,              i, n_nodes, n_nodes)

            if compute_skeleton_logZ:
                skeleton_summed_scores = get_triu_lr_scores(skeleton_inner_scores[..., None], i, triu_left_indices, triu_right_indices) # (batch_size, max_len-i, i, 1) each
                skeleton_summed_scores = (skeleton_summed_scores[0] + skeleton_summed_scores[1]).squeeze(-1)


            if self.has_root_score:
                triu_root_scores = get_triu_scores(node_scores, triu_indices)
                summed_scores += triu_root_scores[:, :, :, None, None, None] # (batch_size, max_len - i, n_nodes, 1, 1,              1)

            
            if self.has_children_score:
                triu_children_scores = get_triu_lr_scores(node_scores, i, triu_left_indices, triu_right_indices)
                triu_children_scores = triu_children_scores[0].unsqueeze(-1) + triu_children_scores[1].unsqueeze(-2)

                summed_scores += triu_children_scores[:, :, None, :, :, :]  # (batch_size, max_len - i, 1,              i, n_nodes, n_nodes)
            
            if self.has_span_score:
                triu_span_scores = get_triu_scores(span_scores[..., None], triu_indices)
                summed_scores += triu_span_scores[:, :, :, None, None, None]
                if compute_skeleton_logZ:
                    skeleton_summed_scores += triu_span_scores # (batch_size, max_len - i, i)

            triu_inner_scores = yield summed_scores # might be sum (inner algorithm) or max (CKY)
            inner_scores = triu_add(inner_scores, triu_inner_scores, 1) # (batch_size, max_len - i, n_nodes)

            if compute_skeleton_logZ:
                triu_skeleton_inner_scores = skeleton_summed_scores.logsumexp(-1) # (batch_size, max_len - i)
                skeleton_inner_scores = triu_add(skeleton_inner_scores.unsqueeze(-1), triu_skeleton_inner_scores.unsqueeze(-1), 1).squeeze(-1)

        torch.use_deterministic_algorithms(False)
        logits = torch.gather(
            input=inner_scores[:,0,:,:], dim=1, # (batch_size, max_len, n_nodes)
            index=seq_masks.sum(-1)[:, None, None].repeat(1, 1, self.n_nodes)-1).squeeze(1) # (batch_size, n_nodes)
        logits += (self.root_mask[None, :] - 1) * 1e10
        outputs['logits'] = logits
        
        if compute_skeleton_logZ:
            skeleton_logZ = torch.gather(
                input=skeleton_inner_scores[:,0,:], dim=1,
                index=seq_masks.sum(-1)[:, None]-1).squeeze(1) # (batch_size)
            outputs['skeleton_logZ'] = skeleton_logZ

        return outputs

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
    ): # inner algorithm
        _, max_len = pos_seqs.shape

        looper = self.inner_looper(
            token_seqs=token_seqs, 
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs, 
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            output_keys=output_keys,
            **kwargs)
        first_layer_summed_scores = looper.send(None) # (batch_size, max_len, n_nodes, n_nodes)
        first_layer_inner_scores = torch.logsumexp(first_layer_summed_scores, -1)

        summed_scores = looper.send(first_layer_inner_scores)
        triu_inner_scores = logsumexp_lastdims(summed_scores, 3)

        for i in range(2, max_len):
            summed_scores = looper.send(triu_inner_scores)
            triu_inner_scores = logsumexp_lastdims(summed_scores, 3)
        try:
            looper.send(triu_inner_scores)
        except StopIteration as e:
            outputs = e.args[0]

        return keep_keys(outputs, output_keys)
    
    @torch.no_grad()
    def decode(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        greedy_on_words: Optional[bool] = False,
        **kwargs
    ): # CKY decoding
        batch_size, max_len = pos_seqs.shape
        device = pos_seqs.device

        max_indices = torch.zeros(batch_size, max_len, max_len, self.n_nodes, 3, dtype=torch.long, device=device)

        looper = self.inner_looper(
            token_seqs=token_seqs, 
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs, 
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            greedy_on_words=greedy_on_words,
            **kwargs)

        first_layer_maxed_scores = looper.send(None) # (batch_size, max_len, n_nodes, n_nodes)
        first_layer_inner_scores, first_layer_max_indices = torch.max(first_layer_maxed_scores, -1)

        maxed_scores = looper.send(first_layer_inner_scores) # (batch_size, max_len - i, n_nodes, i, n_nodes, n_nodes)
        triu_inner_scores, triu_max_indices = max_lastdims(maxed_scores, 3)
        # (batch_size, max_len - i, n_nodes)
        # (batch_size, max_len - i, n_nodes, 3)
        max_indices = triu_add(max_indices, triu_max_indices, 1)
        
        for i in range(2, max_len):
            maxed_scores = looper.send(triu_inner_scores)
            triu_inner_scores, triu_max_indices = max_lastdims(maxed_scores, 3)
            max_indices = triu_add(max_indices, triu_max_indices, 1)
        try:
            looper.send(triu_inner_scores)
        except StopIteration as e:
            outputs = e.args[0]

        return {'last_scores': outputs['logits'], 'max_indices': max_indices, 'first_layer_max_indices': first_layer_max_indices}

    @torch.no_grad()
    def predict(
        self,
        token_seqs: torch.LongTensor, # (batch_size, max_token_seq_len)
        token_seq_masks: torch.LongTensor, # (batch_size, max_token_seq_len)
        pos_seqs: torch.LongTensor, # (batch_size, max_seq_len)
        seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        pos_seq_masks: torch.LongTensor, # (batch_size, max_seq_len)
        offsets: Optional[torch.LongTensor] = None,
        greedy_on_words: Optional[bool] = False,
        **kwargs
    ):
        outputs = self.forward(
            token_seqs=token_seqs, 
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs, 
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            output_keys={'logits', 'seq_logits'},
            **kwargs)
        outputs.update(self.decode(
            token_seqs=token_seqs, 
            token_seq_masks=token_seq_masks,
            pos_seqs=pos_seqs, 
            seq_masks=seq_masks,
            pos_seq_masks=pos_seq_masks,
            offsets=offsets,
            greedy_on_words=greedy_on_words,
            **kwargs))

        return outputs
