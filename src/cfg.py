import torch
import queue
from typing import List, Union, Optional

class ContextFreeGrammar:
    """The class handle indexes of non-terminals
    """
    def __init__(
        self,
        nodes: List[str],
        roots: List[str],
        prenodes: List[str], # len(prenodes) >= len(dataset.labels). elements in prenodes and labels are corresponded.
        posnodes: List[str],
        rules: Optional[List[str]] = None,
        pos_unary_rules: Optional[List[str]] = None, # pair indicating head and one child
    ):  # `rules` and `excluded_rules` at least one has values.

        self.nodes = nodes
        self.prenodes = prenodes
        self.posnodes = posnodes
        self.roots = roots
        self.rules = rules
        self.pos_unary_rules = pos_unary_rules

        # index node
        self.node2id = {node: i for i, node in enumerate(nodes)}
        # encode rule
        if rules is None:
            self.encoded_rules = None
        else:
            self.encoded_rules = []
            for rule in self.rules:
                head, children = rule.split(' -> ')
                child_1, child_2 = children.split(' ')
                encoded_rule = [
                    self.node2id[head],
                    self.node2id[child_1],
                    self.node2id[child_2]
                ]
                self.encoded_rules.append(encoded_rule)
                
        if pos_unary_rules is None:
            self.encoded_pos_unary_rules = None
        else:
            self.encoded_pos_unary_rules = []
            for rule in self.pos_unary_rules:
                head, child = rule.split(' -> ')
                encoded_rule = [
                    self.node2id[head],
                    self.node2id[child]
                ]
                self.encoded_pos_unary_rules.append(encoded_rule)

    def modify_left(self, s, node_tag):
        splitted = s.split(' ')
        splitted.insert(-1, f'({node_tag}')
        return ' '.join(splitted)

    def modify_right(self, s):
        return s + ')'

    def decode_ptb(
        self,
        seq: List[str],
        root_id: int,
        index: torch.LongTensor, # (seq_len, seq_len, n_nodes, 3)
        first_layer_max_index: torch.LongTensor, # (seq_len, n_nodes)
        has_special_tokens: Optional[bool] = False
    ):
        # if has_special_tokens:
        #     seq = seq[1:-1]
        seq_len = len(seq)
        seq_ptb = seq.copy()
        q = queue.Queue()
        q.put((root_id, 0, seq_len - 1))
        while not q.empty():
            node_id, i, j = q.get()
            seq_ptb[i] = self.modify_left(seq_ptb[i], self.nodes[node_id])
            seq_ptb[j] = self.modify_right(seq_ptb[j])
            if i != j:
                k, left_node_id, right_node_id = index[i, j, node_id]
                q.put((left_node_id, i, i + k))
                q.put((right_node_id, i + k + 1, j))
            else:
                if first_layer_max_index is not None:
                    posnode_id = first_layer_max_index[i, node_id]
                    if posnode_id != node_id:
                        seq_ptb[i] = self.modify_left(seq_ptb[i], self.nodes[posnode_id])
                        seq_ptb[j] = self.modify_right(seq_ptb[j])

        seq_ptb = ' '.join(seq_ptb)
        return seq_ptb

    def batch_decode_ptb(
        self,
        seqs: List[List[str]],
        root_ids: torch.LongTensor, # (batch_size,)
        indices: torch.LongTensor, # (batch_size, seq_len, seq_len, n_nodes, 3)
        first_layer_max_indices: torch.LongTensor, # (batch_size, seq_len, n_nodes)
        has_special_tokens: Optional[bool] = False
    ):
        seqs_ptb = []
        for seq, root_id, index, first_layer_max_index in zip(seqs, root_ids, indices, first_layer_max_indices):
            seq_len = len(seq)
            index = index[:seq_len, :seq_len] # (seq_len, seq_len, n_nonterminals, 3)
            seq_ptb = self.decode_ptb(seq, root_id, index, first_layer_max_index, has_special_tokens)
            seqs_ptb.append(seq_ptb)
        return seqs_ptb