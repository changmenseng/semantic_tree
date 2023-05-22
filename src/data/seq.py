import math
import json
import torch
import random
import binpacking
from nltk.tree import Tree
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizerFast
from typing import Optional, List, Union, Dict

from .utils import penn_postags, get_offsets

class SeqDataset(Dataset):

    def __init__(
        self,
        sent_fnames,
        label_fnames,
        labels: List[str],
        postags: Optional[List[str]] = penn_postags
    ):
        self.sent_fnames = sent_fnames
        self.labels = labels
        self.postags = postags

        self.label2id = {label: i for i, label in enumerate(labels)}
        self.postag2id = {postag: i for i, postag in enumerate(postags)}

        self.seqs = []; self.pos_seqs = []
        for sent_fname in sent_fnames:
            with open(sent_fname, 'r', encoding='utf8') as f:
                for line in f:
                    doc = line.rstrip()
                    seq = []; pos_seq = []
                    for ptb_seq in doc.split('\t'):
                        tree = Tree.fromstring(ptb_seq)
                        for pos in tree.treepositions('leaves'):
                            if pos[-1] > 0:
                                subtree = tree[pos[:-1]]
                                tree[pos[:-1]] = Tree(subtree.label(), [' '.join(subtree.leaves())])

                        for leaftree in tree.subtrees(lambda t: t.height() == 2):
                            postag = leaftree.label().split('/')[0]
                            postag = postag.split('+')[-1]
                            word = leaftree.leaves()[0]

                            seq.append(word)
                            pos_seq.append(self.postag2id.get(postag, -1))

                    self.seqs.append(seq)
                    self.pos_seqs.append(pos_seq)

        self.gold_labels = []
        for label_fname in label_fnames:
            with open(label_fname, 'r', encoding='utf8') as f:
                for line in f:
                    label = self.label2id[line.rstrip()]
                    self.gold_labels.append(label)
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        return self.seqs[index], self.pos_seqs[index], self.gold_labels[index]

def get_seq_dl(
    sent_fnames: str,
    label_fnames: str,
    labels: List[str],
    tokenizer: PreTrainedTokenizerFast,
    postags: Optional[List[str]] = penn_postags,
    add_special_tokens: Optional[bool] = False,
    batch_size: Optional[int] = 32,
    max_length: Optional[int] = 1024,
    shuffle: Optional[bool] = False,
    num_workers: Optional[bool] = 8,
    drop_last: Optional[bool] = False,
):
    def collate_fn(batch_raw):
        seqs = []; pos_seqs = []; labels = []
        for seq, pos_seq, label in batch_raw:
            seq = seq[:max_length]
            pos_seq = pos_seq[:max_length]
            if add_special_tokens:
                seq = [tokenizer.bos_token] + seq + [tokenizer.eos_token]
                pos_seq = [-1] + pos_seq + [-1]
            seqs.append(seq)
            pos_seqs.append(pos_seq)
            labels.append(label)
        labels = torch.tensor(labels, dtype=torch.long)
        encoding = tokenizer.pad(
            encoded_inputs={'input_ids': pos_seqs},
            return_tensors='pt')
        pos_seqs = encoding['input_ids'] # (batch_size, max_seq_len)
        seq_masks = encoding['attention_mask'] # (batch_size, max_seq_len)
        pos_seq_masks = (pos_seqs != -1).long() * seq_masks # (batch_size, max_seq_len)
        pos_seqs *= pos_seq_masks # (batch_size, max_seq_len)

        encoding = tokenizer(
            text=seqs,
            padding=True,
            truncation=True,
            max_length=100000,
            return_tensors='pt',
            is_split_into_words=True,
            add_special_tokens=False,
            return_offsets_mapping=True)
        token_seqs = encoding['input_ids'] # (batch_size, max_token_seq_len)
        token_seq_masks = encoding['attention_mask']
    
        batch = {
            'labels': labels,
            'seq_masks': seq_masks, # (batch_size, max_seq_len)
            'token_seqs': token_seqs, # (batch_size, max_token_seq_len), max_token_seq_len >= max_seq_len
            'token_seq_masks': token_seq_masks,
            'pos_seqs': pos_seqs, # (batch_size, max_seq_len)
            'pos_seq_masks': pos_seq_masks, # (batch_size, max_seq_len),
            'offsets': None,
            'tokenized_seqs': seqs}
        batch['data_args'] = {'add_special_tokens': add_special_tokens}
        if token_seqs.shape[-1] != pos_seqs.shape[-1] or (not (seq_masks == token_seq_masks).all()):
            offsets = get_offsets(encoding['offset_mapping'], seqs)
            batch['offsets'] = torch.LongTensor(offsets)

        return batch
    
    ds = SeqDataset(
        sent_fnames=sent_fnames,
        label_fnames=label_fnames,
        labels=labels,
        postags=postags)

    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last)

    return dl