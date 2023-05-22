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

from .vocab import get_sentitag
from .utils import penn_postags, get_offsets, get_const_mat, const_mat_pad, const_mat_add_special_tokens


class DocDataset(Dataset): # has label, const_mat

    def __init__(
        self,
        sent_fnames,
        label_fnames,
        labels: List[str],
        max_doc_size: Optional[int] = 128,
        sent_labels: Optional[List[str]] = None, # for specify flags of non-tree model
        word_labels: Optional[List[str]] = None, # for specify flags of non-tree model
        doc_labels: Optional[List[str]] = None, # for specify flags of non-tree model
        postags: Optional[List[str]] = penn_postags,
        cnf_direction: Optional[str] = 'right',
        unlabel_word_possible_labels: Optional[List[str]] = None
    ):
        self.sent_fnames = sent_fnames
        self.labels = labels
        self.postags = postags
        self.cnf_direction = cnf_direction
        self.max_doc_size = max_doc_size
        self.sent_labels = labels if sent_labels is None else sent_labels
        self.word_labels = labels if word_labels is None else word_labels
        self.doc_labels = labels if doc_labels is None else doc_labels
        self.unlabel_word_possible_labels = self.word_labels if unlabel_word_possible_labels is None else unlabel_word_possible_labels

        self.label2id = {label: i for i, label in enumerate(labels)}
        self.postag2id = {postag: i for i, postag in enumerate(postags)}

        self.docs = []; self.ptb_docs = []; self.pos_docs = []; self.label_docs = []; self.const_mats = []
        for sent_fname in sent_fnames:
            with open(sent_fname, 'r', encoding='utf8') as f:
                for line in f:
                    doc = []; ptb_doc = []; pos_doc = []; label_doc = []; const_mats_of_a_doc = []
                    for i, ptb_sent in enumerate(line.rstrip().split('\t')):
                        if i == self.max_doc_size:
                            break
                        ptb_doc.append(ptb_sent)
                        tree = Tree.fromstring(ptb_sent)
                        tree.chomsky_normal_form(factor=cnf_direction)
                        tree.collapse_unary(collapsePOS=True, collapseRoot=True)
                        for pos in tree.treepositions('leaves'):
                            if pos[-1] > 0:
                                subtree = tree[pos[:-1]]
                                tree[pos[:-1]] = Tree(subtree.label(), [' '.join(subtree.leaves())])

                        seq = []; pos_seq = []; label_seq = []
                        for leaftree in tree.subtrees(lambda t: t.height() == 2):
                            word = leaftree.leaves()[0]
                            postag = leaftree.label().split('+')[-1]
                            sentitag = get_sentitag(word, postag)

                            seq.append(word)
                            pos_seq.append(self.postag2id.get(postag, -1))
                            label_seq.append(self.label2id.get(sentitag, -1))
                            
                        const_mat = get_const_mat(tree)

                        doc.append(seq)
                        pos_doc.append(pos_seq)
                        label_doc.append(label_seq)
                        const_mats_of_a_doc.append(const_mat)
                
                    self.docs.append(doc)
                    self.ptb_docs.append(ptb_doc)
                    self.pos_docs.append(pos_doc)
                    self.label_docs.append(label_doc)
                    self.const_mats.append(const_mats_of_a_doc)
        
        self.gold_labels = []
        for label_fname in label_fnames:
            with open(label_fname, 'r', encoding='utf8') as f:
                for line in f:
                    label = self.label2id[line.rstrip()]
                    self.gold_labels.append(label)

    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, index):
        return  self.docs[index], \
                self.ptb_docs[index], \
                self.pos_docs[index], \
                self.label_docs[index], \
                self.const_mats[index], \
                self.gold_labels[index]

class DocSampler(DistributedSampler):

    def __init__(
        self, 
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        id_doclen_pairs = [(i, len(example[0])) for i, example in enumerate(self.dataset)]
        if self.shuffle:
            random.shuffle(id_doclen_pairs)
        batches = binpacking.to_constant_volume(id_doclen_pairs, self.batch_size, weight_pos=1)
        batches = [[example[0] for example in batch] for batch in batches]
        if len(batches) % world_size != 0:
            n_now_batches = len(batches)
            n_future_batches = math.ceil(len(batches) / world_size) * world_size
            sorted_batch_ids = iter(torch.tensor([len(x) for x in batches]).argsort(descending=True).tolist())
            for i in range(n_future_batches - n_now_batches):
                batch_id = next(sorted_batch_ids)
                batch = batches[batch_id].copy()
                assert len(batch) > 1
                batches[batch_id] = batch[len(batch) // 2:]
                batches.append(batch[: len(batch) // 2])
        
        assert len(batches) % world_size == 0
        self._len = len(batches)

        if self.shuffle:
            random.shuffle(batches)
        # subsample of this rank
        rank_batches = batches[rank::world_size]

        assert len(rank_batches) == int(len(batches) / world_size)

        return iter(rank_batches)
    
    def __len__(self):
        return self._len


class DocBatchSampler(BatchSampler):

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        yield from self.sampler
    
    def __len__(self):
        try:
            world_size = dist.get_world_size()
        except:
            world_size = 1
        return int(len(DocSampler) / world_size)

def get_doc_dl(
    sent_fnames: str,
    label_fnames: str,
    labels: List[str],
    tokenizer: PreTrainedTokenizerFast,
    max_doc_size: Optional[int] = None,
    sent_labels: Optional[List[str]] = None,
    word_labels: Optional[List[str]] = None,
    doc_labels: Optional[List[str]] = None,
    postags: Optional[List[str]] = penn_postags,
    cnf_direction: Optional[str] = 'right',
    add_special_tokens: Optional[bool] = False,
    batch_size: Optional[int] = 32,
    max_length: Optional[int] = 1024, # deprecated, since the data is truncated on preprocessing.
    shuffle: Optional[bool] = False,
    num_workers: Optional[bool] = 8,
    drop_last: Optional[bool] = False,
    unlabel_word_possible_labels: Optional[List[str]] = None,
    special_token_label: Optional[str] = 'O'
) -> DataLoader:
    
    label2id = {label: i for i, label in enumerate(labels)}
    def collate_fn(batch_raw):
        doc_sizes = []; seqs = []; ptb_seqs = []
        pos_seqs = []; label_seqs = []; const_mats = []; labels = []
        for doc, ptb_doc, pos_doc, label_doc, const_mats_of_a_doc, label in batch_raw:
            doc_sizes.append(len(doc))
            labels.append(label)
            for seq, ptb_seq, pos_seq, label_seq, const_mat in zip(doc, ptb_doc, pos_doc, label_doc,const_mats_of_a_doc):
                if add_special_tokens:
                    seq = [tokenizer.bos_token] + seq + [tokenizer.eos_token]
                    pos_seq = [-1] + pos_seq + [-1]
                    special_token_label_id = label2id.get(special_token_label, -1)
                    label_seq = [special_token_label_id] + label_seq + [special_token_label_id]
                    ptb_seq = f'(S (({special_token_label} {tokenizer.bos_token}) {ptb_seq} ({special_token_label} {tokenizer.eos_token})))'
                    # ptb_seq = f'(S ({special_token_label} {tokenizer.bos_token}) (S {ptb_seq} ({special_token_label} {tokenizer.eos_token})))'
                    const_mat = const_mat_add_special_tokens(const_mat, cnf_direction)
                seqs.append(seq)
                ptb_seqs.append(ptb_seq)
                pos_seqs.append(pos_seq)
                label_seqs.append(label_seq)
                const_mats.append(const_mat)
        labels = torch.tensor(labels, dtype=torch.long)
        encoding = tokenizer.pad(
            encoded_inputs={'input_ids': pos_seqs},
            return_tensors='pt')
        pos_seqs = encoding['input_ids'] # (batch_size, max_seq_len)
        seq_masks = encoding['attention_mask'] # (batch_size, max_seq_len)
        pos_seq_masks = (pos_seqs != -1).long() * seq_masks # (batch_size, max_seq_len)
        pos_seqs *= pos_seq_masks # (batch_size, max_seq_len)

        encoding = tokenizer.pad(
            encoded_inputs={'input_ids': label_seqs},
            return_tensors='pt')
        label_seqs = encoding['input_ids']
        label_seq_masks = (label_seqs != -1).long()
        label_seq_masks *= seq_masks
        label_seqs *= label_seq_masks

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

        const_mats = const_mat_pad(const_mats)

        batch = {
            'labels': labels,
            'seq_masks': seq_masks, # (batch_size, max_seq_len)
            'token_seqs': token_seqs, # (batch_size, max_token_seq_len)
            'token_seq_masks': token_seq_masks,
            'pos_seqs': pos_seqs, # (batch_size, max_seq_len)
            'pos_seq_masks': pos_seq_masks, # (batch_size, max_seq_len)
            'label_seqs': label_seqs, # (batch_size, max_seq_len)
            'label_seq_masks': label_seq_masks,  # (batch_size, max_seq_len)
            'const_mats': const_mats,
            'doc_sizes': doc_sizes,
            'offsets': None,
            'tokenized_seqs': seqs,
            'ptb_seqs': ptb_seqs}
        batch['data_args'] = {'add_special_tokens': add_special_tokens}

        if token_seqs.shape[-1] != pos_seqs.shape[-1] or (not (seq_masks == token_seq_masks).all()):
            offsets = get_offsets(encoding['offset_mapping'], seqs)
            batch['offsets'] = torch.LongTensor(offsets) # (batch_size, max_seq_len, 2)
            # for i in range(len(seq_masks)):
            #     word_seq = batch['tokenized_seqs'][i]
            #     token_seq = tokenizer.convert_ids_to_tokens(batch['token_seqs'][i])
            #     offset = batch['offsets'][i]
            #     # 判断分割前后的区别
            #     for j in range(len(word_seq)):
            #         span = offset[j]
            #         word = word_seq[j].lower()
            #         # cat_tokens = tokenizer.decode(tokenizer(token_seq[span[0]: span[1] + 1], is_split_into_words=True, add_special_tokens=False)['input_ids'], clean_up_tokenization_spaces=False)
            #         cat_tokens = ''.join(token_seq[span[0]: span[1] + 1]).lower()
            #         if word != '#':
            #             cat_tokens = cat_tokens.replace('#', '')
            #         if word != cat_tokens:
            #             print(word, cat_tokens)

        
        return batch

    if max_doc_size is None:
        max_doc_size = batch_size
    ds = DocDataset(
        sent_fnames=sent_fnames, 
        label_fnames=label_fnames, 
        labels=labels, 
        max_doc_size=max_doc_size,
        sent_labels=sent_labels,
        word_labels=word_labels,
        doc_labels=doc_labels,
        postags=postags,
        cnf_direction=cnf_direction,
        unlabel_word_possible_labels=unlabel_word_possible_labels)
    
    sampler = DocSampler(ds, batch_size, shuffle)
    batch_sampler = DocBatchSampler(sampler, batch_size, drop_last)

    dl = DataLoader(
        dataset=ds,
        num_workers=num_workers,
        collate_fn=collate_fn,
        batch_sampler=batch_sampler)
    dl._DataLoader__initialized = False
    dl.sampler = sampler
    dl._DataLoader__initialized = True
    return dl