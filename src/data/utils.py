import torch
import itertools
from transformers import PreTrainedTokenizerFast, AutoTokenizer

penn_postags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', \
    'MD', 'NN', 'NNS', 'NNP', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', \
    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', \
    'WP', 'WP$', 'WRB', '$', "''", ',', '.', ':', 'HYPH', 'NFP', '``', \
    '-LRB-', '-RRB-', 'AFX', 'NNPS', 'UNK']

def get_offsets(offset_mapping, tokenized_seqs):
    offsets = []
    max_seq_len = max([len(seq) for seq in tokenized_seqs])
    for offset_mapping_of_a_seq, tokenized_seq in zip(offset_mapping, tokenized_seqs):
        cur_start = 0
        cur_end = 0
        offset = []
        for word in tokenized_seq:
            while offset_mapping_of_a_seq[cur_end][-1] < len(word):
                cur_end += 1
            offset.append([cur_start, cur_end])
            cur_start = cur_end + 1
            cur_end = cur_end + 1
        for i in range(max_seq_len - len(tokenized_seq)):
            offset.append([-1, -1])
        offsets.append(offset)
    return offsets

def get_const_mat(tree):
    for i, pos in enumerate(tree.treepositions('leaves')):
        tree[pos] += f'_{i}'
    words = tree.leaves()
    seq_len = len(words)
    word2id = {word: i for i, word in enumerate(words)}
    mat = torch.zeros(seq_len, seq_len, dtype=torch.long)
    for subtree in tree.subtrees():
        constituent = subtree.leaves()
        start_id = word2id[constituent[0]]
        end_id = word2id[constituent[-1]]
        mat[start_id, end_id] = 1
    return mat

def const_mat_add_special_tokens(mat, cnf_direction):
    seq_len, _ = mat.shape
    new_mat = torch.zeros(seq_len + 2, seq_len + 2, dtype=torch.long)
    new_mat[1:-1, 1:-1] = mat
    new_mat[0, -1] = 1 # root
    if cnf_direction == 'right':
        new_mat[1, -1] = 1
    elif cnf_direction == 'left':
        new_mat[0, -2] = 1
    new_mat[0, 0] = 1 # bos
    new_mat[-1, -1] = 1 # eos
    return new_mat

def const_mat_pad(mats):
    max_seq_len = max([mat.shape[1] for mat in mats])
    new_mats = torch.zeros(len(mats), max_seq_len, max_seq_len, dtype=torch.long)
    for i in range(len(mats)):
        seq_len = mats[i].shape[0]
        new_mats[i][: seq_len, : seq_len] = mats[i]
    return new_mats

def get_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        token_mapping = {
            "-LRB-": "(",
            "-lrb-": "(",
            "-RRB-": ")",
            "-rrb-": ")",
            "-LCB-": "{",
            "-lcb-": "{",
            "-RCB-": "}",
            "-rcb-": "}",
            "-LSB-": "[",
            "-lsb-": "[",
            "-RSB-": "]",
            "-rsb-": "]"}
        from tokenizers.normalizers import Replace, Sequence
        normalizers = [Replace(key, value) for key, value in token_mapping.items()]
        normalizers.append(tokenizer._tokenizer.normalizer)
        tokenizer._tokenizer.normalizer = Sequence(normalizers)
    except:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer

def data_repeater(dl, k=None):
    # create infinite length data loader
    for loader in itertools.repeat(dl):
        for data in loader:
            yield data