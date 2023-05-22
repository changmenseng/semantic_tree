import os
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import BasePredictionWriter

class DDPBatchPredictionWriter(BasePredictionWriter):

    def __init__(self, fname):
        super().__init__(write_interval='batch')
        self.fname = fname
    
    @property
    def fname(self):
        return self._fname
    
    @fname.setter
    def fname(self, value):
        path = os.path.split(value)[0]
        if not os.path.exists(path):
            os.makedirs(path)
        self._fname = value

    def rank_fname(self, rank):
        return f'{self.fname}.rank={rank}'

    def on_predict_epoch_end(
        self,
        trainer,
        pl_module,
        outputs
    ):  
        if pl_module.global_rank == 0:
            for rank in range(1, trainer.world_size): # wait for all complete
                dist.recv(torch.tensor(0, device=pl_module.device), rank)
            idx_data_pairs = []
            for rank in range(trainer.world_size):
                with open(self.rank_fname(rank), 'r', encoding='utf8') as f:
                    for line in f: 
                        idx, data = line.rstrip().split('\t\t')
                        idx_data_pair = (int(idx), data)
                        idx_data_pairs.append(idx_data_pair)
                os.remove(self.rank_fname(rank))
            idx_data_pairs.sort(key=lambda x: x[0])
            with open(self.fname, 'w', encoding='utf8') as f:
                for _, data in idx_data_pairs:
                    f.write(data + '\n')
        else:
            dist.send(torch.tensor(1, device=pl_module.device), 0)

class LabelWriter(DDPBatchPredictionWriter):

    def __init__(self, fname, labels):
        super().__init__(fname=fname)
        self.labels = labels
    
    def write_on_batch_end(
        self, 
        trainer, 
        pl_module, 
        prediction, 
        batch_indices, 
        batch,
        batch_idx, 
        dataloader_idx
    ):
        labels = prediction['logits'].argmax(-1).tolist()
        with open(self.rank_fname(pl_module.global_rank), 'a', encoding='utf8') as f:
            for idx, label in zip(batch_indices, labels):
                f.write(f'{idx}\t\t{self.labels[label]}\n')

def split(l, sizes):
    assert sum(sizes) == len(l)
    res = []
    for size in sizes:
        res.append(l[:size])
        l = l[size:]
    return res

class TreeWriter(DDPBatchPredictionWriter):
    
    def __init__(self, fname, tokenizer, cfg):
        super().__init__(fname)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def write_on_batch_end(
        self, 
        trainer, 
        pl_module, 
        prediction, 
        batch_indices, 
        batch,
        batch_idx, 
        dataloader_idx
    ):
        tokenized_seqs = batch['tokenized_seqs']
        seq_labels = prediction['seq_logits'].argmax(-1).cpu().detach()
        indices = prediction['max_indices'].cpu().detach()
        first_layer_max_indices = prediction['first_layer_max_indices'].cpu().detach()
        has_special_tokens = batch['data_args']['add_special_tokens']
        ptb_seqs = self.cfg.batch_decode_ptb(
            seqs=tokenized_seqs, 
            root_ids=seq_labels, 
            indices=indices, 
            first_layer_max_indices=first_layer_max_indices, 
            has_special_tokens=has_special_tokens)
        f = open(self.rank_fname(pl_module.global_rank), 'a', encoding='utf8')
        for data_idx, ptb_doc in zip(batch_indices, split(ptb_seqs, batch['doc_sizes'])):
            ptb_doc = '\t'.join(ptb_doc)
            f.write(f'{data_idx}\t\t{ptb_doc}\n')
        f.close()