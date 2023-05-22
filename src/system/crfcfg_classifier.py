import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import pytorch_lightning as pl
import transformers
from typing import Optional

from ..module import get_module
from ..module.mixin import DocMixin
from .utils import collect, collect_from_other_rank, eval_single_tree, TreeWriter

class CRFCFGClassifier(pl.LightningModule):

    def __init__(
        self,
        model: dict,
        loss: dict,
        optimizer: dict,
        init_weights: Optional[bool] = True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_module(model)
        if init_weights:
            self.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            # orthogonal init
            for name, param in m.named_parameters():
                if name.startswith('weight'):
                    nn.init.orthogonal_(param)
                elif name.startswith('bias'):
                    nn.init.constant_(param, 0)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), 
            lr=self.hparams.optimizer['lr'],
            momentum=self.hparams.optimizer['momentum'])
        
        if self.hparams.optimizer.get('n_cycles', None) is None:
            n_cycle_steps = self.hparams.optimizer['max_steps'] - self.hparams.optimizer['warmup_steps']
            n_cycles = math.ceil(n_cycle_steps / self.hparams.optimizer['period'])
            self.hparams.optimizer['n_cycles'] = n_cycles
            self.hparams.optimizer['max_steps'] = self.hparams.optimizer['warmup_steps'] + n_cycles * self.hparams.optimizer['period']

        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.optimizer['warmup_steps'],
            num_training_steps=self.hparams.optimizer['max_steps'],
            num_cycles=self.hparams.optimizer['n_cycles'])

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def get_coef(self, type, start, end=None):
        if type == 'exponential_decay':
            decay = (end / start) ** (self.global_step / self.trainer.max_steps)
            return start * decay
        elif type == 'liner_decay':
            raise NotImplementedError()
        elif type == 'constant':
            return start

    def training_step(self, batch, batch_idx):
        output_keys = set()
        if self.hparams['loss']['label_loss']['be']:
            output_keys.add('logits')
        if self.hparams['loss']['label_seq_loss']['be'] or self.hparams['loss']['unlabel_seq_loss']['be']:
            output_keys.add('posnode_scores')
        if self.hparams['loss']['syntax_loss']['be']:
            output_keys.add('skeleton_logZ')
            output_keys.add('span_scores')
        # self.log('train/average_label', batch['labels'].float().mean())
        
        outputs = self.model(**batch, output_keys=output_keys)
        if self.hparams['loss']['label_seq_loss']['be'] or self.hparams['loss']['unlabel_seq_loss']['be']:
            logit_seqs = outputs['posnode_scores'] # (batch_size, max_len, n_nodes)

        loss = 0

        if self.hparams['loss']['label_loss']['be']:
            label_loss = F.cross_entropy(outputs['logits'], batch['labels'])
            coef = self.get_coef(**self.hparams['loss']['label_loss']['coef'])
            loss += coef * label_loss
            
            acc = (outputs['logits'].argmax(-1) == batch['labels']).float().mean()
            self.log_dict({
                'train/label_loss': label_loss,
                'train/acc': acc})

        if self.hparams['loss']['label_seq_loss']['be']:
            label_seq_loss = (F.cross_entropy(
                input=logit_seqs.flatten(0, 1), 
                target=batch['label_seqs'].flatten(0, 1), 
                reduction='none') * batch['label_seq_masks'].flatten(0, 1)).sum() \
                / batch['label_seq_masks'].sum()
            coef = self.get_coef(**self.hparams['loss']['label_seq_loss']['coef'])
            loss += coef * label_seq_loss

            seq_acc = ((logit_seqs.argmax(-1) * batch['label_seq_masks'] == \
                batch['label_seqs'] * batch['label_seq_masks']).float().sum() - \
                (1 - batch['label_seq_masks']).sum()) / batch['label_seq_masks'].sum()
            self.log_dict({
                'train/label_seq_loss': label_seq_loss,
                'train/seq_acc': seq_acc})

        if self.hparams['loss']['unlabel_seq_loss']['be']:
            unlabel_seq_masks = (1 - batch['label_seq_masks']) * batch['seq_masks']
            unlabel_word_possible_labels = self.hparams.loss['unlabel_seq_loss']['unlabel_word_possible_labels']

            # unlabel_seq_loss = ((logit_seqs.logsumexp(-1) - logit_seqs[..., unlabel_word_possible_labels].logsumexp(-1)) * unlabel_seq_masks).sum() / unlabel_seq_masks.sum()
            possible_min_logits = logit_seqs[..., unlabel_word_possible_labels].min(-1).values
            unlabel_word_impossible_labels = list(set(range(self.model.n_nodes)) - set(unlabel_word_possible_labels))
            impossible_max_logits = logit_seqs[..., unlabel_word_impossible_labels].max(-1).values
            unlabel_seq_loss = ((impossible_max_logits - possible_min_logits + self.hparams.loss['unlabel_seq_loss']['th']).clamp_min(0) * unlabel_seq_masks).sum() / unlabel_seq_masks.sum()            
            
            coef = self.get_coef(**self.hparams['loss']['unlabel_seq_loss']['coef'])
            loss += coef * unlabel_seq_loss

            self.log('train/unlabel_seq_loss', unlabel_seq_loss)

        if self.hparams['loss']['syntax_loss']['be']:
            syntax_loss = (outputs['skeleton_logZ'] - \
                (batch['const_mats'] * outputs['span_scores']).sum(-1).sum(-1)).mean()
            coef = self.get_coef(**self.hparams['loss']['syntax_loss']['coef'])
            loss += coef * syntax_loss
            self.log('train/syntax_loss', syntax_loss)
        
        return loss
    
    def eval_step(self, batch):
        outputs = self.model.predict(**batch)
        cfg = list(filter(lambda x: isinstance(x, TreeWriter), self.trainer.callbacks))[0].cfg
        seqs_ptb = cfg.batch_decode_ptb(
            seqs=batch['tokenized_seqs'],
            root_ids=outputs['logits'].argmax(-1),
            indices=outputs['max_indices'],
            first_layer_max_indices=outputs['first_layer_max_indices'],
            has_special_tokens=batch['data_args']['add_special_tokens'])
        tree_f1s = []
        for pred_ptb, gold_ptb in zip(seqs_ptb, batch['ptb_seqs']):
            _, _, tree_f1 = eval_single_tree(pred_ptb, gold_ptb, False)
            tree_f1s.append(tree_f1)
        tree_f1s = torch.tensor(tree_f1s, device=self.device)
        
        logits = outputs['logits']
        label_losses = F.cross_entropy(logits, batch['labels'], reduction='none') # (batch_size,)
        label_corrects = (logits.argmax(-1) == batch['labels']).float() # (batch_size,)

        return tree_f1s, label_losses, label_corrects
    
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch)

    def validation_epoch_end(self, outputs):
        tree_f1s, label_losses, label_corrects = collect(outputs)
        if self.trainer.world_size > 1:
            tree_f1s = collect_from_other_rank(tree_f1s, self.device, self.trainer.world_size)
            label_losses = collect_from_other_rank(label_losses, self.device, self.trainer.world_size)
            label_corrects = collect_from_other_rank(label_corrects, self.device, self.trainer.world_size)

        self.log_dict({
            'val/tree_f1': tree_f1s.mean(),
            'val/acc': label_corrects.mean(),
            'val/label_loss': label_losses.mean()})
    
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch)
    
    def test_epoch_end(self, outputs):
        tree_f1s, label_losses, label_corrects = collect(outputs)
        if self.trainer.world_size > 1:
            tree_f1s = collect_from_other_rank(tree_f1s, self.device, self.trainer.world_size)
            label_losses = collect_from_other_rank(label_losses, self.device, self.trainer.world_size)
            label_corrects = collect_from_other_rank(label_corrects, self.device, self.trainer.world_size)

        results = {
            'test/tree_f1': tree_f1s.mean(),
            'test/acc': label_corrects.mean(),
            'test/label_loss': label_losses.mean()}
        self.log_dict(results)
        return results
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction = self.model.predict(greedy_on_words=True, **batch)
        return prediction