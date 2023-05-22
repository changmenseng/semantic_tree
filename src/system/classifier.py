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
from .utils import collect, collect_from_other_rank

class Classifier(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer,
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

    def training_step(self, batch, batch_idx):

        logits = self.model(**batch)['logits']
        loss = F.cross_entropy(logits, batch['labels'])
        acc = (logits.argmax(-1) == batch['labels']).float().mean()

        self.log_dict({
            'train/loss': loss,
            'train/acc': acc})
        return loss

    def eval_step(self, batch):
        
        logits = self.model(**batch)['logits']
        losses = F.cross_entropy(logits, batch['labels'], reduction='none')
        corrects = (logits.argmax(-1) == batch['labels']).float()

        return losses, corrects
    
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch)

    def validation_epoch_end(self, outputs):
        losses, corrects = collect(outputs)
        if self.trainer.world_size > 1:
            losses = collect_from_other_rank(losses, self.device, self.trainer.world_size)
            corrects = collect_from_other_rank(corrects, self.device, self.trainer.world_size)
        
        self.log_dict({
            'val/acc': corrects.mean(),
            'val/loss': losses.mean()})
    
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch)
    
    def test_epoch_end(self, outputs):
        losses, corrects = collect(outputs)
        if self.trainer.world_size > 1:
            losses = collect_from_other_rank(losses, self.device, self.trainer.world_size)
            corrects = collect_from_other_rank(corrects, self.device, self.trainer.world_size)        
        results = {
            'test/acc': corrects.mean(),
            'test/loss': losses.mean()}
        self.log_dict(results)
        return results
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        prediction = self.model.predict(**batch)
        return prediction