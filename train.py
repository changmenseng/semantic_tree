import os
import torch
import argparse
import json
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from src.data import get_tokenizer, data_repeater, get_doc_dl
from src.data import get_doc_dl
from src.cfg import ContextFreeGrammar
from src.system import CRFCFGClassifier
from src.system.utils import LabelWriter, TreeWriter

def main(args):
    os.environ['PL_GLOBAL_SEED'] = str(args['seed'])
    os.environ['PL_SEED_WORKERS'] = '1'
    pl.seed_everything(args['seed'], workers=True)
    # build data
    tokenizer = get_tokenizer(args['data']['tokenizer_path'])
    train_dl = get_doc_dl(
        **args['data']['shared_kwargs'],
        **args['data']['train'],
        tokenizer=tokenizer,
        shuffle=True,
    )
    val_dl = get_doc_dl(
        **args['data']['shared_kwargs'],
        **args['data']['val'],
        tokenizer=tokenizer,
        shuffle=False,
    )
    test_dls = []
    for test_arg in args['data']['test']:
        test_dl = get_doc_dl(
            **args['data']['shared_kwargs'],
            **test_arg,
            tokenizer=tokenizer,
            shuffle=False,
        )
        test_dls.append(test_dl)

    pretrained_fname = args['model'].get('pretrained_fname', None)
    cfg = ContextFreeGrammar(**args['data']['cfg'])
    if pretrained_fname is None:

        args['model']['model']['hparams']['n_nodes'] = len(cfg.nodes)
        args['model']['model']['hparams']['roots'] = [cfg.node2id[n] for n in cfg.roots]
        args['model']['model']['hparams']['prenodes'] = [cfg.node2id[n] for n in cfg.prenodes]
        args['model']['model']['hparams']['posnodes'] = [cfg.node2id[n] for n in cfg.posnodes]
        args['model']['model']['hparams']['rules'] = cfg.encoded_rules
        args['model']['model']['hparams']['pos_unary_rules'] = cfg.encoded_pos_unary_rules
        args['model']['model']['hparams']['doc_labels'] = [cfg.node2id[n] for n in train_dl.dataset.doc_labels]
        args['model']['model']['hparams']['n_postags'] = len(train_dl.dataset.postags)
        args['model']['loss']['unlabel_seq_loss']['unlabel_word_possible_labels'] = [cfg.node2id[n] for n in train_dl.dataset.unlabel_word_possible_labels]

        # build model
        model = CRFCFGClassifier(**args['model'])

    else:
        model = CRFCFGClassifier.load_from_checkpoint(pretrained_fname)

    # build trainer
    save_dir, name = os.path.split(args['save_dir'])
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=args.get('version', None)
    )
    ckpt_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename='step={step}_val_acc={val/acc:.4f}_val_tree_f1={val/tree_f1:.4f}',
        auto_insert_metric_name=False,
        monitor='val/acc',
        mode='max',
        save_top_k=5
    )
    lr_monitor = LearningRateMonitor('step')
    label_writer = LabelWriter(
        fname=f'{logger.log_dir}/labels.txt',
        labels=train_dl.dataset.labels)
    tree_writer = TreeWriter(
        fname=f'{logger.log_dir}/trees.txt',
        tokenizer=tokenizer,
        cfg=cfg)
    callbacks = [ckpt_callback, label_writer, lr_monitor, tree_writer]
    plugins = None
    if args['trainer']['gpus'] > 1 :
        # args['trainer']['distributed_backend'] = 'ddp'
        plugins = DDPPlugin(find_unused_parameters=True)
    trainer = pl.Trainer(
        **args['trainer'],
        deterministic=True,
        log_every_n_steps=5,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins
    )
    # train
    trainer.fit(model, data_repeater(train_dl), val_dl)
    if model.global_rank == 0:
        with open(f'{logger.log_dir}/config.json', 'w', encoding='utf8') as f:
            json.dump(args, f, indent=4, ensure_ascii=False)
    
    # test
    results = dict()
    for ckpt_path in ckpt_callback.best_k_models.keys():
        results[ckpt_path] = dict()
        for test_dl in test_dls:
            results_of_dl = trainer.test(model, test_dl, ckpt_path=ckpt_path)[0]
            fname = test_dl.dataset.sent_fnames[0]
            results[ckpt_path][fname] = results_of_dl
    if model.global_rank == 0:
        with open(f'{logger.log_dir}/results.json', 'w', encoding='utf8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    # predict
    for ckpt_path in ckpt_callback.best_k_models.keys():
        output_prefix = os.path.split(ckpt_path)[-1][:-5]
        label_writer.fname = f'{logger.log_dir}/outputs/{output_prefix}_labels.txt'
        tree_writer.fname = f'{logger.log_dir}/outputs/{output_prefix}_trees.txt'
        predictions = trainer.predict(model, test_dls[0], ckpt_path=ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Config file.')
    config_file = parser.parse_args().config

    with open(config_file, 'r', encoding='utf8') as f:
        args = json.load(f)
    main(args)