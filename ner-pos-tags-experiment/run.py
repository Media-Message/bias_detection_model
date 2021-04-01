import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb

from experiment_lib.data import SubjectivityDetectionDataModule
from experiment_lib.model import SubjectivityDetectionTransformer


def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SubjectivityDetectionDataModule.add_argparse_args(parser)
    parser = SubjectivityDetectionTransformer.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


def main(args):
    pl.seed_everything(args.seed)
    dm = SubjectivityDetectionDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup('fit')
    model = SubjectivityDetectionTransformer(
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        **vars(args))
    wandb_logger = WandbLogger(project=args.task_name)
    wandb.login()
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    return dm, model, trainer


if __name__ == '__main__':
    mocked_args = """
    --model_name_or_path GroNLP/bert-base-dutch-cased
    --task_name sample-cased
    --max_epochs 1
    --precision 16
    --gpus 1""".split()

    args = parse_args(mocked_args)
    dm, model, trainer = main(args)
    trainer.fit(model, dm)
    trainer.test()
    trainer.save_checkpoint('pl-models/model.ckpt')
