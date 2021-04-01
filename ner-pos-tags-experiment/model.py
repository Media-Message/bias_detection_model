from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import datasets
import numpy as np
import pytorch_lightning as pl
import torch
from transformers import (
    AdamW,
    AutoConfig,
    get_linear_schedule_with_warmup,
    glue_compute_metrics
)

from experiment_lib.token_classification import (
    ModifiedBertForTokenClassification
)

label_list = ['O', 'B-SUBJ']
num_labels = len(label_list)
LABEL_TO_ID = {i: i for i in range(len(label_list))}


class SubjectivityDetectionTransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        eval_splits: Optional[list] = None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels)

        self.model = ModifiedBertForTokenClassification.from_pretrained(
            model_name_or_path,
            config=self.config)

        self.metric = datasets.load_metric(
            'seqeval',
            self.hparams.task_name,
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.return_entity_level_metrics = True

    def compute_metrics(self, predictions, labels):
        #         predictions = np.argmax(predictions, axis=2)
        predictions = np.argmax(predictions, axis=1)

        if not isinstance(predictions[0], list):
            predictions = [predictions]

        if not isinstance(labels[0], list):
            labels = [labels]

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(
            predictions=true_predictions,
            references=true_labels)

        if self.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        labels = batch["labels"]

        preds = torch.cat([x for x in logits]).detach().cpu().numpy()
        labels = torch.cat([x for x in labels]).detach().cpu().numpy()
        current_loss = loss.cpu().numpy()

        self.log('val_loss', loss, prog_bar=True)

        metrics_summary = self.compute_metrics(
            predictions=preds, labels=labels)

        metrics_summary = {
            ''.join(['val_', k]): v for k, v in metrics_summary.items()}
        self.log_dict(metrics_summary, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss, logits = outputs[:2]
        labels = batch["labels"]

        preds = torch.cat([x for x in logits]).detach().cpu().numpy()
        labels = torch.cat([x for x in labels]).detach().cpu().numpy()
        current_loss = loss.cpu().numpy()

        self.log('test_loss', loss, prog_bar=True)

        metrics_summary = self.compute_metrics(
            predictions=preds, labels=labels)

        metrics_summary = {
            ''.join(['test_', k]): v for k, v in metrics_summary.items()}
        self.log_dict(metrics_summary, prog_bar=True)

        return loss

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader()
            # is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            self.total_steps = (
                (len(train_loader.dataset) //
                 (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parser
