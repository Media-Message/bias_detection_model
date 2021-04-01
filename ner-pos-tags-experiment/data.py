import datasets
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiment_lib.utils import (
    get_label_map,
    get_pos_tag_id_map)

from transformers import (
    DataCollatorForTokenClassification)

DATASET_LOADER_FILEPATH = 'dataset_loader_v2.py'
LABEL_LIST, NUM_LABELS, LABEL_TO_ID = get_label_map()
POS_TAG_TO_ID_MAP, POS_ID_TO_TAG_MAP = get_pos_tag_id_map()


def collate(features,
            tokenizer,
            padding=True,
            pad_to_multiple_of=400,
            max_length=None,
            label_pad_token_id=-100,
            pos_tag_pad_token_id=0):

    label_name = "label" if "label" in features[0].keys() else "labels"
    labels = [feature[label_name]
              for feature in features] if label_name in features[0].keys() else None
    pos_tags = [feature['pos_tags'] for feature in features]

    batch = tokenizer.pad(
        features,
        padding=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt" if labels is None else None,
    )

    if labels is None:
        return batch

    sequence_length = torch.tensor(batch["input_ids"]).shape[1]
    padding_side = tokenizer.padding_side

    if padding_side == "right":
        batch["labels"] = [label + [label_pad_token_id] *
                           (sequence_length - len(label)) for label in labels]
        batch["pos_tags"] = [tags + [pos_tag_pad_token_id] *
                             (sequence_length - len(tags)) for tags in pos_tags]
    else:
        batch["labels"] = [[label_pad_token_id] *
                           (sequence_length - len(label)) + label for label in labels]
        batch["pos_tags"] = [[pos_tag_pad_token_id] *
                             (sequence_length - len(tags)) for tags in pos_tags]

    batch = {k: torch.tensor(v, dtype=torch.int64)
             for k, v in batch.items() if k not in ['text', 'tokens']}

    return batch


class SubjectivityDetectionDataModule(pl.LightningDataModule):

    loader_columns = [
        'input_ids',
        'attention_mask',
        'labels',
        'pos_tags'
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 4,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.text_field = 'tokens'
        self.pos_tags_field = 'pos_tags'
        self.num_labels = NUM_LABELS
        self.label_all_tokens = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True)
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset = datasets.load_dataset(
            DATASET_LOADER_FILEPATH,
            self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )

            self.columns = [
                c for c in self.dataset[split].column_names
                if c in self.loader_columns]

#             self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [
            x for x in self.dataset.keys() if 'validation' in x]

    def prepare_data(self):
        # Cache dataset and tokenizer
        datasets.load_dataset(DATASET_LOADER_FILEPATH, self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        print('Calling training dataset')
        return DataLoader(
            self.dataset['train'],
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda x: collate(x, tokenizer=self.tokenizer))

    def val_dataloader(self):
        print('Calling validation dataset')
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset['validation'],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=lambda x: collate(x, tokenizer=self.tokenizer))
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                    collate_fn=lambda x: collate(x, tokenizer=self.tokenizer)) for x in self.eval_splits
            ]

    def test_dataloader(self):
        print('Calling test dataset')
        if len(self.eval_splits) == 1:
            return DataLoader(
                self.dataset['test'],
                batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                collate_fn=lambda x: collate(x, tokenizer=self.tokenizer))
        elif len(self.eval_splits) > 1:
            return [
                DataLoader(
                    self.dataset[x],
                    batch_size=self.eval_batch_size,
                    num_workers=self.num_workers,
                    collate_fn=lambda x: collate(x, tokenizer=self.tokenizer)) for x in self.eval_splits
            ]

    def tokenize_and_align_labels_and_tags(self, examples, text_column_name):

        batch_features = self.tokenizer(
            examples[text_column_name],
            padding=True,
            truncation=True,
            is_split_into_words=True
        )

        batch_labels = []
        batch_pos_tags = []
        labels_and_tags = zip(examples['labels'], examples['pos_tags'])
        for i, (labels, tags) in enumerate(labels_and_tags):

            word_ids = batch_features.word_ids(batch_index=i)

            previous_word_idx = None
            label_ids = []
            pos_tag_ids = []
            for word_idx in word_ids:

                if word_idx is None:
                    label_ids.append(-100)
                    pos_tag_ids.append(0)

                elif word_idx != previous_word_idx:
                    label_ids.append(LABEL_TO_ID[labels[word_idx]])
                    pos_tag_ids.append(tags[word_idx])

                else:
                    label_ids.append(
                        LABEL_TO_ID[labels[word_idx]]
                        if self.label_all_tokens else -100)
                    pos_tag_ids.append(
                        tags[word_idx] if self.label_all_tokens else 0)

                previous_word_idx = word_idx

            batch_labels.append(label_ids)
            batch_pos_tags.append(pos_tag_ids)

        batch_features['labels'] = batch_labels
        batch_features['pos_tags'] = batch_pos_tags

        return batch_features

    def convert_to_features(self, example_batch, indices=None):

        features = \
            self.tokenize_and_align_labels_and_tags(
                example_batch,
                text_column_name=self.text_field)

        return features
