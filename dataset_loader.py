# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import json

import datasets

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """1234"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

URLS = {
    'sample-cased': {
        'train': 'https://mm-experiments.s3.eu-central-1.amazonaws.com/sample/cased-pos/train.json',
        'validation': 'https://mm-experiments.s3.eu-central-1.amazonaws.com/sample/cased-pos/validation.json',
        'test': 'https://mm-experiments.s3.eu-central-1.amazonaws.com/sample/cased-pos/test.json'
    },
    'full-cased': {
        'train': 'https://mm-experiments.s3.eu-central-1.amazonaws.com/full/cased-pos/train.json',
        'validation': 'https://mm-experiments.s3.eu-central-1.amazonaws.com/full/cased-pos/validation.json',
        'test': 'https://mm-experiments.s3.eu-central-1.amazonaws.com/full/cased-pos/test.json'
    },
    'all-cased': {
        'train': 'https://mm-experiments.s3.eu-central-1.amazonaws.com/full/all-cased/dataset.json',
        'validation': 'https://mm-experiments.s3.eu-central-1.amazonaws.com/full/all-cased/dataset.json'
    }
}


class DutchNeutralityCorpusDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="sample-cased",
            version=VERSION,
            description="This is a sample of the dataset"),
        datasets.BuilderConfig(
            name="full-cased",
            version=VERSION,
            description="This is the entire cased dataset"),
        datasets.BuilderConfig(
            name="all-cased",
            version=VERSION,
            description="This is the all-in-one cased dataset")
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "pos_tags": datasets.Sequence(datasets.Value("int8")),
                "labels": datasets.Sequence(
                    datasets.features.ClassLabel(
                        names=[
                            "O",
                            "B-SUBJ",
                        ]
                    )
                )
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(my_urls)

        if self.config.name != 'all-cased':
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        'filepath': data_dir['train']
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        'filepath': data_dir['validation']
                    }
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        'filepath': data_dir['test']
                    }
                )
            ]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'filepath': data_dir['train']
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    'filepath': data_dir['validation']
                }
            ),
        ]

    def _generate_examples(self, filepath):
        """ Yields examples. """
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                row = json.loads(row)
                yield id_, {
                    'text': row['text'],
                    'tokens': row['tokens'],
                    'labels': row['class_labels'],
                    'pos_tags': row['pos_tag_ids']
                }
