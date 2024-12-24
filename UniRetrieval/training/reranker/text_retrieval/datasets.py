import os
import math
import random
import logging
import datasets
import torch.distributed as dist
from dataclasses import dataclass
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
    BatchEncoding
)
from typing import List

from .arguments import AbsTextRerankerDataArguments
from UniRetrieval.abc.training.reranker import AbsRerankerTrainDataset, AbsRerankerCollator

logger = logging.getLogger(__name__)


class AbsTextRerankerTrainDataset(AbsRerankerTrainDataset):
    """Abstract class for reranker training dataset.

    Args:
        args (AbsRerankerDataArguments): Data arguments.
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
    """
    def __init__(
        self,
        args: AbsTextRerankerDataArguments,
        tokenizer: PreTrainedTokenizer
    ):
        self.args = args
        self.tokenizer = tokenizer

        train_datasets = []
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                if not (data_dir.endswith('.json') or data_dir.endswith('.jsonl')): continue
                temp_dataset = self._load_dataset(data_dir)
                if len(temp_dataset) == 0: continue
                train_datasets.append(temp_dataset)
            else:
                for file in os.listdir(data_dir):
                    if not (file.endswith('.json') or file.endswith('.jsonl')): continue
                    temp_dataset = self._load_dataset(os.path.join(data_dir, file))
                    if len(temp_dataset) == 0: continue
                    train_datasets.append(temp_dataset)
        self.dataset = datasets.concatenate_datasets(train_datasets)

        self.max_length = self.args.query_max_len + self.args.passage_max_len

    def _load_dataset(self, file_path: str):
        """Load dataset from path.

        Args:
            file_path (str): Path to load the datasets from.

        Raises:
            ValueError: `pos_scores` and `neg_scores` not found in the features of training data

        Returns:
            datasets.Dataset: Loaded HF dataset.
        """
        if dist.get_rank() == 0:
            logger.info(f'loading data from {file_path} ...')

        temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=self.args.cache_path)
        if len(temp_dataset) > self.args.max_example_num_per_dataset:
            temp_dataset = temp_dataset.select(random.sample(list(range(len(temp_dataset))), self.args.max_example_num_per_dataset))
        if not self.args.knowledge_distillation:
            if 'pos_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['pos_scores'])
            if 'neg_scores' in temp_dataset.column_names:
                temp_dataset = temp_dataset.remove_columns(['neg_scores'])
        else:
            if 'pos_scores' not in temp_dataset.column_names or 'neg_scores' not in temp_dataset.column_names:
                raise ValueError(f"`pos_scores` and `neg_scores` not found in the features of training data in {file_path}, which is necessary when using knowledge distillation.")
        return temp_dataset

    def _shuffle_text(self, text):
        """shuffle the input text.

        Args:
            text (str): Input text.

        Returns:
            str: Shuffled text.
        """
        if self.args.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.args.shuffle_ratio:
            split_text = []
            chunk_size = len(text)//3 + 1
            for i in range(0, len(text), chunk_size):
                split_text.append(text[i:i+chunk_size])
            random.shuffle(split_text)
            return " ".join(split_text)
        else:
            return text

    def __len__(self):
        return len(self.dataset)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        """Creates a single input example by encoding and preparing a query and document pair for the model.

        Args:
            qry_encoding (str): Query to be encoded.
            doc_encoding (str): Document to be encoded.

        Returns:
            dict: A dictionary containing tokenized and prepared inputs, ready for model consumption.
        """
        qry_inputs = self.tokenizer.encode(qry_encoding, truncation=True, max_length=self.args.query_max_len + self.args.passage_max_len // 4, add_special_tokens=False)
        doc_inputs = self.tokenizer.encode(doc_encoding, truncation=True, max_length=self.args.passage_max_len + self.args.query_max_len // 2, add_special_tokens=False)
        item = self.tokenizer.prepare_for_model(
            qry_inputs,
            doc_inputs,
            truncation='only_second',
            max_length=self.args.query_max_len + self.args.passage_max_len,
            padding=False,
        )
        return item

    def __getitem__(self, item):
        data = self.dataset[item]
        train_group_size = self.args.train_group_size

        query = data['query']
        if self.args.query_instruction_for_rerank is not None:
            query = self.args.query_instruction_format.format(
                data['query_prompt'] if 'query_prompt' in data else self.args.query_instruction_for_rerank,
                query
            )

        passages = []
        teacher_scores = []

        assert isinstance(data['pos'], list) and isinstance(data['neg'], list)

        pos_idx = random.choice(list(range(len(data['pos']))))
        passages.append(self._shuffle_text(data['pos'][pos_idx]))

        neg_all_idx = list(range(len(data['neg'])))
        if len(data['neg']) < train_group_size - 1:
            num = math.ceil((train_group_size - 1) / len(data['neg']))
            neg_idxs = random.sample(neg_all_idx * num, train_group_size - 1)
        else:
            neg_idxs = random.sample(neg_all_idx, self.args.train_group_size - 1)
        for neg_idx in neg_idxs:
            passages.append(data['neg'][neg_idx])

        if self.args.knowledge_distillation:
            assert isinstance(data['pos_scores'], list) and isinstance(data['neg_scores'], list)
            teacher_scores.append(data['pos_scores'][pos_idx])
            for neg_idx in neg_idxs:
                teacher_scores.append(data['neg_scores'][neg_idx])
            if not all(isinstance(score, (int, float)) for score in teacher_scores):
                raise ValueError(f"pos_score or neg_score must be digit")
        else:
            teacher_scores = None

        if self.args.passage_instruction_for_rerank is not None:
            passages = [
                self.args.passage_instruction_format.format(
                    data['passage_prompt'] if 'passage_prompt' in data else self.args.passage_instruction_for_rerank, p
                )
                for p in passages
            ]

        batch_data = []
        for passage in passages:
            batch_data.append(self.create_one_example(query, passage))

        return batch_data, teacher_scores

@dataclass
class AbsTextRerankerCollator(AbsRerankerCollator, DataCollatorWithPadding):
    """
    The abstract reranker collator.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features) -> list[BatchEncoding]:
        teacher_scores = [f[1] for f in features]
        if teacher_scores[0] is None:
            teacher_scores = None
        elif isinstance(teacher_scores[0], list):
            teacher_scores = sum(teacher_scores, [])

        features = [f[0] for f in features]
        if isinstance(features[0], list):
            features = sum(features, [])

        collated = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.query_max_len + self.passage_max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # return {
        #     "pair": collated,
        #     "teacher_scores": teacher_scores,
        # }
        return collated, teacher_scores