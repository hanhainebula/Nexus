import math
import os
import random
from dataclasses import dataclass
from copy import deepcopy

import datasets
from transformers import DataCollatorWithPadding

from Nexus.abc.training.embedder import AbsEmbedderCollator, AbsEmbedderTrainDataset
from Nexus.modules.multimodal import (
    apply_instruction,
    build_prefixed_multimodal_group,
    build_media_base_dir,
    is_empty_multimodal_item,
)

from .arguments import MultimodalEmbedderDataArguments


class AbsMultimodalEmbedderTrainDataset(AbsEmbedderTrainDataset):
    def __init__(
        self,
        args: MultimodalEmbedderDataArguments,
        tokenizer=None,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.shuffle_ratio = args.shuffle_ratio

        train_datasets = []
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                if not (data_dir.endswith(".json") or data_dir.endswith(".jsonl")):
                    continue
                temp_dataset = self._load_dataset(data_dir)
                if len(temp_dataset) > 0:
                    train_datasets.append(temp_dataset)
                continue

            for file_name in os.listdir(data_dir):
                if not (file_name.endswith(".json") or file_name.endswith(".jsonl")):
                    continue
                temp_dataset = self._load_dataset(os.path.join(data_dir, file_name))
                if len(temp_dataset) > 0:
                    train_datasets.append(temp_dataset)

        if len(train_datasets) == 0:
            raise ValueError(f"No multimodal training files were found in {args.train_data}.")
        self.dataset = datasets.concatenate_datasets(train_datasets)

    def _load_dataset(self, file_path: str):
        temp_dataset = datasets.load_dataset(
            "json",
            data_files=file_path,
            split="train",
            cache_dir=self.args.cache_path,
        )
        if "base_dir" not in temp_dataset.column_names:
            temp_dataset = temp_dataset.add_column(
                "base_dir",
                [os.path.dirname(os.path.abspath(file_path))] * len(temp_dataset),
            )
        if len(temp_dataset) > self.args.max_example_num_per_dataset:
            selected_indices = random.sample(list(range(len(temp_dataset))), self.args.max_example_num_per_dataset)
            temp_dataset = temp_dataset.select(selected_indices)
        if not self.args.knowledge_distillation:
            removable_columns = [col for col in ["pos_scores", "neg_scores"] if col in temp_dataset.column_names]
            if len(removable_columns) > 0:
                temp_dataset = temp_dataset.remove_columns(removable_columns)
        else:
            if "pos_scores" not in temp_dataset.column_names or "neg_scores" not in temp_dataset.column_names:
                raise ValueError(
                    f"`pos_scores` and `neg_scores` are required for KD training but missing in {file_path}."
                )
        return temp_dataset

    def _sample_random_negative(self, current_index: int):
        if len(self.dataset) <= 1:
            raise ValueError(
                "The multimodal training set contains a single sample and no negatives. "
                "Provide explicit negatives or increase the dataset size."
            )
        sampled_index = current_index
        while sampled_index == current_index:
            sampled_index = random.randrange(len(self.dataset))
        sampled_row = self.dataset[sampled_index]
        sampled_base_dir = build_media_base_dir(
            base_dir=sampled_row.get("base_dir"),
            media_root=sampled_row.get("media_root", self.args.media_root),
            image_root=sampled_row.get("image_root"),
            video_root=sampled_row.get("video_root"),
        )
        sampled_candidates = build_prefixed_multimodal_group(sampled_row, "pos", base_dir=sampled_base_dir)
        if len(sampled_candidates) == 0:
            sampled_candidates = build_prefixed_multimodal_group(sampled_row, "query", base_dir=sampled_base_dir)
        if len(sampled_candidates) == 0:
            raise ValueError("Unable to draw fallback negatives from the multimodal dataset.")
        sampled_candidate = deepcopy(random.choice(sampled_candidates))
        sampled_candidate["text"] = apply_instruction(
            sampled_candidate.get("text", ""),
            instruction=self.args.passage_instruction_for_retrieval,
            instruction_format=self.args.passage_instruction_format,
        )
        return sampled_candidate

    def _shuffle_text(self, item):
        text = item.get("text", "")
        if self.shuffle_ratio <= 0 or len(text) <= 100 or random.random() >= self.shuffle_ratio:
            return item

        split_text = []
        chunk_size = len(text) // 3 + 1
        for i in range(0, len(text), chunk_size):
            split_text.append(text[i : i + chunk_size])
        random.shuffle(split_text)

        shuffled = dict(item)
        shuffled["text"] = " ".join(split_text)
        return shuffled

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        base_dir = build_media_base_dir(
            base_dir=data.get("base_dir"),
            media_root=data.get("media_root", self.args.media_root),
            image_root=data.get("image_root"),
            video_root=data.get("video_root"),
        )

        query_candidates = build_prefixed_multimodal_group(data, "query", base_dir=base_dir)
        if len(query_candidates) == 0:
            raise ValueError("Each multimodal training record must provide one query item.")
        query = query_candidates[0]
        query["text"] = apply_instruction(
            query.get("text", ""),
            instruction=data.get("prompt", self.args.query_instruction_for_retrieval),
            instruction_format=self.args.query_instruction_format,
        )

        positives = build_prefixed_multimodal_group(data, "pos", base_dir=base_dir)
        negatives = build_prefixed_multimodal_group(data, "neg", base_dir=base_dir)

        positives = [self._shuffle_text(example) for example in positives if not is_empty_multimodal_item(example)]
        negatives = [self._shuffle_text(example) for example in negatives if not is_empty_multimodal_item(example)]

        if len(positives) == 0:
            raise ValueError("Each multimodal training record must provide at least one positive target.")

        pos_idx = random.randrange(len(positives))
        passages = [positives[pos_idx]]

        neg_needed = max(self.args.train_group_size - 1, 0)
        if neg_needed > 0:
            neg_pool = list(range(len(negatives)))
            if len(negatives) == 0:
                neg_indices = []
            elif len(negatives) < neg_needed:
                repeat_factor = math.ceil(neg_needed / len(negatives))
                neg_indices = random.sample(neg_pool * repeat_factor, neg_needed)
            else:
                neg_indices = random.sample(neg_pool, neg_needed)
            passages.extend(negatives[idx] for idx in neg_indices)
            while len(passages) < self.args.train_group_size:
                passages.append(self._sample_random_negative(item))
        else:
            neg_indices = []

        for passage in passages:
            passage["text"] = apply_instruction(
                passage.get("text", ""),
                instruction=self.args.passage_instruction_for_retrieval,
                instruction_format=self.args.passage_instruction_format,
            )

        if self.args.knowledge_distillation:
            teacher_scores = [data["pos_scores"][pos_idx]]
            teacher_scores.extend(data["neg_scores"][idx] for idx in neg_indices)
            while len(teacher_scores) < len(passages):
                teacher_scores.append(0.0)
        else:
            teacher_scores = None

        return query, passages, teacher_scores


@dataclass
class AbsMultimodalEmbedderCollator(AbsEmbedderCollator, DataCollatorWithPadding):
    def __call__(self, features):
        queries = [feature[0] for feature in features]
        passages = [feature[1] for feature in features]
        teacher_scores = [feature[2] for feature in features]

        flattened_passages = []
        for group in passages:
            flattened_passages.extend(group)

        if teacher_scores[0] is None:
            flattened_teacher_scores = None
        else:
            flattened_teacher_scores = []
            for scores in teacher_scores:
                flattened_teacher_scores.extend(scores)

        return queries, flattened_passages, flattened_teacher_scores, False
