import os
import tempfile
from typing import List, Optional, Union

import datasets

from Nexus.abc.evaluation import AbsEvalDataLoader
from Nexus.modules.multimodal import build_media_base_dir, normalize_multimodal_item


def _default_hf_cache_dir() -> str:
    candidates = []
    for env_key in ["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"]:
        env_value = os.getenv(env_key)
        if env_value not in [None, ""]:
            expanded = os.path.expanduser(env_value)
            if env_key != "HF_HUB_CACHE":
                expanded = os.path.join(expanded, "huggingface", "hub")
            candidates.append(expanded)
    candidates.append("/tmp/huggingface/hub")

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=candidate):
                pass
            return candidate
        except OSError:
            continue
    return "/tmp/huggingface/hub"


class MultimodalRetrievalEvalDataLoader(AbsEvalDataLoader):
    def __init__(
        self,
        eval_name: str,
        dataset_dir: Optional[str] = None,
        media_root: Optional[str] = None,
        image_root: Optional[str] = None,
        video_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_redownload: bool = False,
    ):
        self.eval_name = eval_name
        self.dataset_dir = dataset_dir
        self.media_root = media_root
        self.image_root = image_root
        self.video_root = video_root
        if cache_dir is None:
            cache_dir = _default_hf_cache_dir()
        self.cache_dir = os.path.join(cache_dir, eval_name)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            self.cache_dir = os.path.join("/tmp/huggingface/hub", eval_name)
            os.makedirs(self.cache_dir, exist_ok=True)
        self.token = token
        self.force_redownload = force_redownload

    def available_dataset_names(self) -> List[str]:
        if self.dataset_dir is None or not os.path.isdir(self.dataset_dir):
            return []
        dataset_names = []
        for child in sorted(os.listdir(self.dataset_dir)):
            child_dir = os.path.join(self.dataset_dir, child)
            if os.path.isdir(child_dir) and os.path.exists(os.path.join(child_dir, "corpus.jsonl")):
                dataset_names.append(child)
        return dataset_names

    def available_splits(self, dataset_name: Optional[str] = None) -> List[str]:
        if self.dataset_dir is None:
            return ["test"]
        save_dir = self.dataset_dir if dataset_name is None else os.path.join(self.dataset_dir, dataset_name)
        if not os.path.isdir(save_dir):
            return ["test"]

        splits = []
        for file_name in os.listdir(save_dir):
            if file_name.endswith("_queries.jsonl"):
                splits.append(file_name.replace("_queries.jsonl", ""))
        return sorted(set(splits)) or ["test"]

    def check_dataset_names(self, dataset_names: Union[str, List[str]]) -> List[str]:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        available_dataset_names = self.available_dataset_names()
        for dataset_name in dataset_names:
            if dataset_name not in available_dataset_names:
                raise ValueError(
                    f"Dataset name '{dataset_name}' not found. Available dataset names: {available_dataset_names}"
                )
        return dataset_names

    def check_splits(self, splits: Union[str, List[str]], dataset_name: Optional[str] = None) -> List[str]:
        if isinstance(splits, str):
            splits = [splits]
        available_splits = self.available_splits(dataset_name=dataset_name)
        return [split for split in splits if split in available_splits]

    def _get_save_dir(self, dataset_name: Optional[str] = None):
        if self.dataset_dir is None:
            raise NotImplementedError("Remote multimodal evaluation loading is not implemented.")
        return self.dataset_dir if dataset_name is None else os.path.join(self.dataset_dir, dataset_name)

    def _get_media_base_dir(self, save_dir: str):
        return build_media_base_dir(
            base_dir=save_dir,
            media_root=self.media_root,
            image_root=self.image_root,
            video_root=self.video_root,
        )

    def load_corpus(self, dataset_name: Optional[str] = None):
        save_dir = self._get_save_dir(dataset_name=dataset_name)
        media_base_dir = self._get_media_base_dir(save_dir)
        corpus_path = os.path.join(save_dir, "corpus.jsonl")
        corpus_data = datasets.load_dataset("json", data_files=corpus_path, cache_dir=self.cache_dir)["train"]

        corpus = {}
        for entry in corpus_data:
            corpus[entry["_id"]] = normalize_multimodal_item(entry, base_dir=media_base_dir)
        return corpus

    def load_qrels(self, dataset_name: Optional[str] = None, split: str = "test"):
        save_dir = self._get_save_dir(dataset_name=dataset_name)
        qrels_path = os.path.join(save_dir, f"{split}_qrels.jsonl")
        qrels_data = datasets.load_dataset("json", data_files=qrels_path, cache_dir=self.cache_dir)["train"]

        qrels = {}
        for entry in qrels_data:
            qid = entry["query-id"]
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][entry["corpus-id"]] = int(entry["score"])
        return qrels

    def load_queries(self, dataset_name: Optional[str] = None, split: str = "test"):
        save_dir = self._get_save_dir(dataset_name=dataset_name)
        media_base_dir = self._get_media_base_dir(save_dir)
        queries_path = os.path.join(save_dir, f"{split}_queries.jsonl")
        queries_data = datasets.load_dataset("json", data_files=queries_path, cache_dir=self.cache_dir)["train"]

        queries = {}
        for entry in queries_data:
            queries[entry["_id"]] = normalize_multimodal_item(entry, base_dir=media_base_dir)
        return queries
