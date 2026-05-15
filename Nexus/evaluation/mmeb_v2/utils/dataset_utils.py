import os
from pathlib import Path
import datasets
import pyarrow.parquet as pq
from datasets import load_dataset, load_from_disk
from .basic_utils import print_rank


def sample_dataset(dataset, **kwargs):
    dataset_name = kwargs.get("dataset_name", "UNKNOWN-DATASET")
    num_sample_per_subset = kwargs.get("num_sample_per_subset", None)

    if num_sample_per_subset is not None and type(num_sample_per_subset) is str and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    if type(num_sample_per_subset) is int and num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print_rank(f"Subsample {dataset_name} to {len(dataset)} samples")

    return dataset


def load_qrels_mapping(qrels):
    """
    Returns:
        {
            "qid1": {"docA": 2, "docB": 1},
            "qid2": {"docC": 3},
            ...
        }
    """
    qrels_mapping = {}

    for row in qrels:
        qid = row["query-id"]
        docid = row["corpus-id"]
        score = row["score"]

        if score > 0:
            if qid not in qrels_mapping:
                qrels_mapping[qid] = {}
            # keep the higher score if already exists
            existing_score = qrels_mapping[qid].get(docid, 0)
            qrels_mapping[qid][docid] = max(existing_score, score)

    return qrels_mapping


def load_hf_dataset(hf_path):
    repo, subset, split = hf_path
    local_dataset = _load_local_dataset_if_available(repo, subset, split)
    if local_dataset is not None:
        return local_dataset

    if subset and split:
        return load_dataset(repo, subset, split=split)
    elif subset:
        return load_dataset(repo, subset)
    elif split:
        return load_dataset(repo, split=split)
    else:
        return load_dataset(repo)


def _load_local_dataset_if_available(repo, subset=None, split=None):
    """Prefer local dataset copies/caches when configured by environment.

    Supported environment variables:
    - LOCAL_HF_DATASETS_ROOT: directories produced by datasets.save_to_disk.
    - OFFLINE_HF_DATASETS_ROOT / HF_DATASETS_CACHE: Hugging Face arrow cache.
    - OFFLINE_HF_HUB_ROOT / HF_HUB_CACHE: Hugging Face hub parquet snapshots.
    """
    if not isinstance(repo, str):
        return None

    for local_root in _split_env_paths(os.environ.get("LOCAL_HF_DATASETS_ROOT")):
        dataset = _try_load_from_disk_root(local_root, repo, subset, split)
        if dataset is not None:
            return dataset

    for cache_root in _split_env_paths(
        os.environ.get("OFFLINE_HF_DATASETS_ROOT") or os.environ.get("HF_DATASETS_CACHE")
    ):
        dataset = _try_load_arrow_cache(cache_root, repo, subset, split)
        if dataset is not None:
            return dataset

    for hub_root in _split_env_paths(os.environ.get("OFFLINE_HF_HUB_ROOT") or os.environ.get("HF_HUB_CACHE")):
        dataset = _try_load_hub_parquet_cache(hub_root, repo, subset, split)
        if dataset is not None:
            return dataset

    return None


def _split_env_paths(value):
    if value in [None, ""]:
        return []
    return [Path(item).expanduser() for item in value.split(os.pathsep) if item]


def _repo_path_candidates(root: Path, repo: str):
    repo_parts = repo.split("/")
    candidates = [root / repo]
    if len(repo_parts) == 2:
        org, name = repo_parts
        candidates.extend(
            [
                root / org / name,
                root / f"{org}___{name}",
                root / f"{org.lower()}___{name}",
                root / f"datasets--{org}--{name}",
                root / f"datasets--{org.lower()}--{name}",
            ]
        )
    return list(dict.fromkeys(candidates))


def _try_load_from_disk_root(root: Path, repo: str, subset=None, split=None):
    if not root.exists():
        return None

    for repo_root in _repo_path_candidates(root, repo):
        candidates = []
        if subset not in [None, ""]:
            candidates.append(repo_root / str(subset))
        candidates.append(repo_root)

        for candidate in candidates:
            if not (candidate / "dataset_info.json").exists() and not (candidate / "dataset_dict.json").exists():
                continue
            dataset = load_from_disk(str(candidate))
            print_rank(f"Local dataset hit: repo={repo}, subset={subset}, split={split}, path={candidate}")
            if split not in [None, ""] and isinstance(dataset, datasets.DatasetDict):
                return dataset[split]
            return dataset
    return None


def _try_load_arrow_cache(root: Path, repo: str, subset=None, split=None):
    if not root.exists() or split in [None, ""] or "/" not in repo:
        return None
    subset_name = subset or "default"
    for cache_dir_name in _candidate_datasets_cache_dir_names(repo):
        subset_root = root / cache_dir_name / subset_name
        if not subset_root.exists():
            continue
        matches = sorted(subset_root.glob(f"0.0.0/*/*-{split}.arrow"))
        if matches:
            print_rank(f"Local arrow cache hit: repo={repo}, subset={subset}, split={split}, path={matches[0]}")
            return datasets.Dataset.from_file(str(matches[0]))
    return None


def _try_load_hub_parquet_cache(root: Path, repo: str, subset=None, split=None):
    if not root.exists() or split in [None, ""] or "/" not in repo:
        return None
    for cache_dir_name in _candidate_hub_cache_dir_names(repo):
        snapshot_root = root / cache_dir_name / "snapshots"
        if not snapshot_root.exists():
            continue
        for snapshot_dir in sorted(snapshot_root.iterdir()):
            target_dirs = []
            if subset not in [None, ""]:
                target_dirs.append(snapshot_dir / str(subset))
            else:
                target_dirs.append(snapshot_dir)
                # Many HF dataset repos store parquet shards under a data/ subdirectory.
                target_dirs.append(snapshot_dir / "data")

            for target_dir in target_dirs:
                if not target_dir.exists():
                    continue
                matches = sorted(target_dir.glob(f"{split}-*.parquet"))
                direct_match = target_dir / f"{split}.parquet"
                if not matches and direct_match.exists():
                    matches = [direct_match]
                if matches:
                    print_rank(
                        f"Local parquet cache hit: repo={repo}, subset={subset}, split={split}, "
                        f"files={[str(path) for path in matches]}"
                    )
                    return _dataset_from_local_parquet_files(matches)
    return None



def _dataset_from_local_parquet_files(paths):
    try:
        return datasets.Dataset.from_parquet([str(path) for path in paths])
    except ValueError as error:
        message = str(error)
        if "Feature type" not in message or "List" not in message:
            raise
        print_rank(
            "Local parquet metadata uses legacy List feature; "
            f"retry without embedded Hugging Face metadata: {[str(path) for path in paths]}"
        )
        tables = [pq.read_table(str(path)).replace_schema_metadata(None) for path in paths]
        if len(tables) == 1:
            table = tables[0]
        else:
            table = __import__("pyarrow").concat_tables(tables, promote_options="default")
        return datasets.Dataset(table)


def _candidate_datasets_cache_dir_names(repo: str):
    org, name = repo.split("/", 1)
    return list(
        dict.fromkeys(
            [
                f"{org}___{name}",
                f"{org}___{name.lower()}",
                f"{org.lower()}___{name}",
                f"{org.lower()}___{name.lower()}",
            ]
        )
    )


def _candidate_hub_cache_dir_names(repo: str):
    org, name = repo.split("/", 1)
    return list(
        dict.fromkeys(
            [
                f"datasets--{org}--{name}",
                f"datasets--{org}--{name.lower()}",
                f"datasets--{org.lower()}--{name}",
                f"datasets--{org.lower()}--{name.lower()}",
            ]
        )
    )


def load_local_hf_dataset(dataset_path: str, subset: str = None, split: str = None):
    """
    Loads a Hugging Face dataset from local Parquet files.
    Args:
        dataset_path (str): The base path to the dataset directory
        subset (str, optional): The name of the subdirectory containing the data files (e.g., "corpus").
        split (str, optional): Which split of the data to load (e.g., "train", "test").
    Returns: Dataset or DatasetDict: The loaded dataset.
    """
    if subset and split:
        dataset = datasets.load_dataset(dataset_path, subset, split=split)
    elif subset:
        dataset = datasets.load_dataset(dataset_path, subset)
    elif split:
        dataset = datasets.load_dataset(dataset_path, split=split)
    else:
        dataset = datasets.load_dataset(dataset_path)
    return dataset


def load_hf_dataset_multiple_subset(hf_path, subset_names):
    """
    Load and concatenate multiple subsets from a Hugging Face dataset (e.g. MVBench)
    """
    repo, _, split = hf_path
    subsets = []
    for subset_name in subset_names:
        dataset = _load_local_dataset_if_available(repo, subset_name, split)
        if dataset is None:
            dataset = load_dataset(repo, subset_name, split=split)
        new_column = [subset_name] * len(dataset)
        dataset = dataset.add_column("subset", new_column)
        subsets.append(dataset)
    dataset = datasets.concatenate_datasets(subsets)

    return dataset
