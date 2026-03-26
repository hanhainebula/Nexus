#!/usr/bin/env python3
"""Shared helpers for exporting MMEB/VLM2Vec manifests and inventories."""

import ast
import os
import runpy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MMEB_V2_MEDIA_REPO = "TIGER-Lab/MMEB-V2"
MMEB_TEST_INSTRUCT_REPO = "ziyjiang/MMEB_Test_Instruct"
MMEB_TRAIN_REPO = "TIGER-Lab/MMEB-train"
SHAREGPT_VIDEO_REPO = "ShareGPTVideo/train_video_and_instruction"
VIDORE_TRAIN_REPO = "vidore/colpali_train_set"
VISRAG_TRAIN_REPO = "openbmb/VisRAG-Ret-Train-In-domain-data"

IMAGE_INSTRUCT_PARSERS = {"image_cls", "image_qa", "image_i2t", "image_t2i", "image_i2i_vg"}
VIDEO_LOCAL_MEDIA_PARSERS = {
    "ssv2",
    "video_classification",
    "msrvtt",
    "msvd",
    "didemo",
    "youcook2",
    "vatex",
    "moment_retrieval",
    "momentseeker",
    "videomme",
    "nextqa",
    "egoschema",
    "mvbench",
    "activitynetqa",
}


def require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "This tool requires `PyYAML`. Create an isolated environment and install Nexus dependencies there."
        ) from exc
    return yaml


def load_yaml(path: str) -> Dict[str, Any]:
    yaml = require_yaml()
    with open(path, "r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def load_python_assignment(path: str, variable_name: str):
    with open(path, "r", encoding="utf-8") as input_file:
        tree = ast.parse(input_file.read(), filename=path)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"Variable `{variable_name}` not found in {path}")


def load_python_variable_via_runpy(path: str, variable_name: str):
    namespace = runpy.run_path(path, init_globals={"__name__": "__nexus_manifest_loader__"})
    if variable_name not in namespace:
        raise ValueError(f"Variable `{variable_name}` not found in {path}")
    return namespace[variable_name]


def normalize_tuple(value):
    if not isinstance(value, (tuple, list)):
        return [value]
    return list(value)


def is_vlm2vec_root(path: Optional[str]) -> bool:
    if path in [None, ""]:
        return False
    root = Path(path).expanduser().resolve()
    required_paths = [
        root / "experiments" / "report_score_v2.py",
        root / "src" / "constant" / "dataset_hf_path.py",
        root / "experiments" / "public" / "eval" / "image.yaml",
    ]
    return all(candidate.exists() for candidate in required_paths)


def discover_vlm2vec_root(
    preferred_root: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    search_roots: Optional[Iterable[str]] = None,
) -> Optional[str]:
    env = os.environ if env is None else env
    candidates: List[str] = []

    for value in [preferred_root, env.get("VLM2VEC_ROOT")]:
        if value not in [None, ""]:
            candidates.append(value)

    if search_roots is None:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parents[1]
        search_roots = [
            str(repo_root.parent / "VLM2Vec"),
            str(repo_root.parent / "vlm2vec"),
            str(Path.cwd() / "VLM2Vec"),
            str(Path.cwd() / "vlm2vec"),
        ]

    candidates.extend(str(value) for value in search_roots if value not in [None, ""])

    seen = set()
    for candidate in candidates:
        normalized = str(Path(candidate).expanduser().resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        if is_vlm2vec_root(normalized):
            return normalized
    return None


def load_vlm2vec_context(vlm2vec_root: str) -> Dict[str, Any]:
    vlm2vec_root = os.path.abspath(vlm2vec_root)
    context = {
        "vlm2vec_root": vlm2vec_root,
        "modality_to_datasets": load_python_assignment(
            os.path.join(vlm2vec_root, "experiments", "report_score_v2.py"),
            "modality2dataset",
        ),
        "hf_mapping": load_python_assignment(
            os.path.join(vlm2vec_root, "src", "constant", "dataset_hf_path.py"),
            "EVAL_DATASET_HF_PATH",
        ),
        "local_mapping": load_python_variable_via_runpy(
            os.path.join(vlm2vec_root, "src", "constant", "dataset_hflocal_path.py"),
            "EVAL_DATASET_HF_PATH",
        ),
        "eval_configs": {
            "image": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "eval", "image.yaml")),
            "video": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "eval", "video.yaml")),
            "visdoc": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "eval", "visdoc.yaml")),
        },
        "train_configs": {
            "image": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_image.yaml")),
            "video": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_video.yaml")),
            "visdoc": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_visdoc.yaml")),
            "alltasks": load_yaml(os.path.join(vlm2vec_root, "experiments", "public", "train", "train_alltasks.yaml")),
        },
    }
    return context


def _normalize_mapping_value(value: Any) -> Tuple[Any, Any, Any]:
    values = normalize_tuple(value)
    while len(values) < 3:
        values.append(None)
    return values[0], values[1], values[2]


def infer_eval_metadata_source(
    dataset_name: str,
    dataset_parser: str,
    hf_value: Tuple[Any, Any, Any],
) -> Tuple[str, Any, Any]:
    hf_repo, hf_subset, hf_split = hf_value
    if dataset_parser in IMAGE_INSTRUCT_PARSERS:
        return MMEB_TEST_INSTRUCT_REPO, dataset_name, "test"
    return hf_repo or "", hf_subset, hf_split


def infer_eval_media_source(
    modality: str,
    dataset_parser: str,
    config: Dict[str, Any],
    metadata_repo: str,
) -> Tuple[str, str]:
    image_root = config.get("image_root")
    video_root = config.get("video_root")
    frame_root = config.get("frame_root")

    if dataset_parser in IMAGE_INSTRUCT_PARSERS and image_root:
        return MMEB_V2_MEDIA_REPO, image_root
    if dataset_parser in VIDEO_LOCAL_MEDIA_PARSERS:
        if frame_root:
            return MMEB_V2_MEDIA_REPO, frame_root
        if video_root:
            return MMEB_V2_MEDIA_REPO, video_root
    if modality == "visdoc" and dataset_parser in {"vidore", "visrag"}:
        return metadata_repo, image_root or ""
    if image_root and image_root.startswith(("image-tasks", "visdoc-tasks")):
        return MMEB_V2_MEDIA_REPO, image_root
    if frame_root and frame_root.startswith("video-tasks"):
        return MMEB_V2_MEDIA_REPO, frame_root
    if video_root and video_root.startswith("video-tasks"):
        return MMEB_V2_MEDIA_REPO, video_root
    return "", image_root or frame_root or video_root or ""


def build_eval_manifest_entries(vlm2vec_root: str) -> List[Dict[str, Any]]:
    context = load_vlm2vec_context(vlm2vec_root)
    entries: List[Dict[str, Any]] = []

    for modality, dataset_names in context["modality_to_datasets"].items():
        for dataset_key in dataset_names:
            config = context["eval_configs"][modality][dataset_key]
            dataset_name = config.get("dataset_name", dataset_key)

            hf_mapping_key = dataset_key if dataset_key in context["hf_mapping"] else dataset_name
            local_mapping_key = dataset_key if dataset_key in context["local_mapping"] else dataset_name
            hf_value = _normalize_mapping_value(context["hf_mapping"].get(hf_mapping_key, ("", "", "")))
            local_value = _normalize_mapping_value(context["local_mapping"].get(local_mapping_key, ("", "", "")))
            dataset_parser = config.get("dataset_parser", "")
            metadata_repo, metadata_subset, metadata_split = infer_eval_metadata_source(
                dataset_name=dataset_key,
                dataset_parser=dataset_parser,
                hf_value=hf_value,
            )
            media_repo, media_rel_root = infer_eval_media_source(
                modality=modality,
                dataset_parser=dataset_parser,
                config=config,
                metadata_repo=metadata_repo,
            )

            entry = {
                "modality": modality,
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "dataset_parser": dataset_parser,
                "eval_type": config.get("eval_type"),
                "metadata_hf_repo": metadata_repo,
                "metadata_hf_subset": metadata_subset,
                "metadata_hf_split": metadata_split,
                "media_hf_repo": media_repo,
                "media_rel_root": media_rel_root,
                "hf_repo": metadata_repo,
                "hf_subset": metadata_subset,
                "hf_split": metadata_split,
                "local_repo": local_value[0],
                "local_subset": local_value[1],
                "local_split": local_value[2],
                "config": {
                    key: value
                    for key, value in config.items()
                    if key not in {"dataset_name", "dataset_parser", "eval_type"}
                },
            }
            entries.append(entry)

    return entries


def _mmeb_image_train_download_patterns(source_config: Dict[str, Any]) -> List[str]:
    subset_name = source_config["subset_name"]
    dataset_split = source_config.get("dataset_split", "original")
    return [
        f"{subset_name}/{dataset_split}-*",
        f"images_zip/{subset_name}.zip",
    ]


def _sharegpt_video_patterns(source_name: str, source_config: Dict[str, Any]) -> List[str]:
    dataset_path = normalize_sharegpt_relative_path(source_config["dataset_path"])
    patterns = [dataset_path]
    if source_name in {"video_caption_300k", "video_caption_300k-video", "video_qa_240k"}:
        patterns.append("train_300k/*.tar.gz")
    return patterns


def normalize_sharegpt_relative_path(path: str) -> str:
    prefixes = [
        "vlm2vec_train/train_video_and_instruction/",
        "vlm2vec_train/video/",
        "train_video_and_instruction/",
        "video/",
    ]
    for prefix in prefixes:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def augment_train_source(modality: str, source_name: str, source_config: Dict[str, Any]) -> Dict[str, Any]:
    entry = {"source_name": source_name}
    entry.update(source_config)

    dataset_parser = source_config.get("dataset_parser", "")
    if dataset_parser == "mmeb":
        entry["metadata_hf_repo"] = MMEB_TRAIN_REPO
        entry["metadata_hf_subset"] = source_config.get("subset_name")
        entry["metadata_hf_split"] = source_config.get("dataset_split", "original")
        entry["media_hf_repo"] = MMEB_TRAIN_REPO
        entry["download_patterns"] = _mmeb_image_train_download_patterns(source_config)
        entry["local_rel_input"] = source_config.get("subset_name")
        entry["image_root_candidates"] = ["images", "image"]
    elif dataset_parser in {"llavahound_caption", "llavahound_qa"}:
        metadata_rel_path = normalize_sharegpt_relative_path(source_config["dataset_path"])
        entry["metadata_hf_repo"] = SHAREGPT_VIDEO_REPO
        entry["metadata_hf_subset"] = None
        entry["metadata_hf_split"] = None
        entry["media_hf_repo"] = SHAREGPT_VIDEO_REPO
        entry["download_patterns"] = _sharegpt_video_patterns(source_name, source_config)
        entry["metadata_rel_path"] = metadata_rel_path
        entry["local_rel_input"] = os.path.dirname(metadata_rel_path)
        entry["video_root_candidates"] = [
            "ShareGPTVideo/train_300k",
            "video/train_300k",
            "train_video_and_instruction/train_300k",
        ]
    elif dataset_parser == "vidore":
        entry["metadata_hf_repo"] = VIDORE_TRAIN_REPO
        entry["metadata_hf_subset"] = None
        entry["metadata_hf_split"] = "train"
        entry["media_hf_repo"] = VIDORE_TRAIN_REPO
        entry["download_patterns"] = ["data/*.parquet"]
        entry["local_rel_input"] = "data"
    elif dataset_parser == "visrag":
        entry["metadata_hf_repo"] = VISRAG_TRAIN_REPO
        entry["metadata_hf_subset"] = None
        entry["metadata_hf_split"] = "train"
        entry["media_hf_repo"] = VISRAG_TRAIN_REPO
        entry["download_patterns"] = ["data/*.parquet"]
        entry["local_rel_input"] = "data"
    else:
        entry["metadata_hf_repo"] = source_config.get("dataset_name", "")
        entry["metadata_hf_subset"] = None
        entry["metadata_hf_split"] = source_config.get("dataset_split")
        entry["media_hf_repo"] = ""
        entry["download_patterns"] = []
        entry["local_rel_input"] = None

    entry["train_modality"] = modality
    return entry


def build_train_manifest(train_configs: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    manifest: Dict[str, List[Dict[str, Any]]] = {}
    for modality, config in train_configs.items():
        manifest[modality] = []
        for source_name, source_config in config.items():
            manifest[modality].append(augment_train_source(modality, source_name, source_config))
    return manifest
