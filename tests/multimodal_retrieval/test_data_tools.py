import importlib.util
import json
from pathlib import Path
import types
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module_from_path(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


convert_train = load_module_from_path(
    "convert_vlm2vec_train_to_nexus",
    "tools/multimodal_retrieval/convert_vlm2vec_train_to_nexus.py",
)
inventory_tool = load_module_from_path(
    "export_mmeb_v2_inventory",
    "tools/multimodal_retrieval/export_mmeb_v2_inventory.py",
)
manifest_lib = load_module_from_path(
    "vlm2vec_manifest_lib",
    "tools/multimodal_retrieval/vlm2vec_manifest_lib.py",
)
hf_tool = load_module_from_path(
    "hf_dataset_manager",
    "tools/multimodal_retrieval/hf_dataset_manager.py",
)
prepare_public_tool = load_module_from_path(
    "prepare_public_data",
    "tools/multimodal_retrieval/prepare_public_data.py",
)
prepare_train_tool = load_module_from_path(
    "prepare_mmeb_v2_train_data",
    "tools/multimodal_retrieval/prepare_mmeb_v2_train_data.py",
)
prepare_eval_tool = load_module_from_path(
    "prepare_mmeb_v2_eval_data",
    "tools/multimodal_retrieval/prepare_mmeb_v2_eval_data.py",
)
convert_eval = load_module_from_path(
    "convert_vlm2vec_eval_to_nexus",
    "tools/multimodal_retrieval/convert_vlm2vec_eval_to_nexus.py",
)
gpu_tool = load_module_from_path(
    "check_idle_gpus",
    "tools/multimodal_retrieval/check_idle_gpus.py",
)
text_eval_utils = load_module_from_path(
    "text_retrieval_utils",
    "Nexus/evaluation/text_retrieval/utils.py",
)


def test_convert_pair_style_row_to_video_item():
    args = types.SimpleNamespace(
        sequence_mode="video",
        dataset_name="MSR-VTT",
        global_dataset_name="video/msrvtt",
        media_root=None,
        image_root=None,
        video_root=None,
    )
    row = {
        "query_text": "find the matching clip",
        "query_image": {"paths": ["frames/1.jpg", "frames/2.jpg"], "bytes": [None, None]},
        "pos_text": "positive caption",
        "pos_image": {"paths": ["frames/3.jpg", "frames/4.jpg"], "bytes": [None, None]},
        "neg_text": [],
        "neg_image": [],
    }

    converted = convert_train.convert_pair_style_row(row, args)

    assert converted["query"]["video_frames"] == ["frames/1.jpg", "frames/2.jpg"]
    assert converted["pos"][0]["video_frames"] == ["frames/3.jpg", "frames/4.jpg"]
    assert converted["metadata"]["dataset_name"] == "MSR-VTT"


def test_convert_llavahound_row_prefers_raw_video():
    args = types.SimpleNamespace(
        source_format="llavahound_caption",
        dataset_name="video_caption_300k",
        global_dataset_name="video_caption_300k",
        media_root=None,
        image_root=None,
        video_root="/data/videos",
        raw_video=True,
        frame_basedir=None,
        llavahound_mode="caption_retrieval",
    )
    row = {
        "id": "sample-1",
        "video": "abc.mp4",
        "conversations": [
            {"from": "human", "value": "<video> Describe this clip"},
            {"from": "gpt", "value": "A person is cooking in the kitchen."},
        ],
    }

    converted = convert_train.convert_llavahound_row(row, args)

    assert converted["query"]["video_path"] == "abc.mp4"
    assert converted["pos"][0]["text"] == "A person is cooking in the kitchen."


def test_load_python_variable_via_runpy_supports_expressions(tmp_path):
    source_file = tmp_path / "mapping.py"
    source_file.write_text(
        "import os\nBASE='root'\nMAPPING={'toy': (os.path.join(BASE, 'images'), None, 'test')}\n",
        encoding="utf-8",
    )

    mapping = manifest_lib.load_python_variable_via_runpy(str(source_file), "MAPPING")

    assert mapping["toy"] == ("root/images", None, "test")


def test_infer_eval_metadata_source_for_image_tasks():
    metadata_repo, metadata_subset, metadata_split = manifest_lib.infer_eval_metadata_source(
        dataset_name="ImageNet-1K",
        dataset_parser="image_cls",
        hf_value=("", "", ""),
    )

    assert metadata_repo == manifest_lib.MMEB_TEST_INSTRUCT_REPO
    assert metadata_subset == "ImageNet-1K"
    assert metadata_split == "test"


def test_augment_train_source_for_mmeb_adds_download_patterns():
    source = manifest_lib.augment_train_source(
        modality="image",
        source_name="VOC2007",
        source_config={
            "dataset_parser": "mmeb",
            "dataset_name": "TIGER-Lab/MMEB-train",
            "subset_name": "VOC2007",
            "dataset_split": "original",
        },
    )

    assert source["metadata_hf_repo"] == manifest_lib.MMEB_TRAIN_REPO
    assert source["download_patterns"] == ["VOC2007/original-*", "images_zip/VOC2007.zip"]
    assert source["image_root_candidates"] == ["images", "image"]


def test_select_entries_respects_filters_and_byte_cap():
    entries = [
        {"path": "a.jsonl", "size": 10},
        {"path": "b.parquet", "size": 20},
        {"path": "images_zip/VOC2007.zip", "size": 30},
    ]

    selected = hf_tool.select_entries(
        entries,
        include_patterns=["*.jsonl", "*.zip"],
        exclude_patterns=["images_zip/*"],
        max_files=None,
        max_bytes=100,
    )

    assert [entry["path"] for entry in selected] == ["a.jsonl"]


def test_parse_next_link_extracts_cursor_url():
    headers = (
        "HTTP/2 200\n"
        "link: <https://huggingface.co/api/datasets/demo/tree/main?cursor=abc>; rel=\"next\"\n"
        "content-type: application/json\n"
    )

    next_url = hf_tool.parse_next_link(headers)

    assert next_url == "https://huggingface.co/api/datasets/demo/tree/main?cursor=abc"


def test_load_any_dataset_supports_nested_local_directory(tmp_path):
    shard_dir = tmp_path / "repo" / "data"
    shard_dir.mkdir(parents=True)
    (shard_dir / "train.jsonl").write_text(
        json.dumps(
            {
                "qry": "q",
                "qry_image_path": "images/example.png",
                "pos_text": "yes",
                "pos_image_path": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = convert_train.load_any_dataset(str(tmp_path / "repo"), subset=None, split="train", cache_dir=str(tmp_path / "cache"))

    assert len(dataset) == 1


def test_load_any_dataset_rejects_git_lfs_pointers(tmp_path):
    pointer_dir = tmp_path / "repo"
    pointer_dir.mkdir()
    (pointer_dir / "train.parquet").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\n"
        "size 123\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Git LFS pointers"):
        convert_train.load_any_dataset(str(pointer_dir), subset=None, split="train", cache_dir=str(tmp_path / "cache"))


def test_prepare_train_source_resolves_local_mmeb_subset(tmp_path):
    raw_root = tmp_path / "raw"
    subset_root = raw_root / "vlm2vec_train" / "MMEB-train" / "HatefulMemes"
    subset_root.mkdir(parents=True)
    (subset_root / "original.jsonl").write_text(
        json.dumps(
            {
                "qry": "classify meme",
                "qry_image_path": "images/HatefulMemes/example.jpg",
                "pos_text": "Yes",
                "pos_image_path": "",
                "neg_text": "No",
                "neg_image_path": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    image_root = raw_root / "vlm2vec_train" / "MMEB-train" / "image"
    image_root.mkdir(parents=True)

    entry = {
        "modality": "image",
        "source_name": "HatefulMemes",
        "dataset_parser": "mmeb",
        "dataset_name": "TIGER-Lab/MMEB-train",
        "subset_name": "HatefulMemes",
        "dataset_split": "original",
        "image_dir": "vlm2vec_train/MMEB-train/image",
    }

    resolved = prepare_train_tool.resolve_train_source(entry, raw_root)

    assert resolved["input"] == str(subset_root)
    assert resolved["split"] == "original"
    assert resolved["image_root"] == str(image_root)
    assert resolved["is_local"] is True


def test_prepare_public_data_augment_source_for_legacy_manifest():
    manifest = {
        "train": {
            "image": [
                {
                    "source_name": "VOC2007",
                    "dataset_parser": "mmeb",
                    "dataset_name": "TIGER-Lab/MMEB-train",
                    "subset_name": "VOC2007",
                    "dataset_split": "original",
                }
            ]
        }
    }

    selected = prepare_public_tool.select_train_sources(
        manifest,
        modalities=["image"],
        source_names=[],
    )

    assert selected[0]["metadata_hf_repo"] == manifest_lib.MMEB_TRAIN_REPO
    assert selected[0]["download_patterns"] == ["VOC2007/original-*", "images_zip/VOC2007.zip"]


def test_discover_vlm2vec_root_prefers_explicit_candidate(tmp_path):
    vlm2vec_root = tmp_path / "VLM2Vec"
    (vlm2vec_root / "experiments" / "public" / "eval").mkdir(parents=True)
    (vlm2vec_root / "src" / "constant").mkdir(parents=True)
    (vlm2vec_root / "experiments" / "report_score_v2.py").write_text("modality2dataset = {}", encoding="utf-8")
    (vlm2vec_root / "src" / "constant" / "dataset_hf_path.py").write_text("EVAL_DATASET_HF_PATH = {}", encoding="utf-8")
    (vlm2vec_root / "experiments" / "public" / "eval" / "image.yaml").write_text("{}", encoding="utf-8")

    discovered = manifest_lib.discover_vlm2vec_root(search_roots=[str(vlm2vec_root)], env={})

    assert discovered == str(vlm2vec_root.resolve())


def test_prepare_eval_source_resolves_local_image_task(tmp_path):
    raw_root = tmp_path / "raw"
    metadata_dir = raw_root / "vlm2vec_eval" / "metadata" / "ziyjiang--MMEB_Test_Instruct" / "HatefulMemes"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "test.jsonl").write_text("{}\n", encoding="utf-8")
    image_root = raw_root / "image-tasks"
    image_root.mkdir(parents=True)

    entry = {
        "dataset_key": "HatefulMemes",
        "modality": "image",
        "metadata_hf_repo": "ziyjiang/MMEB_Test_Instruct",
        "metadata_hf_subset": "HatefulMemes",
        "metadata_hf_split": "test",
        "media_hf_repo": "TIGER-Lab/MMEB-V2",
        "media_rel_root": "image-tasks/",
        "local_repo": "image-tasks",
        "local_subset": "HatefulMemes",
        "local_split": "test",
        "config": {"image_root": "image-tasks/"},
    }

    resolved = prepare_eval_tool.resolve_eval_source(entry, raw_root)

    assert resolved["input"] == str(metadata_dir)
    assert resolved["is_local"] is True
    assert resolved["image_root"] == str(image_root)
    assert resolved["media_root"] == str(image_root)


def test_prepare_eval_source_prefers_frame_root_for_video(tmp_path):
    raw_root = tmp_path / "raw"
    metadata_dir = raw_root / "vlm2vec_eval" / "metadata" / "VLM2Vec--MSVD"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "test.jsonl").write_text("{}\n", encoding="utf-8")
    frame_root = raw_root / "video-tasks" / "frames" / "video_ret" / "MSVD"
    frame_root.mkdir(parents=True)
    video_root = raw_root / "video-tasks" / "videos" / "video_ret" / "MSVD"
    video_root.mkdir(parents=True)

    entry = {
        "dataset_key": "MSVD",
        "modality": "video",
        "metadata_hf_repo": "VLM2Vec/MSVD",
        "metadata_hf_subset": None,
        "metadata_hf_split": "test",
        "media_hf_repo": "TIGER-Lab/MMEB-V2",
        "media_rel_root": "video-tasks/frames/video_ret/MSVD/",
        "local_repo": "video-tasks/MSVD",
        "local_subset": None,
        "local_split": "test",
        "config": {
            "frame_root": "video-tasks/frames/video_ret/MSVD/",
            "video_root": "video-tasks/videos/video_ret/MSVD/",
        },
    }

    resolved = prepare_eval_tool.resolve_eval_source(entry, raw_root)

    assert resolved["input"] == str(metadata_dir)
    assert resolved["image_root"] == str(frame_root)
    assert resolved["video_root"] == str(video_root)
    assert resolved["media_root"] == str(frame_root)


def test_build_eval_convert_command_uses_remote_repo_when_allowed(tmp_path):
    args = types.SimpleNamespace(
        python_bin="python",
        cache_dir=str(tmp_path / "cache"),
        max_rows_per_dataset=7,
        local_only=False,
    )
    entry = {
        "dataset_key": "ViDoRe_arxivqa",
        "modality": "visdoc",
        "metadata_hf_repo": "vidore/arxivqa_test_subsampled_beir",
    }
    resolved = {
        "input": None,
        "subset": None,
        "split": "test",
        "media_root": str(tmp_path / "raw"),
        "image_root": str(tmp_path / "raw" / "visdoc-tasks" / "ViDoRe_arxivqa"),
        "video_root": None,
    }

    command = prepare_eval_tool.build_convert_command(
        entry=entry,
        resolved=resolved,
        output_dir=tmp_path / "out" / "ViDoRe_arxivqa",
        args=args,
    )

    assert command[:4] == ["python", str(prepare_eval_tool.SCRIPT_DIR / "convert_vlm2vec_eval_to_nexus.py"), "--input", "vidore/arxivqa_test_subsampled_beir"]
    assert "--input-format" in command
    assert "beir_configs" in command
    assert "--sequence-mode" in command
    assert "image" in command
    assert "--max-rows" in command
    assert "7" in command


def test_build_eval_config_writes_single_dataset_config(tmp_path):
    entry = {"dataset_key": "MSVD"}
    resolved = {
        "split": "test",
        "media_root": str(tmp_path / "raw"),
        "image_root": str(tmp_path / "raw" / "frames"),
        "video_root": str(tmp_path / "raw" / "videos"),
    }

    config = prepare_eval_tool.build_eval_config(entry=entry, output_root=tmp_path / "eval_ready", resolved=resolved)

    assert config["dataset_dir"] == str(tmp_path / "eval_ready")
    assert config["dataset_names"] == ["MSVD"]
    assert config["splits"] == ["test"]
    assert config["image_root"] == str(tmp_path / "raw" / "frames")
    assert config["cache_path"] == str(tmp_path / "eval_ready" / "_cache" / "hf_datasets")


def test_write_stage_configs_uses_local_runtime_cache(tmp_path):
    write_dir = tmp_path / "configs"
    summary_rows = [
        {
            "source_name": "toy-image",
            "modality": "image",
            "status": "converted",
            "output": str(tmp_path / "prepared" / "toy-image.jsonl"),
        }
    ]

    prepare_train_tool.write_stage_configs(write_dir, summary_rows)

    config = json.loads((write_dir / "stage_a_data_config.json").read_text(encoding="utf-8"))

    assert config["train_data"] == [str(tmp_path / "prepared" / "toy-image.jsonl")]
    assert config["cache_path"] == str(tmp_path / "_cache" / "stage_a")


def test_convert_instruction_style_eval_row_for_image_cls():
    args = types.SimpleNamespace(
        sequence_mode="auto",
        dataset_name="HatefulMemes",
        query_prefix="q",
    )
    row = {
        "qry_text": "",
        "qry_img_path": "HatefulMemes/image_0.jpg",
        "tgt_text": ["Yes", "No"],
        "tgt_img_path": ["", ""],
        "qry_inst": "<|image_1|>\nRepresent the given image for binary classification.",
        "tgt_inst": "",
    }

    query_record, corpus_records, qrels = convert_eval.convert_instruction_style_row(0, row, args)

    assert query_record["_id"] == "q0"
    assert query_record["image_path"] == "HatefulMemes/image_0.jpg"
    assert "Represent the given image" in query_record["text"]
    assert [record["_id"] for record in corpus_records] == ["Yes", "No"]
    assert qrels == [{"query-id": "q0", "corpus-id": "Yes", "score": 1}]


def test_convert_instruction_style_eval_row_for_image_t2i():
    args = types.SimpleNamespace(
        sequence_mode="auto",
        dataset_name="ImageSearch",
        query_prefix="q",
    )
    row = {
        "qry_inst": "Find the matching image",
        "qry_text": "a cat on a sofa",
        "tgt_inst": "",
        "tgt_text": ["", ""],
        "tgt_img_path": ["images/cat.jpg", "images/dog.jpg"],
    }

    query_record, corpus_records, qrels = convert_eval.convert_instruction_style_row(1, row, args)

    assert query_record["_id"] == "q1"
    assert query_record["text"] == "Find the matching image a cat on a sofa"
    assert corpus_records[0]["_id"] == "images/cat.jpg"
    assert corpus_records[0]["image_path"] == "images/cat.jpg"
    assert qrels == [{"query-id": "q1", "corpus-id": "images/cat.jpg", "score": 1}]


def test_serialize_image_like_supports_pil_images():
    from PIL import Image

    image = Image.new("RGB", (2, 2), color=(255, 0, 0))
    payload = convert_eval.serialize_image_like(image)

    assert isinstance(payload, dict)
    assert "b64" in payload


def test_check_idle_gpus_parses_csv_dump():
    gpus = gpu_tool.parse_gpu_lines(
        [
            "index, memory.used, utilization.gpu",
            "0, 15 MiB, 0 %",
            "1, 2048 MiB, 35 %",
        ]
    )

    assert gpus == [
        {"index": 0, "memory_used": 15, "utilization": 0},
        {"index": 1, "memory_used": 2048, "utilization": 35},
    ]


def test_check_idle_gpus_query_falls_back_when_nounits_format_fails(monkeypatch):
    commands = []

    class FakeCompletedProcess:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, check, capture_output, text):
        commands.append(command)
        if command[-1] == "--format=csv,noheader,nounits":
            raise gpu_tool.subprocess.CalledProcessError(returncode=255, cmd=command)
        return FakeCompletedProcess("0, 15 MiB, 0 %\n1, 1024 MiB, 5 %\n")

    monkeypatch.setattr(gpu_tool.subprocess, "run", fake_run)

    gpus = gpu_tool.query_gpus()

    assert gpus == [
        {"index": 0, "memory_used": 15, "utilization": 0},
        {"index": 1, "memory_used": 1024, "utilization": 5},
    ]
    assert commands[0][-1] == "--format=csv,noheader,nounits"
    assert commands[1][-1] == "--format=csv,noheader"


def test_text_eval_utils_fallback_to_numpy_when_faiss_is_missing(monkeypatch):
    monkeypatch.setattr(text_eval_utils, "faiss", None)

    index = text_eval_utils.index(
        corpus_embeddings=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    scores, indices = text_eval_utils.search(
        faiss_index=index,
        k=1,
        query_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    assert scores.shape == (1, 1)
    assert indices.tolist() == [[0]]
