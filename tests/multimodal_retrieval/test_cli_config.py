import json
import inspect
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nexus.evaluation.multimodal_retrieval.__main__ import parse_cli_args as parse_eval_cli_args
from Nexus.training.embedder.multimodal_retrieval.__main__ import parse_cli_args as parse_train_cli_args


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_train_cli_accepts_split_json_configs(tmp_path):
    model_config = tmp_path / "model.json"
    data_config = tmp_path / "data.json"
    training_config = tmp_path / "training.json"
    train_data = tmp_path / "dataset" / "train.jsonl"
    train_data.parent.mkdir(parents=True)
    train_data.write_text("{}", encoding="utf-8")

    write_json(
        model_config,
        {
            "model_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
            "processor_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
        },
    )
    write_json(
        data_config,
        {
            "train_data": ["dataset/train.jsonl"],
        },
    )
    write_json(
        training_config,
        {
            "output_dir": "output",
            "overwrite_output_dir": True,
        },
    )

    model_args, data_args, training_args = parse_train_cli_args(
        [
            "--model_config",
            str(model_config),
            "--data_config",
            str(data_config),
            "--training_config",
            str(training_config),
        ]
    )

    assert model_args.model_name_or_path == "Qwen/Qwen2-VL-2B-Instruct"
    assert data_args.train_data == [str(train_data.resolve())]
    assert training_args.output_dir == str(tmp_path / "output")


def test_eval_cli_accepts_json_configs_without_onnx(tmp_path):
    eval_config = tmp_path / "eval.json"
    model_config = tmp_path / "model.json"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    write_json(
        eval_config,
        {
            "eval_name": "toy_eval",
            "dataset_dir": "dataset",
            "splits": ["test"],
            "k_values": [1, 10],
        },
    )
    write_json(
        model_config,
        {
            "embedder_name_or_path": "./model_dir",
            "devices": ["cuda:0"],
        },
    )

    eval_args, model_args = parse_eval_cli_args(
        [
            "--eval_config",
            str(eval_config),
            "--model_config",
            str(model_config),
        ]
    )

    assert eval_args.eval_name == "toy_eval"
    assert eval_args.dataset_dir == str(dataset_dir.resolve())
    assert eval_args.splits == ["test"]
    assert eval_args.k_values == [1, 10]
    assert model_args.embedder_name_or_path == str(model_dir.resolve())
    assert model_args.devices == ["cuda:0"]


def test_train_cli_accepts_processor_overrides_in_json_config(tmp_path):
    model_config = tmp_path / "model.json"
    data_config = tmp_path / "data.json"
    training_config = tmp_path / "training.json"
    train_data = tmp_path / "dataset" / "train.jsonl"
    train_data.parent.mkdir(parents=True)
    train_data.write_text("{}", encoding="utf-8")

    write_json(
        model_config,
        {
            "model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
            "processor_name_or_path": "Qwen/Qwen3-VL-2B-Instruct",
            "backbone_load_strategy": "prefer_base_model",
            "processor_kwargs": {"max_pixels": 262144},
            "processor_call_kwargs": {"size": {"shortest_edge": 448}},
        },
    )
    write_json(
        data_config,
        {
            "train_data": ["dataset/train.jsonl"],
        },
    )
    write_json(
        training_config,
        {
            "output_dir": "output",
            "overwrite_output_dir": True,
        },
    )

    model_args, _, _ = parse_train_cli_args(
        [
            "--model_config",
            str(model_config),
            "--data_config",
            str(data_config),
            "--training_config",
            str(training_config),
        ]
    )

    assert model_args.backbone_load_strategy == "prefer_base_model"
    assert model_args.processor_kwargs == {"max_pixels": 262144}
    assert model_args.processor_call_kwargs == {"size": {"shortest_edge": 448}}


def test_training_import_patches_accelerate_unwrap_model_signature():
    from accelerate import Accelerator

    signature = inspect.signature(Accelerator.unwrap_model)

    assert "keep_torch_compile" in signature.parameters
