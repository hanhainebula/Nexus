#!/usr/bin/env python3
"""Validate multimodal backbone family availability and tiny-checkpoint loading."""

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import transformers


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nexus.modules.multimodal import load_multimodal_backbone


FAMILY_CLASS_NAMES = {
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "llava_next": "LlavaNextForConditionalGeneration",
}


def build_qwen2_vl_case() -> Tuple[object, object]:
    config = transformers.Qwen2VLConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 128,
        },
        vision_config={
            "depth": 2,
            "embed_dim": 64,
            "hidden_size": 64,
            "hidden_act": "quick_gelu",
            "mlp_ratio": 2,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    )
    return config, transformers.Qwen2VLForConditionalGeneration


def build_qwen2_5_vl_case() -> Tuple[object, object]:
    config = transformers.Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 128,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "hidden_act": "quick_gelu",
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "tokens_per_second": 2,
            "window_size": 4,
            "fullatt_block_indexes": [1],
            "out_hidden_size": 64,
        },
    )
    return config, transformers.Qwen2_5_VLForConditionalGeneration


def build_qwen3_vl_case() -> Tuple[object, object]:
    config = transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "max_position_embeddings": 128,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 4, 4]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "hidden_act": "quick_gelu",
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
        },
        tie_word_embeddings=False,
    )
    return config, transformers.Qwen3VLForConditionalGeneration


def build_qwen3_5_case() -> Tuple[object, object]:
    config = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 128,
            "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [4, 4, 4],
                "mrope_interleaved": True,
                "rope_theta": 1000000,
                "partial_rotary_factor": 0.25,
            },
            "attn_output_gate": True,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "mamba_ssm_dtype": "float32",
            "mlp_only_layers": [],
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "hidden_act": "gelu_pytorch_tanh",
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [],
        },
        tie_word_embeddings=False,
    )
    return config, transformers.Qwen3_5ForConditionalGeneration


def build_llava_next_case() -> Tuple[object, object]:
    config = transformers.LlavaNextConfig(
        text_config=transformers.LlamaConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        ).to_dict(),
        vision_config=transformers.CLIPVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            projection_dim=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=224,
            patch_size=32,
        ).to_dict(),
        image_seq_length=49,
        tie_word_embeddings=False,
    )
    return config, transformers.LlavaNextForConditionalGeneration


CASE_BUILDERS: Dict[str, Callable[[], Tuple[object, object]]] = {
    "qwen2_vl": build_qwen2_vl_case,
    "qwen2_5_vl": build_qwen2_5_vl_case,
    "qwen3_vl": build_qwen3_vl_case,
    "qwen3_5": build_qwen3_5_case,
    "llava_next": build_llava_next_case,
}


def sync_legacy_qwen_top_level_fields(config):
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return config
    for field_name in ["hidden_size", "vocab_size"]:
        if not hasattr(config, field_name) and hasattr(text_config, field_name):
            setattr(config, field_name, getattr(text_config, field_name))
    return config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the validation report and tiny checkpoints will be written.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label for the current validation environment.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILY_CLASS_NAMES.keys()),
        choices=sorted(FAMILY_CLASS_NAMES.keys()),
        help="Subset of backbone families to validate.",
    )
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="Return a non-zero exit code when a requested family is unavailable in the current environment.",
    )
    parser.add_argument(
        "--fail-on-failure",
        action="store_true",
        help="Return a non-zero exit code when tiny-checkpoint loading fails for a requested family.",
    )
    return parser.parse_args()


def count_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def write_markdown(path: Path, payload: Dict[str, object]):
    lines = [
        "# Backbone 验证矩阵",
        "",
        f"- 标签：`{payload['label']}`",
        f"- Python：`{payload['python_executable']}`",
        f"- transformers：`{payload['transformers_version']}`",
        "",
        "| family | class_available | status | loaded_model | config_model_type | params | note |",
        "| :-- | :--: | :-- | :-- | :-- | --: | :-- |",
    ]
    for row in payload["results"]:
        lines.append(
            "| {family} | {class_available} | {status} | {loaded_model_class} | {loaded_config_model_type} | {tiny_param_count} | {note} |".format(
                family=row["family"],
                class_available="yes" if row["class_available"] else "no",
                status=row["status"],
                loaded_model_class=row.get("loaded_model_class", "-"),
                loaded_config_model_type=row.get("loaded_config_model_type", "-"),
                tiny_param_count=row.get("tiny_param_count", 0),
                note=row.get("note", "-"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_family(family: str, output_dir: Path) -> Dict[str, object]:
    class_name = FAMILY_CLASS_NAMES[family]
    result = {
        "family": family,
        "expected_model_class": class_name,
        "class_available": hasattr(transformers, class_name),
        "status": "unavailable",
        "note": "",
    }
    if not result["class_available"]:
        result["note"] = f"{class_name} is not available in transformers {transformers.__version__}"
        return result

    model = None
    loaded_model = None
    model_dir = output_dir / "tiny_checkpoints" / family
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        config, model_cls = CASE_BUILDERS[family]()
        config = sync_legacy_qwen_top_level_fields(config)
        model = model_cls(config)
        result["config_class"] = type(config).__name__
        result["tiny_param_count"] = count_parameters(model)
        model.save_pretrained(model_dir)

        loaded_model, loaded_config = load_multimodal_backbone(str(model_dir), model_type=family)
        result["status"] = "loaded"
        result["saved_model_dir"] = str(model_dir)
        result["loaded_model_class"] = type(loaded_model).__name__
        result["loaded_config_model_type"] = getattr(loaded_config, "model_type", None)
        result["note"] = "tiny checkpoint saved and reloaded successfully"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["note"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        del model
        del loaded_model
        gc.collect()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "label": args.label or output_dir.name,
        "python_executable": sys.executable,
        "transformers_version": transformers.__version__,
        "results": [validate_family(family, output_dir=output_dir) for family in args.families],
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(output_dir / "summary.md", payload)

    has_unavailable = any(not row["class_available"] for row in payload["results"])
    has_failure = any(row["status"] == "failed" for row in payload["results"])
    if (args.fail_on_unavailable and has_unavailable) or (args.fail_on_failure and has_failure):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
