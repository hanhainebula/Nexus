#!/usr/bin/env python3
"""Run runtime validation for the multimodal embedding stack."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nexus import MultimodalEmbedder
from Nexus.evaluation.multimodal_retrieval.arguments import (
    MultimodalRetrievalEvalArgs,
    MultimodalRetrievalEvalModelArgs,
)
from Nexus.evaluation.multimodal_retrieval.runner import MultimodalRetrievalEvalRunner


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", required=True, help="Local base multimodal model path.")
    parser.add_argument(
        "--adapter-model-path",
        default=None,
        help="Optional LoRA / PEFT output directory to validate for inference and evaluation.",
    )
    parser.add_argument("--processor-path", default=None, help="Optional processor path for the base model.")
    parser.add_argument("--model-type", default="qwen2_vl", help="Explicit multimodal model type.")
    parser.add_argument("--single-device", default="cuda:0", help="Device used for single-device runtime checks.")
    parser.add_argument(
        "--multi-devices",
        nargs="+",
        default=None,
        help="Optional device list used for multi-device consistency checks.",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", help="Torch dtype string for runtime loading.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument(
        "--dataset-dir",
        default=str(REPO_ROOT / "examples" / "multimodal_retrieval" / "data" / "eval"),
        help="Local evaluation dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where runtime validation artifacts and summaries will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size used for embedding inference and evaluation.",
    )
    return parser.parse_args()


def sample_queries(repo_root: Path) -> List[Dict[str, object]]:
    media_root = repo_root / "examples" / "multimodal_retrieval" / "data" / "media"
    return [
        {
            "text": "find the matching color grid",
            "image_path": str(media_root / "query.ppm"),
        },
        {
            "text": "retrieve the reference grid with a red corner",
        },
        {
            "text": "find the matching color grid",
            "image_path": str(media_root / "query.ppm"),
        },
        {
            "text": "retrieve the reference grid with a red corner",
        },
    ]


def sample_passages(repo_root: Path) -> List[Dict[str, object]]:
    media_root = repo_root / "examples" / "multimodal_retrieval" / "data" / "media"
    return [
        {
            "text": "the reference grid with a red corner and green center",
            "image_path": str(media_root / "doc.ppm"),
        },
        {
            "text": "an alternate grid dominated by blue pixels",
            "image_path": str(media_root / "neg.ppm"),
        },
        {
            "text": "the reference grid with a red corner and green center",
            "image_path": str(media_root / "doc.ppm"),
        },
        {
            "text": "an alternate grid dominated by blue pixels",
            "image_path": str(media_root / "neg.ppm"),
        },
    ]


def build_embedder(
    model_path: str,
    processor_path: Optional[str],
    model_type: str,
    devices,
    batch_size: int,
    trust_remote_code: bool,
    torch_dtype: str,
):
    return MultimodalEmbedder(
        model_name_or_path=model_path,
        processor_name_or_path=processor_path,
        model_type=model_type,
        devices=devices,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        normalize_embeddings=True,
        pooling_method="last_token",
        use_chat_template=True,
        torch_dtype=torch_dtype,
        convert_to_numpy=True,
    )


def run_inference_validation(
    model_path: str,
    processor_path: Optional[str],
    model_type: str,
    single_device: str,
    multi_devices: Optional[List[str]],
    batch_size: int,
    trust_remote_code: bool,
    torch_dtype: str,
) -> Dict[str, object]:
    queries = sample_queries(REPO_ROOT)
    passages = sample_passages(REPO_ROOT)

    single_model = build_embedder(
        model_path=model_path,
        processor_path=processor_path,
        model_type=model_type,
        devices=[single_device],
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    single_query_embeddings = single_model.encode_queries(queries)
    single_passage_embeddings = single_model.encode_corpus(passages)
    single_scores = single_query_embeddings @ single_passage_embeddings.T
    single_model.stop_self_pool()

    result = {
        "model_path": model_path,
        "single_device": single_device,
        "query_shape": list(single_query_embeddings.shape),
        "passage_shape": list(single_passage_embeddings.shape),
        "single_score_matrix": single_scores.tolist(),
    }

    if multi_devices is not None and len(multi_devices) > 1:
        multi_model = build_embedder(
            model_path=model_path,
            processor_path=processor_path,
            model_type=model_type,
            devices=multi_devices,
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        multi_query_embeddings = multi_model.encode_queries(queries)
        multi_passage_embeddings = multi_model.encode_corpus(passages)
        multi_scores = multi_query_embeddings @ multi_passage_embeddings.T
        multi_model.stop_self_pool()

        result["multi_device"] = {
            "devices": multi_devices,
            "query_shape": list(multi_query_embeddings.shape),
            "passage_shape": list(multi_passage_embeddings.shape),
            "score_matrix": multi_scores.tolist(),
            "query_allclose": bool(np.allclose(single_query_embeddings, multi_query_embeddings, atol=1e-4)),
            "passage_allclose": bool(np.allclose(single_passage_embeddings, multi_passage_embeddings, atol=1e-4)),
            "max_query_abs_diff": float(np.max(np.abs(single_query_embeddings - multi_query_embeddings))),
            "max_passage_abs_diff": float(np.max(np.abs(single_passage_embeddings - multi_passage_embeddings))),
        }

    return result


def collect_eval_results(eval_output_dir: Path) -> Dict[str, object]:
    collected = {}
    for result_path in sorted(eval_output_dir.rglob("eval_results.json")):
        relative_path = result_path.relative_to(eval_output_dir).as_posix()
        with result_path.open("r", encoding="utf-8") as input_file:
            collected[relative_path] = json.load(input_file)
    return collected


def run_eval_validation(
    model_path: str,
    processor_path: Optional[str],
    model_type: str,
    single_device: str,
    batch_size: int,
    trust_remote_code: bool,
    torch_dtype: str,
    dataset_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.md"
    eval_args = MultimodalRetrievalEvalArgs(
        eval_name="stage1_runtime_validation",
        dataset_dir=str(dataset_dir),
        media_root=None,
        image_root=None,
        video_root=None,
        dataset_names=None,
        splits=["test"],
        corpus_embd_save_dir=str(output_dir / "cache"),
        search_top_k=10,
        cache_path=str(output_dir / "hf_cache"),
        ignore_identical_ids=False,
        force_redownload=False,
        overwrite=True,
        eval_output_dir=str(output_dir / "results"),
        eval_output_method="markdown",
        eval_output_path=str(summary_path),
        eval_metrics=["ndcg_at_10", "recall_at_10"],
        k_values=[1, 3, 5, 10],
    )
    model_args = MultimodalRetrievalEvalModelArgs(
        embedder_name_or_path=model_path,
        processor_name_or_path=processor_path,
        model_type=model_type,
        normalize_embeddings=True,
        pooling_method="last_token",
        use_fp16=False,
        devices=[single_device],
        query_instruction_for_retrieval=None,
        query_instruction_format_for_retrieval="{}{}",
        passage_instruction_for_retrieval=None,
        passage_instruction_format_for_retrieval="{}{}",
        trust_remote_code=trust_remote_code,
        cache_dir=None,
        token=None,
        embedder_batch_size=batch_size,
        embedder_query_max_length=512,
        embedder_passage_max_length=1024,
        use_chat_template=True,
        torch_dtype=torch_dtype,
        attn_implementation=None,
    )
    runner = MultimodalRetrievalEvalRunner(eval_args=eval_args, model_args=model_args)
    runner.run()
    return {
        "summary_path": str(summary_path),
        "summary_exists": summary_path.exists(),
        "summary_preview": summary_path.read_text(encoding="utf-8")[:1000] if summary_path.exists() else "",
        "eval_results": collect_eval_results(Path(eval_args.eval_output_dir)),
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    report = {
        "base_model": run_inference_validation(
            model_path=args.base_model_path,
            processor_path=args.processor_path or args.base_model_path,
            model_type=args.model_type,
            single_device=args.single_device,
            multi_devices=args.multi_devices,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
        ),
        "base_eval": run_eval_validation(
            model_path=args.base_model_path,
            processor_path=args.processor_path or args.base_model_path,
            model_type=args.model_type,
            single_device=args.single_device,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            dataset_dir=Path(args.dataset_dir),
            output_dir=output_root / "base_eval",
        ),
    }

    if args.adapter_model_path:
        report["adapter_model"] = run_inference_validation(
            model_path=args.adapter_model_path,
            processor_path=args.adapter_model_path,
            model_type=args.model_type,
            single_device=args.single_device,
            multi_devices=None,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
        )
        report["adapter_eval"] = run_eval_validation(
            model_path=args.adapter_model_path,
            processor_path=args.adapter_model_path,
            model_type=args.model_type,
            single_device=args.single_device,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            dataset_dir=Path(args.dataset_dir),
            output_dir=output_root / "adapter_eval",
        )

    report_path = output_root / "runtime_validation_report.json"
    with report_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote runtime validation report to {report_path}")


if __name__ == "__main__":
    main()
