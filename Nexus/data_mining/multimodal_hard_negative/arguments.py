from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class MiningArguments:
    input_train_jsonl: str
    output_train_jsonl: str
    candidate_train_jsonl: Optional[str] = None
    candidate_corpus_jsonl: Optional[str] = None
    mmeb_eval_corpus_datasets: Optional[str] = None
    mmeb_eval_dataset_config: Optional[str] = None
    mmeb_data_basedir: Optional[str] = None
    mmeb_generated_media_basedir: Optional[str] = None
    mmeb_hf_cache_dir: Optional[str] = None
    mmeb_max_video_frames: Optional[int] = None
    mmeb_video_fps: Optional[float] = None
    mmeb_num_sample_per_subset: Optional[int] = None
    mode: str = "mock"
    teacher_model: Optional[str] = None
    student_model: Optional[str] = None
    student_processor_name_or_path: Optional[str] = None
    student_backbone_load_strategy: Optional[str] = None
    student_cache_dir: Optional[str] = None
    student_torch_dtype: Optional[str] = None
    student_query_max_len: Optional[int] = None
    student_passage_max_len: Optional[int] = None
    media_root: Optional[str] = None
    candidate_source: str = "all_pos"
    max_examples: int = 100
    max_candidate_examples: Optional[int] = None
    top_k: int = 50
    num_hard_negatives: int = 3
    num_medium_negatives: int = 0
    min_teacher_positive_score: Optional[float] = None
    min_teacher_negative_score: Optional[float] = None
    false_negative_margin: float = 0.02
    max_negative_score_ratio_to_positive: Optional[float] = None
    filter_duplicate_text: bool = False
    filter_duplicate_media: bool = False
    batch_size: int = 8
    query_max_len: int = 2048
    passage_max_len: int = 4096
    score_block_size: int = 4096
    score_device: Optional[str] = None
    device: str = "cuda:0"
    devices: Optional[str] = None
    torch_dtype: str = "bfloat16"
    backbone_load_strategy: str = "auto"
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    report_path: Optional[str] = None
    manifest_path: Optional[str] = None
    audit_sample_path: Optional[str] = None
    audit_sample_size: int = 20
    report_row_limit: int = 100
    dry_run: bool = False
    keep_existing_negatives: bool = False
    write_scores: bool = False
    validate_output: bool = True
    require_existing_media: bool = True
    random_seed: int = 42


def parse_args(argv: Optional[list[str]] = None) -> MiningArguments:
    parser = argparse.ArgumentParser(
        description="Mine hard negatives for Nexus multimodal retrieval training JSONL files."
    )
    parser.add_argument("--input_train_jsonl", required=True, help="Input Nexus train.jsonl path.")
    parser.add_argument("--output_train_jsonl", required=True, help="Output hard-negative train.jsonl path.")
    parser.add_argument(
        "--candidate_train_jsonl",
        default=None,
        help="Optional JSONL used only for building the candidate pool. Defaults to input_train_jsonl.",
    )
    parser.add_argument(
        "--candidate_corpus_jsonl",
        default=None,
        help=(
            "Optional comma-separated JSONL paths used as extra candidate corpus. "
            "Rows may be Nexus train-style rows, MMEB candidate-pool rows such as txt/img_path, "
            "or eval-style rows with cand_input."
        ),
    )
    parser.add_argument(
        "--mmeb_eval_corpus_datasets",
        default=None,
        help=(
            "Optional comma-separated MMEB v2 dataset names whose evaluation candidates/corpus "
            "will be appended to the candidate pool, e.g. EDIS,WebQA or VisRAG_InfoVQA."
        ),
    )
    parser.add_argument(
        "--mmeb_eval_dataset_config",
        default=None,
        help=(
            "Optional comma-separated MMEB v2 YAML config paths. When provided, dataset configs "
            "are read from these files; otherwise built-in common configs are searched."
        ),
    )
    parser.add_argument(
        "--mmeb_data_basedir",
        default=None,
        help="Optional MMEB v2 data_basedir, same meaning as evaluation DataArguments.data_basedir.",
    )
    parser.add_argument(
        "--mmeb_generated_media_basedir",
        default=None,
        help="Optional MMEB v2 generated_media_basedir for generated VisDoc page images.",
    )
    parser.add_argument(
        "--mmeb_hf_cache_dir",
        default=None,
        help=(
            "Optional Hugging Face hub cache root for MMEB v2 metadata datasets. "
            "If omitted, common server cache paths are tried automatically."
        ),
    )
    parser.add_argument(
        "--mmeb_max_video_frames",
        type=int,
        default=None,
        help="Optional max video frames passed to MMEB v2 evaluation dataset preparation.",
    )
    parser.add_argument(
        "--mmeb_video_fps",
        type=float,
        default=None,
        help="Optional video fps passed to MMEB v2 evaluation dataset preparation.",
    )
    parser.add_argument(
        "--mmeb_num_sample_per_subset",
        type=int,
        default=None,
        help="Optional subsampling for MMEB v2 corpus smoke tests. Leave unset for full corpus.",
    )
    parser.add_argument(
        "--mode",
        default="mock",
        choices=["mock", "teacher_only", "student_aware"],
        help="Mining mode. mock validates the data flow without loading a model.",
    )
    parser.add_argument("--teacher_model", default=None, help="Teacher embedding model path for teacher_only mode.")
    parser.add_argument(
        "--student_model",
        default=None,
        help=(
            "Student embedding model path for student_aware mode. This can be a base model "
            "or a PEFT/LoRA adapter directory if Nexus can resolve its base model."
        ),
    )
    parser.add_argument(
        "--student_processor_name_or_path",
        default=None,
        help="Optional processor path for student_aware mode. Defaults to --student_model.",
    )
    parser.add_argument(
        "--student_backbone_load_strategy",
        default=None,
        help="Student backbone load strategy. Defaults to --backbone_load_strategy.",
    )
    parser.add_argument("--student_cache_dir", default=None, help="Optional cache dir for the student model.")
    parser.add_argument("--student_torch_dtype", default=None, help="Optional torch dtype for the student model.")
    parser.add_argument("--student_query_max_len", type=int, default=None, help="Student query max length.")
    parser.add_argument("--student_passage_max_len", type=int, default=None, help="Student passage max length.")
    parser.add_argument(
        "--media_root",
        default=None,
        help=(
            "Optional fallback root for relative image/video/audio paths. Row-level roots "
            "such as image_root/video_root/media_root are used first."
        ),
    )
    parser.add_argument(
        "--candidate_source",
        default="all_pos",
        choices=["all_pos", "all_pos_and_existing_neg", "train_all_docs", "all_pos_plus_own_neg"],
        help=(
            "How to build the candidate pool. all_pos uses global positives only; "
            "all_pos_and_existing_neg also makes every row's existing negatives global candidates; "
            "train_all_docs is an alias for all_pos_and_existing_neg and is clearer when "
            "extra eval corpora are also appended; "
            "all_pos_plus_own_neg adds each row's own existing negatives only for that row."
        ),
    )
    parser.add_argument("--max_examples", type=int, default=100, help="Maximum query/training rows to output; <=0 means all.")
    parser.add_argument(
        "--max_candidate_examples",
        type=int,
        default=None,
        help=(
            "Maximum rows used to build the candidate pool. If omitted, reuse the query rows "
            "for backward-compatible smoke tests; <=0 means all rows from candidate_train_jsonl/input."
        ),
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-K candidates to inspect per query.")
    parser.add_argument("--num_hard_negatives", type=int, default=3, help="Number of hard negatives per row.")
    parser.add_argument("--num_medium_negatives", type=int, default=0, help="Reserved for future mixed hardness sampling.")
    parser.add_argument(
        "--min_teacher_positive_score",
        type=float,
        default=None,
        help="Drop rows whose best positive teacher score is below this threshold.",
    )
    parser.add_argument(
        "--min_teacher_negative_score",
        type=float,
        default=None,
        help="Skip candidate negatives below this teacher score. Useful for filtering out easy negatives.",
    )
    parser.add_argument(
        "--false_negative_margin",
        type=float,
        default=0.02,
        help="Skip non-positive candidates scoring above best_positive + margin.",
    )
    parser.add_argument(
        "--max_negative_score_ratio_to_positive",
        type=float,
        default=None,
        help=(
            "Optional upper bound for selected negatives relative to the best positive teacher score. "
            "For example, 0.95 keeps only negatives with teacher_score <= best_positive_score * 0.95, "
            "then still picks the hardest remaining negatives from --top_k. Leave unset to preserve old behavior."
        ),
    )
    parser.add_argument(
        "--filter_duplicate_text",
        action="store_true",
        help="Skip negatives whose normalized text matches a positive or already selected negative in the same row.",
    )
    parser.add_argument(
        "--filter_duplicate_media",
        action="store_true",
        help=(
            "Skip negatives sharing media with a positive or already selected negative in the same row. "
            "Matches exact paths, basenames, and local file md5 when available."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Embedding batch size.")
    parser.add_argument("--query_max_len", type=int, default=2048, help="Teacher query max length.")
    parser.add_argument("--passage_max_len", type=int, default=4096, help="Teacher passage max length.")
    parser.add_argument(
        "--score_block_size",
        type=int,
        default=4096,
        help=(
            "Number of candidates scored per block when mining real embeddings. "
            "This keeps full-candidate exact top-k semantics without materializing "
            "the full query x candidate score matrix; <=0 uses the old full-matrix path."
        ),
    )
    parser.add_argument(
        "--score_device",
        default=None,
        help="Optional torch device for blockwise scoring. Defaults to the first --devices entry or --device.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device for teacher encoding, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for teacher encoding, e.g. cuda:0,cuda:1. Overrides --device when set.",
    )
    parser.add_argument("--torch_dtype", default="bfloat16", help="Teacher torch dtype name.")
    parser.add_argument(
        "--backbone_load_strategy",
        default="auto",
        help=(
            "Backbone load strategy passed to Nexus MultimodalEmbedder. Use "
            "prefer_base_model if worker outputs miss hidden states."
        ),
    )
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_dir", default=None, help="HF/model cache directory.")
    parser.add_argument("--report_path", default=None, help="Optional JSON mining report path.")
    parser.add_argument("--manifest_path", default=None, help="Optional JSON manifest path.")
    parser.add_argument(
        "--audit_sample_path",
        default=None,
        help="Optional JSONL path with human-readable sampled mining decisions for manual review.",
    )
    parser.add_argument(
        "--audit_sample_size",
        type=int,
        default=20,
        help="Number of per-row audit samples to write when --audit_sample_path is set.",
    )
    parser.add_argument(
        "--report_row_limit",
        type=int,
        default=100,
        help="Maximum number of per-row decision records kept inline in report.json.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Process but make the output explicitly a dry-run artifact.")
    parser.add_argument(
        "--keep_existing_negatives",
        action="store_true",
        help="Append mined negatives after existing negatives instead of replacing neg.",
    )
    parser.add_argument("--write_scores", action="store_true", help="Write pos_scores/neg_scores for KD experiments.")
    parser.add_argument(
        "--validate_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate the mined JSONL before reporting success.",
    )
    parser.add_argument(
        "--require_existing_media",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When validating, require referenced local media files to exist.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    ns = parser.parse_args(argv)
    return MiningArguments(**vars(ns))
