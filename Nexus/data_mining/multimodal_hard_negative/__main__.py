from __future__ import annotations

from .arguments import parse_args
from .mining import run_mining


def main() -> None:
    args = parse_args()
    report = run_mining(args)
    print("HARD_NEGATIVE_MINING_DONE")
    print(f"input_rows={report['input_rows']}")
    print(f"candidate_count={report['candidate_count']}")
    print(f"output_rows={report['output_rows']}")
    print(f"total_mined_negatives={report['total_mined_negatives']}")
    print(f"avg_mined_negatives_per_row={report['avg_mined_negatives_per_row']:.4f}")
    print(f"rows_with_zero_negatives={report.get('rows_with_zero_negatives', 0)}")
    if report.get("audit_sample_path"):
        print(f"audit_sample_path={report['audit_sample_path']}")
    score_summary = report.get("score_summary", {})
    if score_summary:
        pos = score_summary.get("best_positive_score", {})
        neg = score_summary.get("mined_negative_score", {})
        margin = score_summary.get("positive_minus_negative_margin", {})
        print(f"best_positive_score_mean={pos.get('mean')}")
        print(f"mined_negative_score_mean={neg.get('mean')}")
        print(f"positive_minus_negative_margin_mean={margin.get('mean')}")


if __name__ == "__main__":
    main()
