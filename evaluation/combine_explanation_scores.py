from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from utils.challenge_eval_utils import round_float, utc_now_iso, write_json


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_sle(value: float) -> float:
    clipped = max(-1.0, min(4.0, float(value)))
    return (clipped + 1.0) / 5.0


def build_combined_report(complex_report: Dict[str, Any], simple_report: Dict[str, Any]) -> Dict[str, Any]:
    complex_bert = float(complex_report.get("summary", {}).get("bertscore_f1_mean") or 0.0)
    simple_bert = float(simple_report.get("summary", {}).get("bertscore_f1_mean") or 0.0)
    simple_sle_raw = float(simple_report.get("summary", {}).get("simplicity_score_mean") or 0.0)

    simple_sle_normalized = _normalize_sle(simple_sle_raw)
    simple_overall = (0.7 * simple_bert) + (0.3 * simple_sle_normalized)
    explanation_score = (complex_bert + simple_overall) / 2.0

    return {
        "created_at_utc": utc_now_iso(),
        "summary": {
            "complex_bertscore_f1_mean": round_float(complex_bert),
            "simple_bertscore_f1_mean": round_float(simple_bert),
            "simple_sle_score_mean": round_float(simple_sle_raw),
            "simple_sle_normalized": round_float(simple_sle_normalized),
            "simple_overall_score": round_float(simple_overall),
            "explanation_score": round_float(explanation_score),
        },
        "notes": [
            "Simple overall is 0.7 * simple BERT F1 + 0.3 * normalized SLE.",
            "Explanation score is the mean of complex BERT F1 and simple overall score.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine separate complex/simple evaluation reports into a temporary summed score.")
    parser.add_argument("--complex-report", required=True)
    parser.add_argument("--simple-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    complex_report = _read_json(args.complex_report)
    simple_report = _read_json(args.simple_report)
    combined_report = build_combined_report(complex_report, simple_report)
    write_json(args.output, combined_report)
    print(f"Wrote combined score report to: {args.output}")
    print(json.dumps(combined_report["summary"], indent=2))


if __name__ == "__main__":
    main()
