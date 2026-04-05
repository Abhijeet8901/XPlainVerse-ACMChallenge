from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from utils.challenge_eval_utils import round_float, utc_now_iso, write_json


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_combined_report(complex_report: Dict[str, Any], simple_report: Dict[str, Any]) -> Dict[str, Any]:
    complex_bert = float(complex_report.get("summary", {}).get("bertscore_f1_mean") or 0.0)
    complex_fact = float(complex_report.get("summary", {}).get("fact_score_mean") or 0.0)
    simple_bert = float(simple_report.get("summary", {}).get("bertscore_f1_mean") or 0.0)
    simple_simplicity = float(simple_report.get("summary", {}).get("simplicity_score_mean") or 0.0)

    complex_score = complex_bert + complex_fact
    simple_score = simple_bert + simple_simplicity
    total_score = complex_score + simple_score

    return {
        "created_at_utc": utc_now_iso(),
        "summary": {
            "complex_bertscore_f1_mean": round_float(complex_bert),
            "complex_fact_score_mean": round_float(complex_fact),
            "simple_bertscore_f1_mean": round_float(simple_bert),
            "simple_simplicity_score_mean": round_float(simple_simplicity),
            "complex_score": round_float(complex_score),
            "simple_score": round_float(simple_score),
            "total_score": round_float(total_score),
        },
        "notes": [
            "This script currently uses plain addition.",
            "Edit this file later if you decide to change weighting, normalization, or leaderboard-specific aggregation.",
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
