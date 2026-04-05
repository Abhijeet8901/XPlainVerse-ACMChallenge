from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from utils.challenge_eval_utils import (
    align_submission_and_reference,
    build_base_report,
    extract_required_text,
    read_jsonl,
    round_float,
    summarize_bertscore,
    write_json,
)
from utils.llm_helpers import (
    build_progress_bar,
    get_bert_scorer,
    get_sle_components,
    preload_bertscorer,
    preload_sle_model,
)

DEFAULT_BERT_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
DEFAULT_BERT_LANG = "en"
DEFAULT_BERT_RESCALE_WITH_BASELINE = False
DEFAULT_ID_KEYS = ("sample_id",)
DEFAULT_SIMPLE_KEYS = ("simple_explanation",)
DEFAULT_SLE_MODEL_ID = "liamcripwell/sle-base"
DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "data" / "val_ground_truth.jsonl"


def _compute_bertscore_batch(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str,
    lang: str,
    rescale_with_baseline: bool,
    batch_size: int,
) -> List[Dict[str, float]]:
    scorer = get_bert_scorer(
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
    )
    precision, recall, f1 = scorer.score(
        list(predictions),
        list(references),
        batch_size=batch_size,
        verbose=False,
    )
    return [
        {
            "bertscore_precision": round_float(p_value),
            "bertscore_recall": round_float(r_value),
            "bertscore_f1": round_float(f_value),
        }
        for p_value, r_value, f_value in zip(precision.tolist(), recall.tolist(), f1.tolist())
    ]


def _compute_sle_scores(
    texts: Sequence[str],
    *,
    model_id: str,
    batch_size: int,
    max_length: int,
    local_files_only: bool,
    show_progress: bool,
) -> List[float]:
    import torch

    loaded = get_sle_components(
        model_id=model_id,
        local_files_only=local_files_only,
    )
    tokenizer = loaded["tokenizer"]
    model = loaded["model"]
    device = loaded["device"]

    scores: List[float] = []
    batch_starts = range(0, len(texts), batch_size)
    batch_iterator = build_progress_bar(
        batch_starts,
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="SLE scoring",
        disable=not show_progress,
    )
    with torch.inference_mode():
        for start in batch_iterator:
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1).detach().cpu()
            if logits.ndim == 0:
                scores.append(round_float(float(logits.item())) or 0.0)
            else:
                scores.extend(round_float(float(score)) or 0.0 for score in logits.tolist())
    return scores


def evaluate_simple_submission(args: argparse.Namespace) -> Dict[str, Any]:
    submission_rows = read_jsonl(args.submission)
    reference_rows = read_jsonl(args.ground_truth)
    aligned_rows, diagnostics = align_submission_and_reference(
        submission_rows,
        reference_rows,
        submission_id_keys=args.submission_id_keys,
        reference_id_keys=args.reference_id_keys,
    )

    per_sample: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    predictions: List[str] = []
    references: List[str] = []

    if getattr(args, "preload_models", True):
        preload_bertscorer(
            model_type=args.bertscore_model_type,
            lang=args.bertscore_lang,
            rescale_with_baseline=args.bertscore_rescale_with_baseline,
        )
        preload_sle_model(
            model_id=args.sle_model_id,
            local_files_only=args.sle_local_files_only,
        )

    sample_iterator = build_progress_bar(
        aligned_rows,
        total=len(aligned_rows),
        desc="Simple evaluation",
        disable=not getattr(args, "show_progress", True),
    )

    for sample_id, submission_row, reference_row in sample_iterator:
        result: Dict[str, Any] = {
            "sample_id": sample_id,
            "status": "pending",
        }
        try:
            candidate_text, candidate_key = extract_required_text(
                submission_row,
                args.submission_simple_keys,
                field_role="submission simple explanation",
                sample_id=sample_id,
            )
            reference_text, reference_key = extract_required_text(
                reference_row,
                args.reference_simple_keys,
                field_role="reference simple explanation",
                sample_id=sample_id,
            )
            result.update(
                {
                    "status": "queued",
                    "label": reference_row.get("label", submission_row.get("label")),
                    "submission_simple_key": candidate_key,
                    "reference_simple_key": reference_key,
                    "submission_simple_explanation": candidate_text,
                    "reference_simple_explanation": reference_text,
                }
            )
            valid_indices.append(len(per_sample))
            predictions.append(candidate_text)
            references.append(reference_text)
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "reason": str(exc),
                    "bertscore_precision": None,
                    "bertscore_recall": None,
                    "bertscore_f1": None,
                    "simplicity_score": None,
                }
            )
        per_sample.append(result)

    if valid_indices:
        print("Computing simple BERTScore for {0} samples...".format(len(valid_indices)))
        bertscore_rows = _compute_bertscore_batch(
            predictions,
            references,
            model_type=args.bertscore_model_type,
            lang=args.bertscore_lang,
            rescale_with_baseline=args.bertscore_rescale_with_baseline,
            batch_size=args.bertscore_batch_size,
        )
        sle_scores = _compute_sle_scores(
            predictions,
            model_id=args.sle_model_id,
            batch_size=args.sle_batch_size,
            max_length=args.sle_max_length,
            local_files_only=args.sle_local_files_only,
            show_progress=getattr(args, "show_progress", True),
        )
        for position, (result_index, bert_row) in enumerate(zip(valid_indices, bertscore_rows)):
            per_sample[result_index].update(bert_row)
            per_sample[result_index]["simplicity_score"] = round_float(sle_scores[position])
            per_sample[result_index]["status"] = "scored"

    scored_samples = [item for item in per_sample if item.get("status") == "scored"]
    simplicity_values = [item["simplicity_score"] for item in scored_samples if item.get("simplicity_score") is not None]
    summary = {
        "sample_count": len(per_sample),
        "scored_samples": len(scored_samples),
        "error_samples": sum(1 for item in per_sample if item.get("status") == "error"),
        **summarize_bertscore(scored_samples),
        "simplicity_score_mean": round_float(sum(float(value) for value in simplicity_values) / len(simplicity_values)) if simplicity_values else None,
    }

    report = build_base_report(
        metric_name="simple_explanations",
        submission_path=Path(args.submission),
        reference_path=Path(args.ground_truth),
        summary=summary,
        per_sample=per_sample,
        config={
            "submission_simple_keys": list(args.submission_simple_keys),
            "reference_simple_keys": list(args.reference_simple_keys),
            "submission_id_keys": list(args.submission_id_keys),
            "reference_id_keys": list(args.reference_id_keys),
            "bertscore_model_type": args.bertscore_model_type,
            "bertscore_lang": args.bertscore_lang,
            "bertscore_rescale_with_baseline": args.bertscore_rescale_with_baseline,
            "sle_model_id": args.sle_model_id,
            "sle_batch_size": args.sle_batch_size,
            "sle_max_length": args.sle_max_length,
            "sle_local_files_only": args.sle_local_files_only,
        },
        diagnostics=diagnostics,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate simple explanations from a submission JSONL against reference JSONL.")
    parser.add_argument("--submission", required=True, help="Path to participant submission JSONL.")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH_PATH, help="Path to validation ground-truth JSONL.")
    parser.add_argument("--output", required=True, help="Path to write the simple evaluation JSON report.")
    parser.add_argument("--submission-id-keys", nargs="+", default=list(DEFAULT_ID_KEYS))
    parser.add_argument("--reference-id-keys", nargs="+", default=list(DEFAULT_ID_KEYS))
    parser.add_argument("--submission-simple-keys", nargs="+", default=list(DEFAULT_SIMPLE_KEYS))
    parser.add_argument("--reference-simple-keys", nargs="+", default=list(DEFAULT_SIMPLE_KEYS))
    parser.add_argument("--bertscore-model-type", default=DEFAULT_BERT_MODEL_TYPE)
    parser.add_argument("--bertscore-lang", default=DEFAULT_BERT_LANG)
    parser.add_argument("--bertscore-rescale-with-baseline", action="store_true", default=DEFAULT_BERT_RESCALE_WITH_BASELINE)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--sle-model-id", default=DEFAULT_SLE_MODEL_ID)
    parser.add_argument("--sle-batch-size", type=int, default=16)
    parser.add_argument("--sle-max-length", type=int, default=512)
    parser.add_argument("--sle-local-files-only", action="store_true", default=False)
    parser.add_argument("--no-preload-models", action="store_true", default=False)
    parser.add_argument("--no-progress", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.preload_models = not args.no_preload_models
    args.show_progress = not args.no_progress
    report = evaluate_simple_submission(args)
    write_json(args.output, report)
    print(f"Wrote simple evaluation report to: {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
