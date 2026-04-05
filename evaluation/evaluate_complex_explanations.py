from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from utils.challenge_eval_utils import (
    align_submission_and_reference,
    build_base_report,
    extract_required_text,
    load_cached_reference,
    read_jsonl,
    round_float,
    save_cached_reference,
    summarize_bertscore,
    write_json,
)
from utils.llm_helpers import (
    build_progress_bar,
    chat_completion,
    compute_coverage_summary,
    extract_first_json,
    get_coverage_claim_matches,
    get_coverage_entity_matches,
    get_bert_scorer,
    get_reference_claims,
    get_reference_entities,
    load_text,
    preload_bertscorer,
    preload_chat_model,
)

DEFAULT_SYSTEM_PROMPT_EXTRACTION = "You are a careful information extraction assistant. Return JSON only."
DEFAULT_SYSTEM_PROMPT_COVERAGE = "You are a careful semantic coverage assistant. Return JSON only."
DEFAULT_BERT_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
DEFAULT_BERT_LANG = "en"
DEFAULT_BERT_RESCALE_WITH_BASELINE = False
DEFAULT_ID_KEYS = ("sample_id",)
DEFAULT_COMPLEX_KEYS = ("complex_explanation",)
DEFAULT_FACT_SCORE_MODE = "mean_entity_and_claim"
DEFAULT_INFERENCE_BACKEND = "transformers"
DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"
DEFAULT_EXTRACTION_MAX_TOKENS = 1024
DEFAULT_COVERAGE_MAX_TOKENS = 1024
DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "data" / "val_ground_truth.jsonl"


def _compute_mean_if_all_present(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return round_float(sum(float(value) for value in values) / len(values))


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
    results: List[Dict[str, float]] = []
    for p_value, r_value, f_value in zip(precision.tolist(), recall.tolist(), f1.tolist()):
        results.append(
            {
                "bertscore_precision": round_float(p_value),
                "bertscore_recall": round_float(r_value),
                "bertscore_f1": round_float(f_value),
            }
        )
    return results


def _extract_reference_payload(
    *,
    sample_id: str,
    reference_text: str,
    prompt_template: str,
    backend: str,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    device_map: str,
    torch_dtype: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
    cache_dir: str | None,
    enable_thinking: bool,
) -> Dict[str, Any]:
    user_prompt = prompt_template.replace("{{EXPLANATION_TEXT}}", reference_text)
    raw_response = chat_completion(
        backend=backend,
        model=model_name,
        system_prompt=DEFAULT_SYSTEM_PROMPT_EXTRACTION,
        user_prompt=user_prompt,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
        enable_thinking=enable_thinking,
    )
    parsed = extract_first_json(raw_response)
    try:
        diagnostic_entities = get_reference_entities(parsed)
        evidence_claims = get_reference_claims(parsed)
    except Exception as exc:
        raw_preview = raw_response[:1200].replace("\n", "\\n")
        raise ValueError(
            "Reference extraction returned an invalid JSON shape for sample '{0}'. "
            "Parsed top-level type: {1}. Raw response preview: {2!r}"
            .format(sample_id, type(parsed).__name__, raw_preview)
        ) from exc
    if not isinstance(diagnostic_entities, list):
        raise ValueError(f"Reference extraction returned an invalid diagnostic_entities list for sample '{sample_id}'.")
    if not isinstance(evidence_claims, list):
        raise ValueError(f"Reference extraction returned an invalid evidence_claims list for sample '{sample_id}'.")
    return {
        "sample_id": sample_id,
        "reference_text_sha256": __import__("hashlib").sha256(reference_text.encode("utf-8")).hexdigest(),
        "explanation": reference_text,
        "diagnostic_entities": diagnostic_entities,
        "evidence_claims": evidence_claims,
        "raw_response": raw_response,
    }


def _check_semantic_coverage(
    *,
    sample_id: str,
    reference_payload: Dict[str, Any],
    candidate_text: str,
    prompt_template: str,
    backend: str,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    device_map: str,
    torch_dtype: str,
    trust_remote_code: bool,
    attn_implementation: str | None,
    cache_dir: str | None,
    enable_thinking: bool,
) -> Dict[str, Any]:
    reference_json_for_prompt = {
        "diagnostic_entities": get_reference_entities(reference_payload),
        "evidence_claims": get_reference_claims(reference_payload),
    }
    user_prompt = prompt_template.replace(
        "{{REFERENCE_JSON}}",
        json.dumps(reference_json_for_prompt, indent=2, ensure_ascii=True),
    ).replace("{{CANDIDATE_EXPLANATION}}", candidate_text)

    raw_response = chat_completion(
        backend=backend,
        model=model_name,
        system_prompt=DEFAULT_SYSTEM_PROMPT_COVERAGE,
        user_prompt=user_prompt,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
        enable_thinking=enable_thinking,
    )
    parsed = extract_first_json(raw_response)
    try:
        entity_matches = get_coverage_entity_matches(parsed)
        claim_matches = get_coverage_claim_matches(parsed)
    except Exception as exc:
        raw_preview = raw_response[:1200].replace("\n", "\\n")
        raise ValueError(
            "Coverage check returned an invalid JSON shape for sample '{0}'. "
            "Parsed top-level type: {1}. Raw response preview: {2!r}"
            .format(sample_id, type(parsed).__name__, raw_preview)
        ) from exc
    if not isinstance(entity_matches, list):
        raise ValueError(f"Coverage model returned an invalid entity_matches list for sample '{sample_id}'.")
    if not isinstance(claim_matches, list):
        raise ValueError(f"Coverage model returned an invalid claim_matches list for sample '{sample_id}'.")
    summary = compute_coverage_summary(entity_matches, claim_matches)
    return {
        "reference_json_for_prompt": reference_json_for_prompt,
        "entity_matches": entity_matches,
        "claim_matches": claim_matches,
        "summary": summary,
        "raw_response": raw_response,
    }


def _compute_fact_score(summary: Dict[str, Any], mode: str) -> float:
    entity_coverage = float(summary.get("entity_coverage", 0.0))
    claim_coverage = float(summary.get("claim_coverage", summary.get("fact_coverage", 0.0)))
    if mode == "claim_only":
        return claim_coverage
    if mode == "entity_only":
        return entity_coverage
    if mode == "mean_entity_and_claim":
        return (entity_coverage + claim_coverage) / 2.0
    raise ValueError(f"Unsupported fact score mode: {mode}")


def evaluate_complex_submission(args: argparse.Namespace) -> Dict[str, Any]:
    submission_rows = read_jsonl(args.submission)
    reference_rows = read_jsonl(args.ground_truth)
    aligned_rows, diagnostics = align_submission_and_reference(
        submission_rows,
        reference_rows,
        submission_id_keys=args.submission_id_keys,
        reference_id_keys=args.reference_id_keys,
    )

    extraction_prompt = load_text(args.entity_fact_prompt)
    coverage_prompt = load_text(args.semantic_coverage_prompt)
    reference_cache_dir = getattr(args, "reference_cache_dir", None)
    cache_dir = Path(reference_cache_dir) if reference_cache_dir else None

    if getattr(args, "preload_models", True):
        preload_chat_model(
            backend=args.backend,
            model=args.model_name,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            cache_dir=args.hf_cache_dir,
        )
        preload_bertscorer(
            model_type=args.bertscore_model_type,
            lang=args.bertscore_lang,
            rescale_with_baseline=args.bertscore_rescale_with_baseline,
        )

    pending_predictions: List[str] = []
    pending_references: List[str] = []
    per_sample: List[Dict[str, Any]] = []

    sample_iterator = build_progress_bar(
        aligned_rows,
        total=len(aligned_rows),
        desc="Complex evaluation",
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
                args.submission_complex_keys,
                field_role="submission complex explanation",
                sample_id=sample_id,
            )
            reference_text, reference_key = extract_required_text(
                reference_row,
                args.reference_complex_keys,
                field_role="reference complex explanation",
                sample_id=sample_id,
            )

            cached_payload = load_cached_reference(cache_dir, sample_id, reference_text)
            if cached_payload is None:
                reference_payload = _extract_reference_payload(
                    sample_id=sample_id,
                    reference_text=reference_text,
                    prompt_template=extraction_prompt,
                    backend=args.backend,
                    model_name=args.model_name,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    temperature=args.temperature,
                    max_tokens=args.extraction_max_tokens,
                    timeout=args.request_timeout_seconds,
                    device_map=args.device_map,
                    torch_dtype=args.torch_dtype,
                    trust_remote_code=args.trust_remote_code,
                    attn_implementation=args.attn_implementation,
                    cache_dir=args.hf_cache_dir,
                    enable_thinking=args.enable_thinking,
                )
                save_cached_reference(cache_dir, sample_id, reference_payload)
            else:
                reference_payload = cached_payload

            coverage_payload = _check_semantic_coverage(
                sample_id=f"{sample_id}::gt_to_submission",
                reference_payload=reference_payload,
                candidate_text=candidate_text,
                prompt_template=coverage_prompt,
                backend=args.backend,
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.coverage_max_tokens,
                timeout=args.request_timeout_seconds,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
                attn_implementation=args.attn_implementation,
                cache_dir=args.hf_cache_dir,
                enable_thinking=args.enable_thinking,
            )
            coverage_summary = coverage_payload["summary"]
            entity_coverage_gt_to_submission = round_float(coverage_summary.get("entity_coverage", 0.0))
            claim_coverage_gt_to_submission = round_float(
                coverage_summary.get("claim_coverage", coverage_summary.get("fact_coverage", 0.0))
            )
            fact_score_gt_to_submission = round_float(_compute_fact_score(coverage_summary, args.fact_score_mode))

            submission_payload = _extract_reference_payload(
                sample_id=f"{sample_id}::submission",
                reference_text=candidate_text,
                prompt_template=extraction_prompt,
                backend=args.backend,
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.extraction_max_tokens,
                timeout=args.request_timeout_seconds,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
                attn_implementation=args.attn_implementation,
                cache_dir=args.hf_cache_dir,
                enable_thinking=args.enable_thinking,
            )
            reverse_coverage_payload = _check_semantic_coverage(
                sample_id=f"{sample_id}::submission_to_gt",
                reference_payload=submission_payload,
                candidate_text=reference_text,
                prompt_template=coverage_prompt,
                backend=args.backend,
                model_name=args.model_name,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.coverage_max_tokens,
                timeout=args.request_timeout_seconds,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
                attn_implementation=args.attn_implementation,
                cache_dir=args.hf_cache_dir,
                enable_thinking=args.enable_thinking,
            )
            reverse_coverage_summary = reverse_coverage_payload["summary"]
            entity_coverage_submission_to_gt = round_float(reverse_coverage_summary.get("entity_coverage", 0.0))
            claim_coverage_submission_to_gt = round_float(
                reverse_coverage_summary.get("claim_coverage", reverse_coverage_summary.get("fact_coverage", 0.0))
            )
            fact_score_submission_to_gt = round_float(
                _compute_fact_score(reverse_coverage_summary, args.fact_score_mode)
            )

            entity_coverage = _compute_mean_if_all_present(
                entity_coverage_gt_to_submission,
                entity_coverage_submission_to_gt,
            )
            claim_coverage = _compute_mean_if_all_present(
                claim_coverage_gt_to_submission,
                claim_coverage_submission_to_gt,
            )
            fact_score = _compute_mean_if_all_present(
                fact_score_gt_to_submission,
                fact_score_submission_to_gt,
            )

            result.update(
                {
                    "status": "queued_for_bertscore",
                    "label": reference_row.get("label", submission_row.get("label")),
                    "submission_complex_key": candidate_key,
                    "reference_complex_key": reference_key,
                    "submission_complex_explanation": candidate_text,
                    "reference_complex_explanation": reference_text,
                    "reference_evidence": {
                        "diagnostic_entities": get_reference_entities(reference_payload),
                        "evidence_claims": get_reference_claims(reference_payload),
                    },
                    "submission_evidence": {
                        "diagnostic_entities": get_reference_entities(submission_payload),
                        "evidence_claims": get_reference_claims(submission_payload),
                    },
                    "coverage_gt_to_submission": {
                        "entity_matches": coverage_payload["entity_matches"],
                        "claim_matches": coverage_payload["claim_matches"],
                        "summary": coverage_summary,
                    },
                    "coverage_submission_to_gt": {
                        "entity_matches": reverse_coverage_payload["entity_matches"],
                        "claim_matches": reverse_coverage_payload["claim_matches"],
                        "summary": reverse_coverage_summary,
                    },
                    "entity_coverage_gt_to_submission": entity_coverage_gt_to_submission,
                    "entity_coverage_submission_to_gt": entity_coverage_submission_to_gt,
                    "entity_coverage": entity_coverage,
                    "claim_coverage_gt_to_submission": claim_coverage_gt_to_submission,
                    "claim_coverage_submission_to_gt": claim_coverage_submission_to_gt,
                    "claim_coverage": claim_coverage,
                    "fact_score_gt_to_submission": fact_score_gt_to_submission,
                    "fact_score_submission_to_gt": fact_score_submission_to_gt,
                    "fact_score": round_float(fact_score),
                }
            )
            pending_predictions.append(candidate_text)
            pending_references.append(reference_text)
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "reason": str(exc),
                    "bertscore_precision": None,
                    "bertscore_recall": None,
                    "bertscore_f1": None,
                    "entity_coverage": None,
                    "claim_coverage": None,
                    "fact_score": None,
                }
            )
        per_sample.append(result)

    bertscore_ready_indices = [index for index, item in enumerate(per_sample) if item["status"] == "queued_for_bertscore"]
    if bertscore_ready_indices:
        print(
            "Computing complex BERTScore for {0} samples...".format(
                len(bertscore_ready_indices)
            )
        )
        bertscore_rows = _compute_bertscore_batch(
            [per_sample[index]["submission_complex_explanation"] for index in bertscore_ready_indices],
            [per_sample[index]["reference_complex_explanation"] for index in bertscore_ready_indices],
            model_type=args.bertscore_model_type,
            lang=args.bertscore_lang,
            rescale_with_baseline=args.bertscore_rescale_with_baseline,
            batch_size=args.bertscore_batch_size,
        )
        for index, bertscore_payload in zip(bertscore_ready_indices, bertscore_rows):
            per_sample[index].update(bertscore_payload)
            per_sample[index]["status"] = "scored"

    scored_samples = [item for item in per_sample if item.get("status") == "scored"]
    entity_present_total_gt_to_submission = sum(
        item["coverage_gt_to_submission"]["summary"].get("entity_present", 0) for item in scored_samples
    )
    entity_total_gt_to_submission = sum(
        item["coverage_gt_to_submission"]["summary"].get("entity_total", 0) for item in scored_samples
    )
    claim_present_total_gt_to_submission = sum(
        item["coverage_gt_to_submission"]["summary"].get(
            "claim_present",
            item["coverage_gt_to_submission"]["summary"].get("fact_present", 0),
        )
        for item in scored_samples
    )
    claim_total_gt_to_submission = sum(
        item["coverage_gt_to_submission"]["summary"].get(
            "claim_total",
            item["coverage_gt_to_submission"]["summary"].get("fact_total", 0),
        )
        for item in scored_samples
    )
    entity_present_total_submission_to_gt = sum(
        item["coverage_submission_to_gt"]["summary"].get("entity_present", 0) for item in scored_samples
    )
    entity_total_submission_to_gt = sum(
        item["coverage_submission_to_gt"]["summary"].get("entity_total", 0) for item in scored_samples
    )
    claim_present_total_submission_to_gt = sum(
        item["coverage_submission_to_gt"]["summary"].get(
            "claim_present",
            item["coverage_submission_to_gt"]["summary"].get("fact_present", 0),
        )
        for item in scored_samples
    )
    claim_total_submission_to_gt = sum(
        item["coverage_submission_to_gt"]["summary"].get(
            "claim_total",
            item["coverage_submission_to_gt"]["summary"].get("fact_total", 0),
        )
        for item in scored_samples
    )
    fact_scores = [item["fact_score"] for item in scored_samples if item.get("fact_score") is not None]

    summary = {
        "sample_count": len(per_sample),
        "scored_samples": len(scored_samples),
        "error_samples": sum(1 for item in per_sample if item.get("status") == "error"),
        **summarize_bertscore(scored_samples),
        "entity_coverage_macro_gt_to_submission": round_float(
            sum(item["entity_coverage_gt_to_submission"] for item in scored_samples if item.get("entity_coverage_gt_to_submission") is not None) / len(scored_samples)
        ) if scored_samples else None,
        "entity_coverage_macro_submission_to_gt": round_float(
            sum(item["entity_coverage_submission_to_gt"] for item in scored_samples if item.get("entity_coverage_submission_to_gt") is not None) / len(scored_samples)
        ) if scored_samples else None,
        "entity_coverage_macro": round_float(
            sum(item["entity_coverage"] for item in scored_samples if item.get("entity_coverage") is not None) / len(scored_samples)
        ) if scored_samples else None,
        "claim_coverage_macro_gt_to_submission": round_float(
            sum(item["claim_coverage_gt_to_submission"] for item in scored_samples if item.get("claim_coverage_gt_to_submission") is not None) / len(scored_samples)
        ) if scored_samples else None,
        "claim_coverage_macro_submission_to_gt": round_float(
            sum(item["claim_coverage_submission_to_gt"] for item in scored_samples if item.get("claim_coverage_submission_to_gt") is not None) / len(scored_samples)
        ) if scored_samples else None,
        "claim_coverage_macro": round_float(
            sum(item["claim_coverage"] for item in scored_samples if item.get("claim_coverage") is not None) / len(scored_samples)
        ) if scored_samples else None,
        "entity_coverage_micro_gt_to_submission": round_float(entity_present_total_gt_to_submission / entity_total_gt_to_submission) if entity_total_gt_to_submission else 0.0,
        "entity_coverage_micro_submission_to_gt": round_float(entity_present_total_submission_to_gt / entity_total_submission_to_gt) if entity_total_submission_to_gt else 0.0,
        "claim_coverage_micro_gt_to_submission": round_float(claim_present_total_gt_to_submission / claim_total_gt_to_submission) if claim_total_gt_to_submission else 0.0,
        "claim_coverage_micro_submission_to_gt": round_float(claim_present_total_submission_to_gt / claim_total_submission_to_gt) if claim_total_submission_to_gt else 0.0,
        "fact_score_mean": round_float(sum(float(score) for score in fact_scores) / len(fact_scores)) if fact_scores else None,
    }

    report = build_base_report(
        metric_name="complex_explanations",
        submission_path=Path(args.submission),
        reference_path=Path(args.ground_truth),
        summary=summary,
        per_sample=per_sample,
        config={
            "submission_complex_keys": list(args.submission_complex_keys),
            "reference_complex_keys": list(args.reference_complex_keys),
            "submission_id_keys": list(args.submission_id_keys),
            "reference_id_keys": list(args.reference_id_keys),
            "bertscore_model_type": args.bertscore_model_type,
            "bertscore_lang": args.bertscore_lang,
            "bertscore_rescale_with_baseline": args.bertscore_rescale_with_baseline,
            "fact_score_mode": args.fact_score_mode,
            "backend": args.backend,
            "model_name": args.model_name,
            "entity_fact_prompt": str(args.entity_fact_prompt),
            "semantic_coverage_prompt": str(args.semantic_coverage_prompt),
        },
        diagnostics=diagnostics,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate complex explanations from a submission JSONL against reference JSONL.")
    parser.add_argument("--submission", required=True, help="Path to participant submission JSONL.")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH_PATH, help="Path to validation ground-truth JSONL.")
    parser.add_argument("--output", required=True, help="Path to write the complex evaluation JSON report.")
    parser.add_argument("--submission-id-keys", nargs="+", default=list(DEFAULT_ID_KEYS))
    parser.add_argument("--reference-id-keys", nargs="+", default=list(DEFAULT_ID_KEYS))
    parser.add_argument("--submission-complex-keys", nargs="+", default=list(DEFAULT_COMPLEX_KEYS))
    parser.add_argument("--reference-complex-keys", nargs="+", default=list(DEFAULT_COMPLEX_KEYS))
    parser.add_argument("--reference-cache-dir", default=None, help="Optional folder for cached extracted reference entities/claims.")
    parser.add_argument("--entity-fact-prompt", type=Path, default=Path(__file__).resolve().parent / "prompts" / "entity_fact_extraction_prompt.txt")
    parser.add_argument("--semantic-coverage-prompt", type=Path, default=Path(__file__).resolve().parent / "prompts" / "semantic_coverage_prompt.txt")
    parser.add_argument("--bertscore-model-type", default=DEFAULT_BERT_MODEL_TYPE)
    parser.add_argument("--bertscore-lang", default=DEFAULT_BERT_LANG)
    parser.add_argument("--bertscore-rescale-with-baseline", action="store_true", default=DEFAULT_BERT_RESCALE_WITH_BASELINE)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--fact-score-mode", choices=["mean_entity_and_claim", "claim_only", "entity_only"], default=DEFAULT_FACT_SCORE_MODE)
    parser.add_argument("--backend", choices=["transformers", "openai_compatible"], default=DEFAULT_INFERENCE_BACKEND)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extraction-max-tokens", type=int, default=DEFAULT_EXTRACTION_MAX_TOKENS)
    parser.add_argument("--coverage-max-tokens", type=int, default=DEFAULT_COVERAGE_MAX_TOKENS)
    parser.add_argument("--request-timeout-seconds", type=int, default=300)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", type=lambda x: str(x).lower() in {'1','true','yes','y'}, default=True)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--no-preload-models", action="store_true", default=False)
    parser.add_argument("--no-progress", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.preload_models = not args.no_preload_models
    args.show_progress = not args.no_progress
    report = evaluate_complex_submission(args)
    write_json(args.output, report)
    print(f"Wrote complex evaluation report to: {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
