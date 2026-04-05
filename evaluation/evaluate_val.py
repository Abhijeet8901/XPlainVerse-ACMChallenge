from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from utils.challenge_eval_utils import (
    align_submission_and_reference,
    extract_required_text,
    read_jsonl,
    round_float,
    write_json,
)
from utils.llm_helpers import (
    build_progress_bar,
    chat_completion_batch,
    clear_chat_model_cache,
    compute_coverage_summary,
    extract_first_json,
    get_bert_scorer,
    get_coverage_claim_matches,
    get_coverage_entity_matches,
    get_reference_claims,
    get_reference_entities,
    get_sle_components,
    load_text,
    preload_bertscorer,
    preload_chat_model,
    preload_sle_model,
)


DEFAULT_SYSTEM_PROMPT_EXTRACTION = "You are a careful information extraction assistant. Return JSON only."
DEFAULT_SYSTEM_PROMPT_COVERAGE = "You are a careful semantic coverage assistant. Return JSON only."
DEFAULT_QWEN_BATCH_SIZE = 4
DEFAULT_BERT_BATCH_SIZE = 8
DEFAULT_SLE_BATCH_SIZE = 16
DEFAULT_GROUND_TRUTH_PATH = Path(__file__).resolve().parent / "data" / "val_ground_truth.jsonl"


def _write_progress_message(iterator, message):
    if hasattr(iterator, "write"):
        iterator.write(message)
        return
    print(message)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compute_mean(values):
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return round_float(sum(numeric_values) / len(numeric_values))


def _compute_harmonic_mean_if_all_present(*values):
    if any(value is None for value in values):
        return None
    numeric_values = [float(value) for value in values]
    denominator = sum(numeric_values)
    if denominator == 0.0:
        return 0.0
    if len(numeric_values) != 2:
        raise ValueError("Harmonic mean helper expects exactly two values.")
    return round_float((2.0 * numeric_values[0] * numeric_values[1]) / denominator)


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _compute_complex_overall_score(complex_bert_f1, complex_entity_f1, complex_facts_f1):
    if any(value is None for value in (complex_bert_f1, complex_entity_f1, complex_facts_f1)):
        return None
    return round_float(
        0.3 * float(complex_bert_f1)
        + 0.4 * float(complex_entity_f1)
        + 0.3 * float(complex_facts_f1)
    )


def _normalize_simple_sle(simple_sle_score):
    if simple_sle_score is None:
        return None
    clipped = _clip(float(simple_sle_score), -1.0, 4.0)
    return round_float((clipped + 1.0) / 5.0)


def _compute_simple_overall_score(simple_bert_f1, simple_sle_score):
    simple_sle_norm = _normalize_simple_sle(simple_sle_score)
    if simple_bert_f1 is None or simple_sle_norm is None:
        return None
    return round_float(0.7 * float(simple_bert_f1) + 0.3 * float(simple_sle_norm))


def _build_final_scores(rows):
    return {
        "samples_completed": len(rows),
        "complex_bert_f1": _compute_mean(item.get("complex_bert_f1") for item in rows),
        "complex_entity_f1": _compute_mean(item.get("complex_entity_f1") for item in rows),
        "complex_facts_f1": _compute_mean(item.get("complex_facts_f1") for item in rows),
        "complex_overall_score": _compute_mean(item.get("complex_overall_score") for item in rows),
        "simple_bert_f1": _compute_mean(item.get("simple_bert_f1") for item in rows),
        "simple_sle_score": _compute_mean(item.get("simple_sle_score") for item in rows),
        "simple_overall_score": _compute_mean(item.get("simple_overall_score") for item in rows),
    }


def _parse_reference_payload(sample_id: str, explanation_text: str, raw_response: str) -> Dict[str, Any]:
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
        raise ValueError(
            "Reference extraction returned an invalid diagnostic_entities list for sample '{0}'.".format(sample_id)
        )
    if not isinstance(evidence_claims, list):
        raise ValueError(
            "Reference extraction returned an invalid evidence_claims list for sample '{0}'.".format(sample_id)
        )
    return {
        "sample_id": sample_id,
        "explanation": explanation_text,
        "diagnostic_entities": diagnostic_entities,
        "evidence_claims": evidence_claims,
    }


def _parse_coverage_summary(sample_id: str, raw_response: str) -> Dict[str, Any]:
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
        raise ValueError(
            "Coverage model returned an invalid entity_matches list for sample '{0}'.".format(sample_id)
        )
    if not isinstance(claim_matches, list):
        raise ValueError(
            "Coverage model returned an invalid claim_matches list for sample '{0}'.".format(sample_id)
        )
    return compute_coverage_summary(entity_matches, claim_matches)


def _compute_bertscore_f1_scores(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str,
    lang: str,
    rescale_with_baseline: bool,
    batch_size: int,
    show_progress: bool,
    desc: str,
) -> List[float]:
    scorer = get_bert_scorer(
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
    )
    scores: List[float] = []
    batch_starts = range(0, len(predictions), batch_size)
    batch_iterator = build_progress_bar(
        batch_starts,
        total=(len(predictions) + batch_size - 1) // batch_size,
        desc=desc,
        disable=not show_progress,
    )
    for start in batch_iterator:
        batch_predictions = list(predictions[start : start + batch_size])
        batch_references = list(references[start : start + batch_size])
        _, _, f1 = scorer.score(
            batch_predictions,
            batch_references,
            batch_size=max(1, min(batch_size, len(batch_predictions))),
            verbose=False,
        )
        scores.extend(round_float(value) for value in f1.tolist())
    return scores


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
        desc="Simple SLE",
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


def _prepare_rows(aligned_rows, args, show_progress):
    rows: List[Dict[str, Any]] = []
    sample_iterator = build_progress_bar(
        aligned_rows,
        total=len(aligned_rows),
        desc="Prepare samples",
        disable=not show_progress,
    )
    for sample_id, submission_row, reference_row in sample_iterator:
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "complex_bert_f1": None,
            "complex_entity_f1": None,
            "complex_facts_f1": None,
            "complex_overall_score": None,
            "simple_bert_f1": None,
            "simple_sle_score": None,
            "simple_overall_score": None,
            "_submission_complex_text": None,
            "_reference_complex_text": None,
            "_submission_simple_text": None,
            "_reference_simple_text": None,
            "_gt_extraction": None,
            "_pred_extraction": None,
            "_gt_to_pred_entity": None,
            "_gt_to_pred_fact": None,
            "_pred_to_gt_entity": None,
            "_pred_to_gt_fact": None,
        }

        try:
            row["_submission_complex_text"], _ = extract_required_text(
                submission_row,
                args.submission_complex_keys,
                field_role="submission complex explanation",
                sample_id=sample_id,
            )
            row["_reference_complex_text"], _ = extract_required_text(
                reference_row,
                args.reference_complex_keys,
                field_role="reference complex explanation",
                sample_id=sample_id,
            )
        except Exception as exc:
            _write_progress_message(
                sample_iterator,
                "Warning: complex text preparation failed for {0}: {1}".format(sample_id, exc),
            )

        try:
            row["_submission_simple_text"], _ = extract_required_text(
                submission_row,
                args.submission_simple_keys,
                field_role="submission simple explanation",
                sample_id=sample_id,
            )
            row["_reference_simple_text"], _ = extract_required_text(
                reference_row,
                args.reference_simple_keys,
                field_role="reference simple explanation",
                sample_id=sample_id,
            )
        except Exception as exc:
            _write_progress_message(
                sample_iterator,
                "Warning: simple text preparation failed for {0}: {1}".format(sample_id, exc),
            )

        rows.append(row)
    return rows


def _run_extraction_stage(
    rows,
    *,
    text_key,
    output_key,
    prompt_template,
    args,
    desc,
    show_progress,
):
    active_indices = [index for index, row in enumerate(rows) if row.get(text_key)]
    if not active_indices:
        return

    batch_starts = range(0, len(active_indices), args.qwen_batch_size)
    batch_iterator = build_progress_bar(
        batch_starts,
        total=(len(active_indices) + args.qwen_batch_size - 1) // args.qwen_batch_size,
        desc=desc,
        disable=not show_progress,
    )
    for start in batch_iterator:
        batch_indices = active_indices[start : start + args.qwen_batch_size]
        user_prompts = [
            prompt_template.replace("{{EXPLANATION_TEXT}}", rows[index][text_key])
            for index in batch_indices
        ]
        try:
            raw_responses = chat_completion_batch(
                backend=args.backend,
                model=args.model_name,
                system_prompt=DEFAULT_SYSTEM_PROMPT_EXTRACTION,
                user_prompts=user_prompts,
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
        except Exception as exc:
            _write_progress_message(
                batch_iterator,
                "Warning: {0} batch failed at {1}: {2}".format(
                    desc,
                    rows[batch_indices[0]]["sample_id"],
                    exc,
                ),
            )
            continue

        for index, raw_response in zip(batch_indices, raw_responses):
            try:
                rows[index][output_key] = _parse_reference_payload(
                    rows[index]["sample_id"],
                    rows[index][text_key],
                    raw_response,
                )
            except Exception as exc:
                _write_progress_message(
                    batch_iterator,
                    "Warning: {0} failed for {1}: {2}".format(
                        desc,
                        rows[index]["sample_id"],
                        exc,
                    ),
                )


def _run_coverage_stage(
    rows,
    *,
    reference_payload_key,
    candidate_text_key,
    entity_output_key,
    fact_output_key,
    prompt_template,
    args,
    desc,
    show_progress,
):
    active_indices = [
        index
        for index, row in enumerate(rows)
        if row.get(reference_payload_key) is not None and row.get(candidate_text_key)
    ]
    if not active_indices:
        return

    batch_starts = range(0, len(active_indices), args.qwen_batch_size)
    batch_iterator = build_progress_bar(
        batch_starts,
        total=(len(active_indices) + args.qwen_batch_size - 1) // args.qwen_batch_size,
        desc=desc,
        disable=not show_progress,
    )
    for start in batch_iterator:
        batch_indices = active_indices[start : start + args.qwen_batch_size]
        user_prompts = []
        for index in batch_indices:
            reference_payload = rows[index][reference_payload_key]
            reference_json_for_prompt = {
                "diagnostic_entities": get_reference_entities(reference_payload),
                "evidence_claims": get_reference_claims(reference_payload),
            }
            user_prompts.append(
                prompt_template.replace(
                    "{{REFERENCE_JSON}}",
                    json.dumps(reference_json_for_prompt, indent=2, ensure_ascii=True),
                ).replace("{{CANDIDATE_EXPLANATION}}", rows[index][candidate_text_key])
            )

        try:
            raw_responses = chat_completion_batch(
                backend=args.backend,
                model=args.model_name,
                system_prompt=DEFAULT_SYSTEM_PROMPT_COVERAGE,
                user_prompts=user_prompts,
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
        except Exception as exc:
            _write_progress_message(
                batch_iterator,
                "Warning: {0} batch failed at {1}: {2}".format(
                    desc,
                    rows[batch_indices[0]]["sample_id"],
                    exc,
                ),
            )
            continue

        for index, raw_response in zip(batch_indices, raw_responses):
            try:
                summary = _parse_coverage_summary(rows[index]["sample_id"], raw_response)
                rows[index][entity_output_key] = round_float(summary.get("entity_coverage", 0.0))
                rows[index][fact_output_key] = round_float(
                    summary.get("claim_coverage", summary.get("fact_coverage", 0.0))
                )
            except Exception as exc:
                _write_progress_message(
                    batch_iterator,
                    "Warning: {0} failed for {1}: {2}".format(
                        desc,
                        rows[index]["sample_id"],
                        exc,
                    ),
                )


def _run_bertscore_stage(
    rows,
    *,
    prediction_text_key,
    reference_text_key,
    output_key,
    model_type,
    lang,
    rescale_with_baseline,
    batch_size,
    show_progress,
    desc,
):
    active_rows = [
        row for row in rows if row.get(prediction_text_key) and row.get(reference_text_key)
    ]
    if not active_rows:
        return

    scores = _compute_bertscore_f1_scores(
        [row[prediction_text_key] for row in active_rows],
        [row[reference_text_key] for row in active_rows],
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        batch_size=batch_size,
        show_progress=show_progress,
        desc=desc,
    )
    for row, score in zip(active_rows, scores):
        row[output_key] = score


def _run_simple_sle_stage(rows, *, args, show_progress):
    active_rows = [row for row in rows if row.get("_submission_simple_text")]
    if not active_rows:
        return

    scores = _compute_sle_scores(
        [row["_submission_simple_text"] for row in active_rows],
        model_id=args.sle_model_id,
        batch_size=args.sle_batch_size,
        max_length=args.sle_max_length,
        local_files_only=args.sle_local_files_only,
        show_progress=show_progress,
    )
    for row, score in zip(active_rows, scores):
        row["simple_sle_score"] = score


def _finalize_rows(rows):
    finalized_rows = []
    for row in rows:
        row["complex_entity_f1"] = _compute_harmonic_mean_if_all_present(
            row.get("_pred_to_gt_entity"),
            row.get("_gt_to_pred_entity"),
        )
        row["complex_facts_f1"] = _compute_harmonic_mean_if_all_present(
            row.get("_pred_to_gt_fact"),
            row.get("_gt_to_pred_fact"),
        )
        row["complex_overall_score"] = _compute_complex_overall_score(
            row.get("complex_bert_f1"),
            row.get("complex_entity_f1"),
            row.get("complex_facts_f1"),
        )
        row["simple_overall_score"] = _compute_simple_overall_score(
            row.get("simple_bert_f1"),
            row.get("simple_sle_score"),
        )
        finalized_rows.append(
            {
                "sample_id": row["sample_id"],
                "complex_bert_f1": row.get("complex_bert_f1"),
                "complex_entity_f1": row.get("complex_entity_f1"),
                "complex_facts_f1": row.get("complex_facts_f1"),
                "complex_overall_score": row.get("complex_overall_score"),
                "simple_bert_f1": row.get("simple_bert_f1"),
                "simple_sle_score": row.get("simple_sle_score"),
                "simple_overall_score": row.get("simple_overall_score"),
            }
        )
    return finalized_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation in stage-wise batches on one GPU."
    )
    parser.add_argument("--submission", required=True)
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--submission-id-keys", nargs="+", default=["sample_id"])
    parser.add_argument("--reference-id-keys", nargs="+", default=["sample_id"])
    parser.add_argument("--submission-complex-keys", nargs="+", default=["complex_explanation"])
    parser.add_argument("--reference-complex-keys", nargs="+", default=["complex_explanation"])
    parser.add_argument("--submission-simple-keys", nargs="+", default=["simple_explanation"])
    parser.add_argument("--reference-simple-keys", nargs="+", default=["simple_explanation"])

    parser.add_argument("--entity-fact-prompt", type=Path, default=Path(__file__).resolve().parent / "prompts" / "entity_fact_extraction_prompt.txt")
    parser.add_argument("--semantic-coverage-prompt", type=Path, default=Path(__file__).resolve().parent / "prompts" / "semantic_coverage_prompt.txt")

    parser.add_argument("--bertscore-model-type", default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--bertscore-rescale-with-baseline", action="store_true", default=False)
    parser.add_argument("--bertscore-batch-size", type=int, default=DEFAULT_BERT_BATCH_SIZE)

    parser.add_argument("--backend", choices=["transformers", "openai_compatible"], default="transformers")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--qwen-batch-size", type=int, default=DEFAULT_QWEN_BATCH_SIZE)
    parser.add_argument("--extraction-max-tokens", type=int, default=1024)
    parser.add_argument("--coverage-max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout-seconds", type=int, default=300)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--trust-remote-code", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--skip-qwen", action="store_true", default=False)
    parser.add_argument("--no-preload-models", action="store_true", default=False)
    parser.add_argument("--no-progress", action="store_true", default=False)

    parser.add_argument("--sle-model-id", default="liamcripwell/sle-base")
    parser.add_argument("--sle-batch-size", type=int, default=DEFAULT_SLE_BATCH_SIZE)
    parser.add_argument("--sle-max-length", type=int, default=512)
    parser.add_argument("--sle-local-files-only", action="store_true", default=False)

    args = parser.parse_args()
    args.preload_models = not args.no_preload_models
    args.show_progress = not args.no_progress

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_output_path = output_dir / "per_sample_scores.jsonl"
    final_scores_output_path = output_dir / "final_scores.json"

    submission_rows = read_jsonl(args.submission)
    reference_rows = read_jsonl(args.ground_truth)
    aligned_rows, diagnostics = align_submission_and_reference(
        submission_rows,
        reference_rows,
        submission_id_keys=args.submission_id_keys,
        reference_id_keys=args.reference_id_keys,
    )

    if diagnostics:
        print("Alignment diagnostics: {0}".format(len(diagnostics)))

    rows = _prepare_rows(aligned_rows, args, args.show_progress)
    if args.skip_qwen:
        print("Skipping Qwen extraction and coverage. Qwen-based complex metrics will be written as null.")
    else:
        extraction_prompt = load_text(args.entity_fact_prompt)
        coverage_prompt = load_text(args.semantic_coverage_prompt)

        if args.preload_models and args.backend == "transformers":
            print("Preloading Qwen model...")
            preload_chat_model(
                backend=args.backend,
                model=args.model_name,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                trust_remote_code=args.trust_remote_code,
                attn_implementation=args.attn_implementation,
                cache_dir=args.hf_cache_dir,
            )

        _run_extraction_stage(
            rows,
            text_key="_reference_complex_text",
            output_key="_gt_extraction",
            prompt_template=extraction_prompt,
            args=args,
            desc="Qwen extract ground truth",
            show_progress=args.show_progress,
        )
        _run_extraction_stage(
            rows,
            text_key="_submission_complex_text",
            output_key="_pred_extraction",
            prompt_template=extraction_prompt,
            args=args,
            desc="Qwen extract prediction",
            show_progress=args.show_progress,
        )
        _run_coverage_stage(
            rows,
            reference_payload_key="_gt_extraction",
            candidate_text_key="_submission_complex_text",
            entity_output_key="_gt_to_pred_entity",
            fact_output_key="_gt_to_pred_fact",
            prompt_template=coverage_prompt,
            args=args,
            desc="Qwen coverage gt->pred",
            show_progress=args.show_progress,
        )
        _run_coverage_stage(
            rows,
            reference_payload_key="_pred_extraction",
            candidate_text_key="_reference_complex_text",
            entity_output_key="_pred_to_gt_entity",
            fact_output_key="_pred_to_gt_fact",
            prompt_template=coverage_prompt,
            args=args,
            desc="Qwen coverage pred->gt",
            show_progress=args.show_progress,
        )

        clear_chat_model_cache()

    if args.preload_models:
        print("Preloading BERTScore and SLE models...")
        preload_bertscorer(
            model_type=args.bertscore_model_type,
            lang=args.bertscore_lang,
            rescale_with_baseline=args.bertscore_rescale_with_baseline,
        )
        preload_sle_model(
            model_id=args.sle_model_id,
            local_files_only=args.sle_local_files_only,
        )

    _run_bertscore_stage(
        rows,
        prediction_text_key="_submission_complex_text",
        reference_text_key="_reference_complex_text",
        output_key="complex_bert_f1",
        model_type=args.bertscore_model_type,
        lang=args.bertscore_lang,
        rescale_with_baseline=args.bertscore_rescale_with_baseline,
        batch_size=args.bertscore_batch_size,
        show_progress=args.show_progress,
        desc="Complex BERTScore",
    )
    _run_bertscore_stage(
        rows,
        prediction_text_key="_submission_simple_text",
        reference_text_key="_reference_simple_text",
        output_key="simple_bert_f1",
        model_type=args.bertscore_model_type,
        lang=args.bertscore_lang,
        rescale_with_baseline=args.bertscore_rescale_with_baseline,
        batch_size=args.bertscore_batch_size,
        show_progress=args.show_progress,
        desc="Simple BERTScore",
    )
    _run_simple_sle_stage(
        rows,
        args=args,
        show_progress=args.show_progress,
    )

    finalized_rows = _finalize_rows(rows)
    _write_jsonl(per_sample_output_path, finalized_rows)
    write_json(final_scores_output_path, _build_final_scores(finalized_rows))

    print("Wrote per-sample scores to: {0}".format(per_sample_output_path))
    print("Wrote final scores to: {0}".format(final_scores_output_path))


if __name__ == "__main__":
    main()
