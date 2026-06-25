from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from utils.challenge_eval_utils import (
    align_submission_and_reference,
    compute_hash,
    extract_required_text,
    get_first_present,
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
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH_ROOT = EVALUATION_DIR / "ground_truth"
DEFAULT_GROUND_TRUTH_SPLIT = "val"
DEFAULT_GROUND_TRUTH_PATH = DEFAULT_GROUND_TRUTH_ROOT / DEFAULT_GROUND_TRUTH_SPLIT / "reference.jsonl"
REFERENCE_FILE_CANDIDATES = (
    "reference.jsonl",
    "final_reference.jsonl",
    "val_ground_truth.jsonl",
    "test_reference.jsonl",
)
GT_ENTITY_FACT_FILE_CANDIDATES = (
    "complex_ground_truth_entity_facts.jsonl",
    "test_complex_ground_truth_entity_facts.jsonl",
    "val_complex_ground_truth_entity_facts.jsonl",
)
DEFAULT_LABEL_KEYS = ("label", "pred_label", "predicted_label")
DEFAULT_ID_KEYS = ("sample_id", "id")
CHALLENGE_DETECTION_FILE = "detection.jsonl"
CHALLENGE_COMPLEX_FILE = "complex.jsonl"
CHALLENGE_SIMPLE_FILE = "simple.jsonl"
CHALLENGE_SUBMISSION_FILES = (
    CHALLENGE_DETECTION_FILE,
    CHALLENGE_COMPLEX_FILE,
    CHALLENGE_SIMPLE_FILE,
)
CHALLENGE_ID_KEYS = ("id", "sample_id")


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


def _compute_complex_overall_score(complex_bert_f1, complex_entity_f1, complex_evidence_f1):
    if any(value is None for value in (complex_bert_f1, complex_entity_f1, complex_evidence_f1)):
        return None
    return round_float(
        0.3 * float(complex_bert_f1)
        + 0.4 * float(complex_entity_f1)
        + 0.3 * float(complex_evidence_f1)
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


def _compute_mean_if_all_present(*values):
    if any(value is None for value in values):
        return None
    return round_float(sum(float(value) for value in values) / len(values))


def _compute_reference_explanation_score(complex_bert_f1, simple_overall_score):
    if complex_bert_f1 is None or simple_overall_score is None:
        return None
    return round_float((float(complex_bert_f1) + float(simple_overall_score)) / 2.0)


def _compute_paper_explanation_score(reference_explanation_score, grounding_score):
    if reference_explanation_score is None or grounding_score is None:
        return None
    return round_float(
        0.4 * float(reference_explanation_score)
        + 0.6 * float(grounding_score)
    )


def _compute_overall_score(detection_f1, explanation_score):
    if detection_f1 is None or explanation_score is None:
        return None
    return round_float((float(detection_f1) + float(explanation_score)) / 2.0)


def _normalize_label(value):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"0", "real"}:
        return "real"
    if normalized in {"1", "fake"}:
        return "fake"
    return None


def _zero_filled_mean(values, denominator):
    if denominator <= 0:
        return None
    total = 0.0
    for value in values:
        if value is None:
            continue
        total += float(value)
    return round_float(total / denominator)


def _bool_mean(values, denominator):
    if denominator <= 0:
        return None
    return round_float(sum(1.0 for value in values if value is True) / denominator)


def _binary_f1(rows, *, positive_label):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    seen_label = False
    for row in rows:
        predicted_label = row.get("predicted_label")
        ground_truth_label = row.get("ground_truth_label")
        if ground_truth_label is None:
            continue
        seen_label = True
        if predicted_label == positive_label and ground_truth_label == positive_label:
            true_positive += 1
        elif predicted_label == positive_label and ground_truth_label != positive_label:
            false_positive += 1
        elif ground_truth_label == positive_label:
            false_negative += 1
    if not seen_label:
        return None
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return round_float((2 * true_positive) / denominator)


def _compute_detection_metrics(rows):
    labeled_rows = [row for row in rows if row.get("ground_truth_label") is not None]
    if not labeled_rows:
        return {
            "detection_macro_f1": None,
            "detection_fake_f1": None,
            "detection_real_f1": None,
            "detection_accuracy": None,
        }
    fake_f1 = _binary_f1(labeled_rows, positive_label="fake")
    real_f1 = _binary_f1(labeled_rows, positive_label="real")
    if fake_f1 is None or real_f1 is None:
        macro_f1 = None
    else:
        macro_f1 = round_float((float(fake_f1) + float(real_f1)) / 2.0)
    return {
        "detection_macro_f1": macro_f1,
        "detection_fake_f1": fake_f1,
        "detection_real_f1": real_f1,
        "detection_accuracy": _bool_mean(
            (item.get("label_correct") for item in labeled_rows),
            len(labeled_rows),
        ),
    }


def _build_final_scores(rows):
    denominator = len(rows)
    complex_bert_f1 = _zero_filled_mean((item.get("complex_bert_f1") for item in rows), denominator)
    simple_sle_score = _compute_mean(item.get("simple_sle_score") for item in rows)
    simple_sle_normalized = _zero_filled_mean((item.get("simple_sle_normalized") for item in rows), denominator)
    simple_bert_f1 = _zero_filled_mean((item.get("simple_bert_f1") for item in rows), denominator)
    if simple_bert_f1 is None or simple_sle_normalized is None:
        simple_overall_score = None
    else:
        simple_overall_score = round_float(0.7 * float(simple_bert_f1) + 0.3 * float(simple_sle_normalized))
    reference_explanation_score = _compute_reference_explanation_score(
        complex_bert_f1,
        simple_overall_score,
    )
    complex_entity_f1 = _zero_filled_mean((item.get("complex_entity_f1") for item in rows), denominator)
    complex_evidence_f1 = _zero_filled_mean((item.get("complex_evidence_f1") for item in rows), denominator)
    grounding_score = _compute_mean_if_all_present(complex_entity_f1, complex_evidence_f1)
    explanation_score = _compute_paper_explanation_score(
        reference_explanation_score,
        grounding_score,
    )
    detection_metrics = _compute_detection_metrics(rows)
    detection_f1 = detection_metrics["detection_macro_f1"]
    overall_score = _compute_overall_score(detection_f1, explanation_score)
    submitted_rows = [
        row
        for row in rows
        if (
            row.get("predicted_label") is not None
            or row.get("complex_bert_f1") is not None
            or row.get("simple_bert_f1") is not None
        )
    ]
    return {
        "metric_version": "acm_mm_2026_paper",
        "samples_expected": denominator,
        "samples_completed": len(rows),
        "submission_rows_with_any_scored_field": len(submitted_rows),
        "detection_macro_f1": detection_metrics["detection_macro_f1"],
        "detection_f1": detection_f1,
        "detection_fake_f1": detection_metrics["detection_fake_f1"],
        "detection_real_f1": detection_metrics["detection_real_f1"],
        "detection_accuracy": detection_metrics["detection_accuracy"],
        "accuracy": detection_metrics["detection_accuracy"],
        "complex_bert_f1": complex_bert_f1,
        "complex_explanation_score": complex_bert_f1,
        "complex_entity_f1": complex_entity_f1,
        "entity_score": complex_entity_f1,
        "complex_evidence_f1": complex_evidence_f1,
        "evidence_score": complex_evidence_f1,
        "complex_overall_score": _zero_filled_mean((item.get("complex_overall_score") for item in rows), denominator),
        "simple_bert_f1": simple_bert_f1,
        "simple_sle_score": simple_sle_score,
        "simple_sle_normalized": simple_sle_normalized,
        "simple_overall_score": simple_overall_score,
        "simple_explanation_score": simple_overall_score,
        "reference_explanation_score": reference_explanation_score,
        "grounding_score": grounding_score,
        "explanation_score": explanation_score,
        "final_score": overall_score,
        "overall_score": overall_score,
        "score_formula": {
            "detection_macro_f1": "(F1_fake + F1_real) / 2",
            "simple_explanation_score": "0.7 * simple_bert_f1 + 0.3 * simple_sle_normalized",
            "reference_explanation_score": "(complex_explanation_score + simple_explanation_score) / 2",
            "grounding_score": "(entity_score + evidence_score) / 2",
            "explanation_score": "0.4 * reference_explanation_score + 0.6 * grounding_score",
            "final_score": "(detection_macro_f1 + explanation_score) / 2",
        },
    }


def _submission_id_aliases(sample_id: str) -> Tuple[str, ...]:
    normalized = sample_id.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    aliases = [sample_id.strip(), normalized, path.name]
    if path.suffix:
        aliases.append(str(path.with_suffix("")))
        aliases.append(path.stem)
    else:
        aliases.append(path.stem)

    unique: List[str] = []
    for alias in aliases:
        if alias and alias not in unique:
            unique.append(alias)
    return tuple(unique)


def _build_reference_alias_map(
    reference_rows: Sequence[Mapping[str, Any]],
    reference_id_keys: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, Optional[str]], List[str]]:
    reference_ids: Dict[str, str] = {}
    aliases: Dict[str, Optional[str]] = {}
    ordered_ids: List[str] = []
    for row_index, row in enumerate(reference_rows, start=1):
        sample_id_value, _ = get_first_present(dict(row), reference_id_keys)
        if sample_id_value is None:
            raise ValueError(
                "Reference row {0} is missing an id. Tried keys: {1}".format(
                    row_index,
                    ", ".join(reference_id_keys),
                )
            )
        canonical_id = str(sample_id_value).strip()
        if not canonical_id:
            raise ValueError("Reference row {0} has an empty id.".format(row_index))
        if canonical_id in reference_ids:
            raise ValueError("Duplicate reference id: {0}".format(canonical_id))

        reference_ids[canonical_id] = canonical_id
        ordered_ids.append(canonical_id)
        for alias in _submission_id_aliases(canonical_id):
            previous = aliases.get(alias)
            if previous is None and alias in aliases:
                continue
            if previous is not None and previous != canonical_id:
                aliases[alias] = None
            else:
                aliases[alias] = canonical_id
    return reference_ids, aliases, ordered_ids


def _resolve_submission_id(
    sample_id: str,
    *,
    reference_ids: Mapping[str, str],
    reference_aliases: Mapping[str, Optional[str]],
) -> Optional[str]:
    stripped_id = sample_id.strip()
    if stripped_id in reference_ids:
        return stripped_id
    matches = {
        canonical_id
        for alias in _submission_id_aliases(stripped_id)
        for canonical_id in [reference_aliases.get(alias)]
        if canonical_id is not None
    }
    if len(matches) == 1:
        return next(iter(matches))
    return None


def _parse_jsonl_text(text: str, *, source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Malformed JSON in {0}:{1}: {2}".format(source, line_number, exc.msg)
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                "Expected JSON object in {0}:{1}, found {2}.".format(
                    source,
                    line_number,
                    type(payload).__name__,
                )
            )
        rows.append(payload)
    return rows


def _read_challenge_submission_files(submission_path: Path) -> Dict[str, str]:
    if submission_path.is_file():
        if submission_path.suffix.lower() != ".zip":
            raise ValueError("Challenge submission file must be a .zip: {0}".format(submission_path))
        try:
            with zipfile.ZipFile(submission_path) as archive:
                members = [name for name in archive.namelist() if not name.endswith("/")]
                contents: Dict[str, str] = {}
                for expected in CHALLENGE_SUBMISSION_FILES:
                    matches = [name for name in members if PurePosixPath(name).name == expected]
                    if len(matches) > 1:
                        raise ValueError("Submission zip contains multiple {0} files.".format(expected))
                    if matches:
                        contents[expected] = archive.read(matches[0]).decode("utf-8")
        except zipfile.BadZipFile as exc:
            raise ValueError("Malformed submission zip: {0}".format(submission_path)) from exc
        except UnicodeDecodeError as exc:
            raise ValueError("Submission files must be UTF-8 encoded: {0}".format(submission_path)) from exc
    elif submission_path.is_dir():
        contents = {}
        for expected in CHALLENGE_SUBMISSION_FILES:
            matches = sorted(path for path in submission_path.rglob(expected) if path.is_file())
            if len(matches) > 1:
                raise ValueError("Submission directory contains multiple {0} files.".format(expected))
            if matches:
                contents[expected] = matches[0].read_text(encoding="utf-8")
    else:
        raise ValueError("Submission path does not exist: {0}".format(submission_path))

    missing = [filename for filename in CHALLENGE_SUBMISSION_FILES if filename not in contents]
    if missing:
        raise ValueError(
            "Challenge submission is missing required file(s): {0}".format(
                ", ".join(missing)
            )
        )
    return contents


def _canonical_submission_id(
    row: Mapping[str, Any],
    *,
    row_number: int,
    source: str,
    reference_ids: Mapping[str, str],
    reference_aliases: Mapping[str, Optional[str]],
) -> str:
    sample_id_value, _ = get_first_present(dict(row), CHALLENGE_ID_KEYS)
    if sample_id_value is None:
        raise ValueError("{0}:{1} missing id.".format(source, row_number))
    raw_id = str(sample_id_value).strip()
    if not raw_id:
        raise ValueError("{0}:{1} has an empty id.".format(source, row_number))
    canonical_id = _resolve_submission_id(
        raw_id,
        reference_ids=reference_ids,
        reference_aliases=reference_aliases,
    )
    if canonical_id is None:
        raise ValueError("{0}:{1} contains unknown or ambiguous id: {2}".format(source, row_number, raw_id))
    return canonical_id


def _add_challenge_field(
    combined_rows: Dict[str, Dict[str, Any]],
    canonical_id: str,
    field_name: str,
    field_value: Any,
) -> None:
    row = combined_rows.setdefault(
        canonical_id,
        {
            "sample_id": canonical_id,
            "id": canonical_id,
        },
    )
    row[field_name] = field_value


def _merge_detection_rows(
    combined_rows: Dict[str, Dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_ids: Mapping[str, str],
    reference_aliases: Mapping[str, Optional[str]],
    label_keys: Sequence[str],
) -> None:
    seen_ids = set()
    for row_number, row in enumerate(rows, start=1):
        canonical_id = _canonical_submission_id(
            row,
            row_number=row_number,
            source=CHALLENGE_DETECTION_FILE,
            reference_ids=reference_ids,
            reference_aliases=reference_aliases,
        )
        if canonical_id in seen_ids:
            raise ValueError("{0} contains duplicate id: {1}".format(CHALLENGE_DETECTION_FILE, canonical_id))
        seen_ids.add(canonical_id)
        label_value, label_key = get_first_present(dict(row), label_keys)
        if label_key is None:
            raise ValueError(
                "{0}:{1} id '{2}' missing predicted label. Tried keys: {3}".format(
                    CHALLENGE_DETECTION_FILE,
                    row_number,
                    canonical_id,
                    ", ".join(label_keys),
                )
            )
        if _normalize_label(label_value) is None:
            raise ValueError(
                "{0}:{1} id '{2}' has invalid predicted label: {3!r}".format(
                    CHALLENGE_DETECTION_FILE,
                    row_number,
                    canonical_id,
                    label_value,
                )
            )
        _add_challenge_field(combined_rows, canonical_id, "pred_label", label_value)


def _merge_explanation_rows(
    combined_rows: Dict[str, Dict[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_ids: Mapping[str, str],
    reference_aliases: Mapping[str, Optional[str]],
    source: str,
    field_name: str,
    candidate_keys: Sequence[str],
) -> None:
    seen_ids = set()
    for row_number, row in enumerate(rows, start=1):
        canonical_id = _canonical_submission_id(
            row,
            row_number=row_number,
            source=source,
            reference_ids=reference_ids,
            reference_aliases=reference_aliases,
        )
        if canonical_id in seen_ids:
            raise ValueError("{0} contains duplicate id: {1}".format(source, canonical_id))
        seen_ids.add(canonical_id)
        explanation_value, explanation_key = get_first_present(dict(row), candidate_keys)
        if explanation_key is None:
            raise ValueError(
                "{0}:{1} id '{2}' missing {3}. Tried keys: {4}".format(
                    source,
                    row_number,
                    canonical_id,
                    field_name,
                    ", ".join(candidate_keys),
                )
            )
        if not isinstance(explanation_value, str):
            raise ValueError(
                "{0}:{1} id '{2}' has non-string {3}.".format(
                    source,
                    row_number,
                    canonical_id,
                    field_name,
                )
            )
        _add_challenge_field(combined_rows, canonical_id, field_name, explanation_value)


def _load_challenge_submission_rows(
    submission_path: Path,
    reference_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    reference_ids, reference_aliases, ordered_reference_ids = _build_reference_alias_map(
        reference_rows,
        args.reference_id_keys,
    )
    files = _read_challenge_submission_files(submission_path)
    combined_rows: Dict[str, Dict[str, Any]] = {}
    _merge_detection_rows(
        combined_rows,
        _parse_jsonl_text(files[CHALLENGE_DETECTION_FILE], source=CHALLENGE_DETECTION_FILE),
        reference_ids=reference_ids,
        reference_aliases=reference_aliases,
        label_keys=args.submission_label_keys,
    )
    _merge_explanation_rows(
        combined_rows,
        _parse_jsonl_text(files[CHALLENGE_COMPLEX_FILE], source=CHALLENGE_COMPLEX_FILE),
        reference_ids=reference_ids,
        reference_aliases=reference_aliases,
        source=CHALLENGE_COMPLEX_FILE,
        field_name="complex_explanation",
        candidate_keys=args.submission_complex_keys,
    )
    _merge_explanation_rows(
        combined_rows,
        _parse_jsonl_text(files[CHALLENGE_SIMPLE_FILE], source=CHALLENGE_SIMPLE_FILE),
        reference_ids=reference_ids,
        reference_aliases=reference_aliases,
        source=CHALLENGE_SIMPLE_FILE,
        field_name="simple_explanation",
        candidate_keys=args.submission_simple_keys,
    )
    return [combined_rows[sample_id] for sample_id in ordered_reference_ids if sample_id in combined_rows]


def _load_submission_rows(args: argparse.Namespace, reference_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    submission_path = Path(args.submission)
    if submission_path.is_file() and submission_path.suffix.lower() == ".zip":
        submission_format = "challenge_zip"
    elif submission_path.is_dir():
        submission_format = "challenge_folder"
    else:
        raise ValueError(
            "Submission must be either a .zip file or an extracted folder containing "
            "detection.jsonl, complex.jsonl, and simple.jsonl: {0}".format(submission_path)
        )
    return _load_challenge_submission_rows(submission_path, reference_rows, args), submission_format


def _first_existing_file(folder: Path, candidates: Sequence[str]) -> Optional[Path]:
    for candidate in candidates:
        path = folder / candidate
        if path.exists():
            return path
    return None


def _resolve_ground_truth_paths(args: argparse.Namespace) -> None:
    if args.ground_truth_split and args.ground_truth_dir:
        raise ValueError("Use either --ground-truth-split or --ground-truth-dir, not both.")

    ground_truth_dir = None
    if args.ground_truth_split:
        ground_truth_dir = DEFAULT_GROUND_TRUTH_ROOT / args.ground_truth_split
    elif args.ground_truth_dir:
        ground_truth_dir = Path(args.ground_truth_dir)
    else:
        default_dir = DEFAULT_GROUND_TRUTH_ROOT / DEFAULT_GROUND_TRUTH_SPLIT
        if default_dir.exists():
            ground_truth_dir = default_dir

    if ground_truth_dir is not None:
        ground_truth_dir = ground_truth_dir.expanduser().resolve()
        if not ground_truth_dir.exists():
            raise FileNotFoundError("Ground-truth folder not found: {0}".format(ground_truth_dir))
        args.ground_truth_dir = ground_truth_dir

        if args.ground_truth is None:
            reference_path = _first_existing_file(ground_truth_dir, REFERENCE_FILE_CANDIDATES)
            if reference_path is None:
                raise FileNotFoundError(
                    "Could not find a reference JSONL in {0}. Tried: {1}".format(
                        ground_truth_dir,
                        ", ".join(REFERENCE_FILE_CANDIDATES),
                    )
                )
            args.ground_truth = reference_path

        if args.gt_entity_facts is None:
            gt_entity_facts = _first_existing_file(ground_truth_dir, GT_ENTITY_FACT_FILE_CANDIDATES)
            if gt_entity_facts is not None:
                args.gt_entity_facts = gt_entity_facts
            else:
                args.gt_entity_facts = ground_truth_dir / GT_ENTITY_FACT_FILE_CANDIDATES[0]

    if args.ground_truth is None:
        args.ground_truth = DEFAULT_GROUND_TRUTH_PATH

    args.ground_truth = Path(args.ground_truth).expanduser().resolve()
    if args.gt_entity_facts is not None:
        args.gt_entity_facts = Path(args.gt_entity_facts).expanduser().resolve()


def _load_gt_entity_fact_cache(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not Path(path).exists():
        return {}

    cache_rows: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id", row.get("id"))
        if sample_id is None:
            continue
        if row.get("status") not in (None, "ok"):
            continue
        try:
            diagnostic_entities = get_reference_entities(row)
            evidence_claims = get_reference_claims(row)
        except Exception:
            continue
        if not isinstance(diagnostic_entities, list) or not isinstance(evidence_claims, list):
            continue
        cache_rows[str(sample_id)] = row
    return cache_rows


def _reference_payload_from_gt_cache(
    sample_id: str,
    reference_text: str,
    cache_row: Mapping[str, Any],
    *,
    verify_text_hash: bool,
) -> Optional[Dict[str, Any]]:
    text_sha256 = cache_row.get("text_sha256") or cache_row.get("reference_text_sha256")
    if verify_text_hash and text_sha256 and text_sha256 != compute_hash(reference_text):
        return None
    return {
        "sample_id": sample_id,
        "reference_text_sha256": compute_hash(reference_text),
        "text_sha256": compute_hash(reference_text),
        "explanation": reference_text,
        "diagnostic_entities": get_reference_entities(dict(cache_row)),
        "evidence_claims": get_reference_claims(dict(cache_row)),
        "judge_model": cache_row.get("judge_model"),
        "source": "gt_entity_facts_cache",
    }


def _gt_entity_fact_cache_rows(rows: Sequence[Mapping[str, Any]], *, judge_model: str) -> List[Dict[str, Any]]:
    cache_rows: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.get("_gt_extraction")
        reference_text = row.get("_reference_complex_text")
        if not payload or not reference_text:
            continue
        cache_rows.append(
            {
                "sample_id": row["sample_id"],
                "status": "ok",
                "label": row.get("ground_truth_label"),
                "reference_complex_key": row.get("_reference_complex_key"),
                "text_sha256": compute_hash(reference_text),
                "judge_model": payload.get("judge_model") or judge_model,
                "diagnostic_entities": get_reference_entities(payload),
                "evidence_claims": get_reference_claims(payload),
            }
        )
    return cache_rows


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


def _prepare_rows(aligned_rows, args, show_progress, gt_entity_fact_cache=None):
    gt_entity_fact_cache = gt_entity_fact_cache or {}
    rows: List[Dict[str, Any]] = []
    sample_iterator = build_progress_bar(
        aligned_rows,
        total=len(aligned_rows),
        desc="Prepare samples",
        disable=not show_progress,
    )
    for sample_id, submission_row, reference_row in sample_iterator:
        submission_present = bool(submission_row)
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "submission_present": submission_present,
            "predicted_label": None,
            "ground_truth_label": None,
            "label_correct": None,
            "complex_bert_f1": None,
            "complex_entity_f1": None,
            "complex_evidence_f1": None,
            "complex_overall_score": None,
            "simple_bert_f1": None,
            "simple_sle_score": None,
            "simple_overall_score": None,
            "_submission_complex_text": None,
            "_reference_complex_text": None,
            "_reference_complex_key": None,
            "_submission_simple_text": None,
            "_reference_simple_text": None,
            "_gt_extraction": None,
            "_gt_extraction_source": None,
            "_pred_extraction": None,
            "_gt_to_pred_entity": None,
            "_gt_to_pred_evidence": None,
            "_pred_to_gt_entity": None,
            "_pred_to_gt_evidence": None,
        }

        normalized_reference_label = None
        try:
            reference_label, _ = extract_required_text(
                reference_row,
                args.reference_label_keys,
                field_role="reference label",
                sample_id=sample_id,
            )
            normalized_reference_label = _normalize_label(reference_label)
            row["ground_truth_label"] = normalized_reference_label
        except Exception as exc:
            _write_progress_message(
                sample_iterator,
                "Warning: reference label preparation failed for {0}: {1}".format(sample_id, exc),
            )

        if submission_present:
            try:
                submission_label, _ = extract_required_text(
                    submission_row,
                    args.submission_label_keys,
                    field_role="submission label",
                    sample_id=sample_id,
                )
                normalized_submission_label = _normalize_label(submission_label)
                row["predicted_label"] = normalized_submission_label
            except Exception as exc:
                _write_progress_message(
                    sample_iterator,
                    "Warning: submission label preparation failed for {0}: {1}".format(sample_id, exc),
                )
        if normalized_reference_label is not None:
            row["label_correct"] = row.get("predicted_label") == normalized_reference_label

        if submission_present:
            try:
                row["_submission_complex_text"], _ = extract_required_text(
                    submission_row,
                    args.submission_complex_keys,
                    field_role="submission complex explanation",
                    sample_id=sample_id,
                )
                row["_reference_complex_text"], row["_reference_complex_key"] = extract_required_text(
                    reference_row,
                    args.reference_complex_keys,
                    field_role="reference complex explanation",
                    sample_id=sample_id,
                )
                cache_payload = gt_entity_fact_cache.get(sample_id)
                if cache_payload is not None:
                    row["_gt_extraction"] = _reference_payload_from_gt_cache(
                        sample_id,
                        row["_reference_complex_text"],
                        cache_payload,
                        verify_text_hash=not args.no_verify_gt_entity_facts,
                    )
                    if row["_gt_extraction"] is not None:
                        row["_gt_extraction_source"] = "cache"
            except Exception as exc:
                _write_progress_message(
                    sample_iterator,
                    "Warning: complex text preparation failed for {0}: {1}".format(sample_id, exc),
                )

        if submission_present:
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
    active_indices = [
        index
        for index, row in enumerate(rows)
        if row.get(text_key) and row.get(output_key) is None
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
                if output_key == "_gt_extraction":
                    rows[index]["_gt_extraction_source"] = "computed"
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
    evidence_output_key,
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
                rows[index][evidence_output_key] = round_float(summary.get("claim_coverage", 0.0))
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
        row["complex_evidence_f1"] = _compute_harmonic_mean_if_all_present(
            row.get("_pred_to_gt_evidence"),
            row.get("_gt_to_pred_evidence"),
        )
        row["complex_overall_score"] = _compute_complex_overall_score(
            row.get("complex_bert_f1"),
            row.get("complex_entity_f1"),
            row.get("complex_evidence_f1"),
        )
        row["simple_overall_score"] = _compute_simple_overall_score(
            row.get("simple_bert_f1"),
            row.get("simple_sle_score"),
        )
        row["simple_sle_normalized"] = _normalize_simple_sle(row.get("simple_sle_score"))
        row["reference_explanation_score"] = _compute_reference_explanation_score(
            row.get("complex_bert_f1"),
            row.get("simple_overall_score"),
        )
        row["grounding_score"] = _compute_mean_if_all_present(
            row.get("complex_entity_f1"),
            row.get("complex_evidence_f1"),
        )
        row["explanation_score"] = _compute_paper_explanation_score(
            row.get("reference_explanation_score"),
            row.get("grounding_score"),
        )
        finalized_rows.append(
            {
                "sample_id": row["sample_id"],
                "submission_present": row.get("submission_present"),
                "predicted_label": row.get("predicted_label"),
                "ground_truth_label": row.get("ground_truth_label"),
                "label_correct": row.get("label_correct"),
                "complex_bert_f1": row.get("complex_bert_f1"),
                "complex_explanation_score": row.get("complex_bert_f1"),
                "complex_entity_f1": row.get("complex_entity_f1"),
                "complex_evidence_f1": row.get("complex_evidence_f1"),
                "complex_overall_score": row.get("complex_overall_score"),
                "simple_bert_f1": row.get("simple_bert_f1"),
                "simple_sle_score": row.get("simple_sle_score"),
                "simple_sle_normalized": row.get("simple_sle_normalized"),
                "simple_overall_score": row.get("simple_overall_score"),
                "simple_explanation_score": row.get("simple_overall_score"),
                "reference_explanation_score": row.get("reference_explanation_score"),
                "grounding_score": row.get("grounding_score"),
                "explanation_score": row.get("explanation_score"),
            }
        )
    return finalized_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation in stage-wise batches on one GPU."
    )
    parser.add_argument("--submission", required=True)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument(
        "--ground-truth-split",
        choices=["val", "test"],
        default=None,
        help="Use evaluation/ground_truth/<split>. Defaults to val when that folder exists.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Folder containing reference.jsonl and complex_ground_truth_entity_facts.jsonl.",
    )
    parser.add_argument(
        "--gt-entity-facts",
        type=Path,
        default=None,
        help="Optional precomputed ground-truth entity/fact JSONL cache.",
    )
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--submission-id-keys", nargs="+", default=list(DEFAULT_ID_KEYS))
    parser.add_argument("--reference-id-keys", nargs="+", default=list(DEFAULT_ID_KEYS))
    parser.add_argument("--submission-label-keys", nargs="+", default=list(DEFAULT_LABEL_KEYS))
    parser.add_argument("--reference-label-keys", nargs="+", default=["label"])
    parser.add_argument("--submission-complex-keys", nargs="+", default=["complex_explanation"])
    parser.add_argument("--reference-complex-keys", nargs="+", default=["complex_explanation", "complex_reference"])
    parser.add_argument("--submission-simple-keys", nargs="+", default=["simple_explanation"])
    parser.add_argument("--reference-simple-keys", nargs="+", default=["simple_explanation", "simple_reference"])

    parser.add_argument("--entity-evidence-prompt", type=Path, default=Path(__file__).resolve().parent / "prompts" / "entity_evidence_extraction_prompt.txt")
    parser.add_argument("--semantic-coverage-prompt", type=Path, default=Path(__file__).resolve().parent / "prompts" / "semantic_coverage_prompt.txt")

    parser.add_argument("--bertscore-model-type", default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--bertscore-lang", default="en")
    parser.add_argument("--bertscore-rescale-with-baseline", action="store_true", default=False)
    parser.add_argument("--bertscore-batch-size", type=int, default=DEFAULT_BERT_BATCH_SIZE)

    parser.add_argument("--backend", choices=["transformers", "openai_compatible", "vllm"], default="transformers")
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
    parser.add_argument("--no-verify-gt-entity-facts", action="store_true", default=False)
    parser.add_argument("--no-update-gt-entity-facts", action="store_true", default=False)
    parser.add_argument("--no-preload-models", action="store_true", default=False)
    parser.add_argument("--no-progress", action="store_true", default=False)

    parser.add_argument("--sle-model-id", default="liamcripwell/sle-base")
    parser.add_argument("--sle-batch-size", type=int, default=DEFAULT_SLE_BATCH_SIZE)
    parser.add_argument("--sle-max-length", type=int, default=512)
    parser.add_argument("--sle-local-files-only", action="store_true", default=False)

    args = parser.parse_args()
    args.preload_models = not args.no_preload_models
    args.show_progress = not args.no_progress
    _resolve_ground_truth_paths(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_output_path = output_dir / "per_sample_scores.jsonl"
    final_scores_output_path = output_dir / "final_scores.json"

    reference_rows = read_jsonl(args.ground_truth)
    gt_entity_fact_cache = _load_gt_entity_fact_cache(args.gt_entity_facts)
    submission_rows, resolved_submission_format = _load_submission_rows(args, reference_rows)
    aligned_rows, diagnostics = align_submission_and_reference(
        submission_rows,
        reference_rows,
        submission_id_keys=args.submission_id_keys,
        reference_id_keys=args.reference_id_keys,
        include_missing_submission=True,
    )

    if diagnostics:
        print("Alignment diagnostics: {0}".format(len(diagnostics)))
    print("Submission format: {0}".format(resolved_submission_format))
    print("Ground truth: {0}".format(args.ground_truth))
    if args.gt_entity_facts:
        print(
            "GT entity/fact cache: {0} ({1} rows loaded)".format(
                args.gt_entity_facts,
                len(gt_entity_fact_cache),
            )
        )
    else:
        print("GT entity/fact cache: none; GT entities/facts will be computed if Qwen stages run.")

    rows = _prepare_rows(aligned_rows, args, args.show_progress, gt_entity_fact_cache)
    if args.skip_qwen:
        print("Skipping Qwen extraction and coverage. Qwen-based complex metrics will be written as null.")
    else:
        extraction_prompt = load_text(args.entity_evidence_prompt)
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
        computed_gt_count = sum(1 for row in rows if row.get("_gt_extraction_source") == "computed")
        cached_gt_count = sum(1 for row in rows if row.get("_gt_extraction_source") == "cache")
        if computed_gt_count and args.gt_entity_facts and not args.no_update_gt_entity_facts:
            _write_jsonl(
                args.gt_entity_facts,
                _gt_entity_fact_cache_rows(rows, judge_model=args.model_name),
            )
            print(
                "Updated GT entity/fact cache: {0} ({1} computed, {2} cached)".format(
                    args.gt_entity_facts,
                    computed_gt_count,
                    cached_gt_count,
                )
            )
        elif cached_gt_count:
            print("Using precomputed GT entity/fact rows: {0}".format(cached_gt_count))
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
            evidence_output_key="_gt_to_pred_evidence",
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
            evidence_output_key="_pred_to_gt_evidence",
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
    final_scores = _build_final_scores(finalized_rows)
    final_scores["submission_format"] = resolved_submission_format
    final_scores["ground_truth"] = str(args.ground_truth)
    final_scores["ground_truth_dir"] = str(args.ground_truth_dir) if args.ground_truth_dir else None
    final_scores["gt_entity_facts"] = str(args.gt_entity_facts) if args.gt_entity_facts else None
    final_scores["gt_entity_fact_cache_rows_loaded"] = len(gt_entity_fact_cache)
    final_scores["diagnostic_count"] = len(diagnostics)
    final_scores["diagnostics"] = diagnostics[:100]
    write_json(final_scores_output_path, final_scores)

    print("Wrote per-sample scores to: {0}".format(per_sample_output_path))
    print("Wrote final scores to: {0}".format(final_scores_output_path))


if __name__ == "__main__":
    main()
