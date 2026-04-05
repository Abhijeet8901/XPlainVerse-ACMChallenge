from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def round_float(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = [float(value) for value in values]
    if not values:
        return None
    return mean(values)


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected each JSONL line to be a JSON object, found {type(payload).__name__} at line {line_number}."
                )
            rows.append(payload)
    return rows


def write_json(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def get_first_present(row: Dict[str, Any], keys: Sequence[str]) -> Tuple[Optional[Any], Optional[str]]:
    for key in keys:
        if key not in row:
            continue
        return row[key], key
    return None, None


def extract_required_text(
    row: Dict[str, Any],
    keys: Sequence[str],
    *,
    field_role: str,
    sample_id: str,
) -> Tuple[str, str]:
    value, key = get_first_present(row, keys)
    if key is None:
        raise ValueError(
            f"Missing {field_role}. None of these keys were found for sample '{sample_id}': {', '.join(keys)}"
        )

    if isinstance(value, str):
        text = normalize_text(value)
    elif isinstance(value, list):
        text = normalize_text(" ".join(str(item) for item in value if item is not None))
    else:
        text = normalize_text(str(value))

    if not text:
        raise ValueError(f"Empty {field_role} found under key '{key}' for sample '{sample_id}'.")
    return text, key


def index_rows_by_id(
    rows: Sequence[Dict[str, Any]],
    *,
    preferred_keys: Sequence[str],
    row_role: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    diagnostics: List[Dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        sample_id_value, id_key = get_first_present(row, preferred_keys)
        if id_key is None:
            diagnostics.append(
                {
                    "status": "error",
                    "reason": f"missing_id: none of the keys were found ({', '.join(preferred_keys)})",
                    "row_index": row_index,
                    "row_role": row_role,
                }
            )
            continue

        sample_id = str(sample_id_value)
        if sample_id in indexed:
            diagnostics.append(
                {
                    "status": "error",
                    "reason": f"duplicate_id: {sample_id}",
                    "row_index": row_index,
                    "row_role": row_role,
                    "sample_id": sample_id,
                }
            )
            continue

        indexed[sample_id] = row

    return indexed, diagnostics


def align_submission_and_reference(
    submission_rows: Sequence[Dict[str, Any]],
    reference_rows: Sequence[Dict[str, Any]],
    *,
    submission_id_keys: Sequence[str],
    reference_id_keys: Sequence[str],
) -> Tuple[List[Tuple[str, Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]]]:
    submission_by_id, submission_diagnostics = index_rows_by_id(
        submission_rows,
        preferred_keys=submission_id_keys,
        row_role="submission",
    )
    reference_by_id, reference_diagnostics = index_rows_by_id(
        reference_rows,
        preferred_keys=reference_id_keys,
        row_role="reference",
    )

    diagnostics = list(submission_diagnostics) + list(reference_diagnostics)
    aligned: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

    missing_from_submission = sorted(set(reference_by_id) - set(submission_by_id))
    for sample_id in missing_from_submission:
        diagnostics.append(
            {
                "status": "error",
                "reason": "missing_submission_row",
                "sample_id": sample_id,
            }
        )

    extra_in_submission = sorted(set(submission_by_id) - set(reference_by_id))
    for sample_id in extra_in_submission:
        diagnostics.append(
            {
                "status": "warning",
                "reason": "submission_row_not_in_reference",
                "sample_id": sample_id,
            }
        )

    seen_aligned = set()
    for row in submission_rows:
        sample_id_value, id_key = get_first_present(row, submission_id_keys)
        if id_key is None:
            continue

        sample_id = str(sample_id_value)
        if sample_id in seen_aligned:
            continue
        if sample_id not in submission_by_id:
            continue
        if sample_id not in reference_by_id:
            continue

        aligned.append((sample_id, submission_by_id[sample_id], reference_by_id[sample_id]))
        seen_aligned.add(sample_id)

    return aligned, diagnostics


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_file_for_sample(cache_dir: Path, sample_id: str) -> Path:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def load_cached_reference(cache_dir: Optional[Path], sample_id: str, reference_text: str) -> Optional[Dict[str, Any]]:
    if cache_dir is None:
        return None
    cache_path = cache_file_for_sample(cache_dir, sample_id)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("reference_text_sha256") != compute_hash(reference_text):
        return None
    return payload


def save_cached_reference(cache_dir: Optional[Path], sample_id: str, payload: Dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_file_for_sample(cache_dir, sample_id)
    write_json(cache_path, payload)


def summarize_bertscore(per_sample: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    precision_values = [item["bertscore_precision"] for item in per_sample if item.get("bertscore_precision") is not None]
    recall_values = [item["bertscore_recall"] for item in per_sample if item.get("bertscore_recall") is not None]
    f1_values = [item["bertscore_f1"] for item in per_sample if item.get("bertscore_f1") is not None]
    return {
        "bertscore_precision_mean": round_float(safe_mean(precision_values)),
        "bertscore_recall_mean": round_float(safe_mean(recall_values)),
        "bertscore_f1_mean": round_float(safe_mean(f1_values)),
    }


def build_base_report(
    *,
    metric_name: str,
    submission_path: Path,
    reference_path: Path,
    summary: Dict[str, Any],
    per_sample: List[Dict[str, Any]],
    config: Dict[str, Any],
    diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "metric": metric_name,
        "created_at_utc": utc_now_iso(),
        "submission_path": str(submission_path),
        "reference_path": str(reference_path),
        "summary": summary,
        "config": config,
        "diagnostics": diagnostics,
        "per_sample": per_sample,
    }
