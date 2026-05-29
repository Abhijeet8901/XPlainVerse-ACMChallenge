# XPlainVerse Evaluation

This repository contains evaluation utilities for XPlainVerse and the Explainable Deepfake Detection Challenge. It provides reference-based metric code, validation references, and baseline documentation for systems that detect manipulated images and generate visual explanations.

The evaluation covers three outputs:

- a detection label: real or fake
- a complex explanation: a detailed visual explanation grounded in image evidence
- a simple explanation: a shorter, accessible explanation preserving the key reason

## Repository Structure

- `evaluation/evaluate_val.py`
  Main reference-based evaluator for JSONL prediction files.
- `evaluation/evaluate_complex_explanations.py`
  Standalone complex-explanation evaluator.
- `evaluation/evaluate_simple_explanations.py`
  Standalone simple-explanation evaluator.
- `evaluation/combine_explanation_scores.py`
  Helper for combining standalone complex and simple explanation reports.
- `evaluation/data/val_ground_truth.jsonl`
  Validation reference file.
- `evaluation/prompts/`
  Prompt templates used by optional LLM-based complex diagnostic metrics.
- `evaluation/utils/`
  Shared loading, batching, and scoring helpers.
- `evaluation/env/`
  Reproducible environment files.
- `baselines/README.md`
  Baseline model setup and validation results.

## Reference Data

`evaluation/data/val_ground_truth.jsonl` contains the released validation labels and reference explanations in the format expected by the evaluator. This lets participants reproduce validation metrics without converting the dataset into a separate reference file.

Hidden final-test labels and reference explanations are not included in this repository.

## Evaluator Input Format

For reference-based metric evaluation with `evaluation/evaluate_val.py`, use one JSONL file. Each row should contain the sample id, a predicted label, and both explanations:

```json
{
  "sample_id": "000001",
  "label": "fake",
  "complex_explanation": "A detailed explanation grounded in visible image evidence.",
  "simple_explanation": "A short explanation with the key reason."
}
```

The evaluator accepts either string labels:

- `real`
- `fake`

or numeric `pred_label` values:

- `0` for real
- `1` for fake

For example:

```json
{
  "sample_id": "000001",
  "pred_label": 1,
  "complex_explanation": "...",
  "simple_explanation": "..."
}
```

Rows are aligned by `sample_id` against the provided reference file.

## Challenge Submission Format

When submitting to the Explainable Deepfake Detection Challenge platform, submit a zip file containing these three files at the zip root:

- `detection.jsonl`
- `complex.jsonl`
- `simple.jsonl`

Do not put the files inside an extra top-level folder.

Official final-test submissions are handled on CodaBench:

https://www.codabench.org/competitions/16461/

`detection.jsonl` should contain one row for every final-test image:

```json
{"id": "sample1.png", "pred_label": 1}
```

`pred_label` must be:

- `0` for real
- `1` for fake

`complex.jsonl` should contain complex explanations for the explanation-evaluation subset:

```json
{"id": "sample1.png", "complex_explanation": "A detailed explanation grounded in visible image evidence."}
```

`simple.jsonl` should contain simple explanations for the same explanation-evaluation subset:

```json
{"id": "sample1.png", "simple_explanation": "A simple explanation for the image."}
```

The `id` value must exactly match the released test image filename, including extension.

Example zip command:

```bash
cd /path/to/submission_folder
zip -r ../submission.zip detection.jsonl complex.jsonl simple.jsonl
```

## Environment Setup

From inside `evaluation/`:

```bash
conda env create -f env/xplainverse_eval_env.yml
conda activate xplainverse_eval_full
```

The environment file installs Python, PyTorch, `transformers`, BERTScore, and the SLE dependencies used by the evaluator.

## Running Evaluation

From inside `evaluation/`:

```bash
python evaluate_val.py \
  --submission /path/to/submission.jsonl \
  --output-dir /path/to/results
```

By default, the evaluator uses:

- `evaluation/data/val_ground_truth.jsonl`

You can override the reference file if needed:

```bash
python evaluate_val.py \
  --submission /path/to/submission.jsonl \
  --ground-truth /path/to/ground_truth.jsonl \
  --output-dir /path/to/results
```

To skip optional LLM-based complex diagnostic metrics:

```bash
python evaluate_val.py \
  --submission /path/to/submission.jsonl \
  --output-dir /path/to/results \
  --skip-qwen
```

With `--skip-qwen`, the evaluator still computes:

- detection F1
- detection accuracy
- complex BERT F1
- simple BERT F1
- simple SLE score
- simple SLE normalized
- simple overall score
- explanation score
- overall score

and writes optional Qwen-dependent complex diagnostic fields as `null`:

- `complex_entity_f1`
- `complex_evidence_f1`
- `complex_overall_score`

## Output Files

The main evaluator writes:

- `per_sample_scores.jsonl`
- `final_scores.json`

`per_sample_scores.jsonl` contains one row per aligned sample with label, complex, simple, and combined explanation scores.

`final_scores.json` contains dataset-level means, including:

- `detection_f1`
- `detection_accuracy`
- `complex_bert_f1`
- `complex_entity_f1`
- `complex_evidence_f1`
- `simple_bert_f1`
- `simple_sle_score`
- `simple_sle_normalized`
- `simple_overall_score`
- `explanation_score`
- `overall_score`

## Scoring

Detection F1 treats `fake` as the positive class.

Detection accuracy:

```text
correct_labels / total_aligned_samples
```

Simple SLE normalization:

```text
simple_sle_normalized = (clip(simple_sle_score, -1, 4) + 1) / 5
```

Simple overall:

```text
simple_overall_score = 0.7 * simple_bert_f1 + 0.3 * simple_sle_normalized
```

Explanation score:

```text
explanation_score = (complex_bert_f1 + simple_overall_score) / 2
```

Overall score:

```text
overall_score = (detection_f1 + explanation_score) / 2
```

For the Explainable Deepfake Detection Challenge, the primary explanation score uses:

- `complex_bert_f1` on the 10,000-sample explanation-evaluation subset
- `simple_overall_score` on the same explanation-evaluation subset
- `explanation_score = (complex_bert_f1 + simple_overall_score) / 2`

The challenge leaderboard is sorted by `overall_score`, which combines detection and explanation performance.

The challenge leaderboard does not use the optional LLM-based `complex_entity_f1` and `complex_evidence_f1` metrics for all submissions. For the top 5 teams, organizers will additionally compute `complex_entity_f1` and `complex_evidence_f1` for final reporting and analysis.

## Baseline Results

Validation baseline results for the fine-tuned models are reported in [`baselines/README.md`](baselines/README.md#results).

## Notes

- `evaluate_val.py` can be used for any split where reference labels and explanations are available.
- Challenge final-test scoring uses hidden references on the evaluation platform.
- For validation `real` samples, where no separate simple explanation exists, the bundled reference uses the complex explanation as the simple explanation.
