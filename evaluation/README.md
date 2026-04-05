# Evaluation

This folder contains the official evaluation bundle for XPlainVerse challenge submissions.

## Contents

- `evaluate_val.py`
  Main end-to-end evaluator. This is the script most users should run.
- `evaluate_complex_explanations.py`
  Standalone evaluator for complex explanations only.
- `evaluate_simple_explanations.py`
  Standalone evaluator for simple explanations only.
- `combine_explanation_scores.py`
  Helper for combining legacy standalone report files.
- `prompts/`
  Prompt templates used for entity/fact extraction and semantic coverage.
- `utils/`
  Shared helpers for JSONL loading, score aggregation, model loading, and LLM calls.
- `env/`
  Reproducible environment files for Qwen, BERTScore, and SLE.
- `docs/`
  Supporting documentation, including environment setup notes.

## Expected Submission Format

Both the submission JSONL and the ground-truth JSONL should use one JSON object per line with this schema:

```json
{
  "sample_id": "000001",
  "label": "fake",
  "complex_explanation": "...",
  "simple_explanation": "..."
}
```

The evaluator aligns rows by `sample_id` and processes them in the order they appear in the submission file.

## Main Command

From the repository root:

```bash
cd evaluation

python evaluate_val.py \
  --submission ../test_data/predictions.jsonl \
  --ground-truth ../test_data/ground_truth.jsonl \
  --output-dir ../outputs
```

## Outputs

Running `evaluate_val.py` writes:

- `per_sample_scores.jsonl`
- `final_scores.json`

`per_sample_scores.jsonl` contains one line per sample:

```json
{
  "sample_id": "000001",
  "complex_bert_f1": 0.812345,
  "complex_entity_f1": 0.571429,
  "complex_facts_f1": 0.600000,
  "complex_overall_score": 0.652275,
  "simple_bert_f1": 0.901234,
  "simple_sle_score": 3.417281,
  "simple_overall_score": 0.895901
}
```

`final_scores.json` contains the dataset-level means of the same metrics.

## Metrics

- `complex_bert_f1`
  BERTScore F1 for the complex explanation.
- `complex_entity_f1`
  Harmonic mean of bidirectional entity coverage.
- `complex_facts_f1`
  Harmonic mean of bidirectional fact coverage.
- `complex_overall_score`
  `0.3 * complex_bert_f1 + 0.4 * complex_entity_f1 + 0.3 * complex_facts_f1`
- `simple_bert_f1`
  BERTScore F1 for the simple explanation.
- `simple_sle_score`
  Simplicity score from the SLE model.
- `simple_overall_score`
  `0.7 * simple_bert_f1 + 0.3 * simple_sle_norm`, where `simple_sle_norm = (clip(simple_sle_score, -1, 4) + 1) / 5`

## Runtime Design

The main evaluator is organized for one-GPU throughput:

1. prepare and align all samples
2. Qwen extraction for ground-truth complex explanations
3. Qwen extraction for predicted complex explanations
4. Qwen coverage for ground truth -> prediction
5. Qwen coverage for prediction -> ground truth
6. complex BERTScore in batches
7. simple BERTScore in batches
8. simple SLE in batches

Each stage has a progress bar.

## Useful Options

```bash
python evaluate_val.py \
  --submission ../test_data/predictions.jsonl \
  --ground-truth ../test_data/ground_truth.jsonl \
  --output-dir ../outputs \
  --qwen-batch-size 4 \
  --bertscore-batch-size 8 \
  --sle-batch-size 16
```

Important defaults:

- `--device-map cuda:0`
- `--model-name Qwen/Qwen3.5-4B`
- `--extraction-max-tokens 1024`
- `--coverage-max-tokens 1024`

## Environment Setup

See [docs/README_environment_setup.md](docs/README_environment_setup.md).
