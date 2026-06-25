# XPlainVerse Evaluation

This repository contains evaluation utilities for XPlainVerse and the Explainable Deepfake Detection Challenge. It provides reference-based metric code, validation references, and baseline documentation for systems that detect manipulated images and generate visual explanations.

The evaluation covers three outputs:

- a detection label: real or fake
- a complex explanation: a detailed visual explanation grounded in image evidence
- a simple explanation: a shorter, accessible explanation preserving the key reason

## Repository Structure

- `evaluation/evaluate_val.py`
  Main evaluator for the challenge three-file submission format.
- `evaluation/evaluate_complex_explanations.py`
  Standalone complex-explanation evaluator.
- `evaluation/evaluate_simple_explanations.py`
  Standalone simple-explanation evaluator.
- `evaluation/combine_explanation_scores.py`
  Helper for combining standalone complex and simple explanation reports.
- `evaluation/ground_truth/`
  Split-specific reference folders with `reference.jsonl` and precomputed GT entity/fact caches.
- `evaluation/prompts/`
  Prompt templates used by the LLM-based grounding metrics.
- `evaluation/utils/`
  Shared loading, batching, and scoring helpers.
- `evaluation/env/`
  Reproducible environment files.
- `baselines/README.md`
  Baseline model setup and validation results.

## Reference Data

`evaluation/ground_truth/` provides split folders:

- `evaluation/ground_truth/val/`
- `evaluation/ground_truth/test/`

Each folder is expected to contain:

- `reference.jsonl`
- `complex_ground_truth_entity_facts.jsonl`

When a GT entity/fact cache is available, the evaluator uses it automatically and skips recomputing ground-truth entity/fact extraction for matching rows. Missing cache rows are computed only if the Qwen/vLLM stages run; the cache is updated unless `--no-update-gt-entity-facts` is passed.

Hidden final-test labels and reference explanations should not be included in a public release. For local/private evaluation, `evaluation/ground_truth/test/` may be populated with the hidden final-test reference and cache.

## Evaluator Input Format

`evaluation/evaluate_val.py` accepts the official challenge format only: either a zip file or the extracted folder containing these three JSONL files:

- `detection.jsonl`
- `complex.jsonl`
- `simple.jsonl`

Official final-test submissions are handled on CodaBench:

https://www.codabench.org/competitions/16461/

`detection.jsonl` should contain one row for every final-test image:

```json
{"id": "sample1.png", "pred_label": 1}
```

`pred_label` must be:

- `0` for real
- `1` for fake

`complex.jsonl` should contain complex explanations for each evaluated image:

```json
{"id": "sample1.png", "complex_explanation": "A detailed explanation grounded in visible image evidence."}
```

`simple.jsonl` should contain simple explanations for each evaluated image:

```json
{"id": "sample1.png", "simple_explanation": "A simple explanation for the image."}
```

The `id` value should match the reference id. The local evaluator accepts common filename aliases, such as full paths or ids with/without image extensions, when they resolve unambiguously.

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

The environment file installs Python, PyTorch, `transformers`, BERTScore, and the SLE dependencies used by the evaluator. The LLM-judge stages can run through a vLLM OpenAI-compatible server by using `--backend vllm`.

## Running Evaluation

From inside `evaluation/`:

```bash
python evaluate_val.py \
  --submission /path/to/submission.zip \
  --output-dir /path/to/results
```

You can also pass the extracted submission folder:

```bash
python evaluate_val.py \
  --submission /path/to/submission_folder \
  --output-dir /path/to/results
```

By default, the evaluator uses:

- `evaluation/ground_truth/val/reference.jsonl`
- `evaluation/ground_truth/val/complex_ground_truth_entity_facts.jsonl`

You can choose a bundled split:

```bash
python evaluate_val.py \
  --submission /path/to/submission.zip \
  --ground-truth-split test \
  --output-dir /path/to/results
```

You can also pass a custom folder:

```bash
python evaluate_val.py \
  --submission /path/to/submission.zip \
  --ground-truth-dir /path/to/ground_truth_folder \
  --output-dir /path/to/results
```

Or override the reference/cache files directly:

```bash
python evaluate_val.py \
  --submission /path/to/submission.zip \
  --ground-truth /path/to/reference.jsonl \
  --gt-entity-facts /path/to/complex_ground_truth_entity_facts.jsonl \
  --output-dir /path/to/results
```

### vLLM Judge Backend

The official/recommended path for the LLM grounding stages is a vLLM OpenAI-compatible server. Start the server in a GPU environment, then point the evaluator at it:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8000
```

```bash
python evaluate_val.py \
  --submission /path/to/submission.zip \
  --output-dir /path/to/results \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --model-name Qwen/Qwen3.5-4B
```

`--backend openai_compatible` is equivalent if you already use that naming. `--backend transformers` remains available for local in-process model loading.

To skip LLM-based grounding metrics for a faster debug run:

```bash
python evaluate_val.py \
  --submission /path/to/submission.zip \
  --output-dir /path/to/results \
  --skip-qwen
```

With `--skip-qwen`, the evaluator still computes:

- detection macro F1
- detection accuracy
- complex BERT F1
- simple BERT F1
- simple SLE score
- simple SLE normalized
- simple overall score

The paper-style final score still includes the LLM grounding term, so skipped or missing grounding scores contribute zero to that final aggregate. Use `--skip-qwen` for debugging, not final ranking.

## Output Files

The main evaluator writes:

- `per_sample_scores.jsonl`
- `final_scores.json`

`per_sample_scores.jsonl` contains one row per aligned sample with label, complex, simple, and combined explanation scores.

`final_scores.json` contains dataset-level means, including:

- `detection_macro_f1`
- `detection_fake_f1`
- `detection_real_f1`
- `detection_accuracy`
- `complex_bert_f1`
- `complex_entity_f1`
- `complex_evidence_f1`
- `simple_bert_f1`
- `simple_sle_score`
- `simple_sle_normalized`
- `simple_overall_score`
- `reference_explanation_score`
- `grounding_score`
- `explanation_score`
- `final_score`
- `overall_score`

## Scoring

The evaluator uses the ACM MM 2026 paper scoring protocol.

Detection macro F1 averages the fake-positive and real-positive F1 scores:

```text
detection_macro_f1 = (detection_fake_f1 + detection_real_f1) / 2
```

Detection accuracy is reported for reference:

```text
detection_accuracy = correct_labels / total_reference_samples
```

Simple SLE normalization:

```text
simple_sle_normalized = (clip(simple_sle_score, -1, 4) + 1) / 5
```

Simple overall:

```text
simple_overall_score = 0.7 * simple_bert_f1 + 0.3 * simple_sle_normalized
```

The complex explanation score is the complex BERTScore-F1:

```text
complex_explanation_score = complex_bert_f1
```

The reference-based explanation subscore is:

```text
reference_explanation_score = (complex_explanation_score + simple_overall_score) / 2
```

The LLM grounding subscore is:

```text
grounding_score = (complex_entity_f1 + complex_evidence_f1) / 2
```

The final explanation score weights reference similarity and LLM grounding as described in the paper:

```text
explanation_score = 0.4 * reference_explanation_score + 0.6 * grounding_score
```

The final challenge score is:

```text
final_score = overall_score = (detection_macro_f1 + explanation_score) / 2
```

Missing rows or failed metric rows contribute zero to dataset-level metric means, so the denominator remains the number of reference samples.

## Baseline Results

Validation baseline results for the fine-tuned models are reported in [`baselines/README.md`](baselines/README.md#results).

## Notes

- `evaluate_val.py` can be used for any split where reference labels and explanations are available.
- Challenge final-test scoring uses hidden references on the evaluation platform.
- For validation `real` samples, where no separate simple explanation exists, the bundled reference uses the complex explanation as the simple explanation.
