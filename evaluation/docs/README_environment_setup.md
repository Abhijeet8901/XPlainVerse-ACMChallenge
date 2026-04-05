# Evaluation Environment Setup

This environment is intended to run the full XPlainVerse evaluation stack in one place:

- Qwen 3.5 for complex-explanation extraction and coverage checking
- BERTScore for semantic similarity
- SLE for simple-explanation simplicity scoring

The key requirement is `torch >= 2.6`, because the current SLE checkpoint is distributed as a legacy `.bin` file and recent `transformers` versions block loading that checkpoint on older Torch versions for safety reasons.

## Files

- `../env/environment_qwen35_bert_sle.yml`
- `../env/requirements_qwen35_bert_sle.txt`

## Recommended Install

From the repository root:

```bash
cd evaluation

conda env create -f env/environment_qwen35_bert_sle.yml
conda activate xplainverse_eval_full
```

If the environment already exists and you want to rebuild it:

```bash
conda remove -n xplainverse_eval_full --all -y
cd evaluation
conda env create -f env/environment_qwen35_bert_sle.yml
conda activate xplainverse_eval_full
```

## What Is Pinned

- Python `3.10.20`
- PyTorch `2.6.0+cu124`
- torchvision `0.21.0+cu124`
- torchaudio `2.6.0+cu124`
- `transformers` pinned to the Git commit used for Qwen 3.5 support
- `accelerate==1.12.0`
- `bert-score==0.3.13`
- `sentencepiece==0.2.1`
- `safetensors==0.7.0`

## Quick Verification

Run this after activation:

```bash
python - <<'PY'
import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("transformers", transformers.__version__)

cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
print("qwen_config", type(cfg).__name__)

sle_model = AutoModelForSequenceClassification.from_pretrained("liamcripwell/sle-base")
print("sle_model_loaded", sle_model.__class__.__name__)
PY
```

Expected:

- `torch` reports `2.6.0+cu124`
- `cuda_available` is `True` on a GPU node
- `qwen_config` resolves to a Qwen 3.5 config class
- `sle_model_loaded` loads successfully

## Running The Evaluator

From the repository root:

```bash
cd evaluation

python evaluate_val.py \
  --submission ../test_data/predictions.jsonl \
  --ground-truth ../test_data/ground_truth.jsonl \
  --output-dir ../outputs
```

## Notes

- The evaluator defaults are already set for this environment.
- The default device map is `cuda:0`.
- Qwen, BERTScore, and SLE are reused within the same run.
- If your system uses a different CUDA stack, adjust the environment file accordingly.
