# Baselines for XPlainVerse ACM Challenge

This document describes how we set up, fine-tuned, and evaluated two baseline models for the XPlainVerse challenge to be held at ACM-Multimedia-2026:

1. **Qwen3-VL-8B-Instruct** (8B parameters)
2. **InternVL3.5-14B** (14B parameters)

Both models were fine-tuned using LoRA and evaluated on the validation set using vLLM-accelerated inference via [ms-swift](https://github.com/modelscope/ms-swift).

### Hardware

All experiments were conducted on **AMD MI210 GPUs (64 GB VRAM each)**:

- **Training:** 4 × AMD MI210 GPUs for both models
- **Inference:** 1 × AMD MI210 GPU for both models

The commands below use `CUDA_VISIBLE_DEVICES` for NVIDIA GPU compatibility. If you are on AMD GPUs with ROCm, replace `CUDA_VISIBLE_DEVICES` with `HIP_VISIBLE_DEVICES`.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Training Data Format](#training-data-format)
- [Data Preparation](#data-preparation)
- [Fine-Tuning](#fine-tuning)
- [LoRA Weight Merging](#lora-weight-merging)
- [Inference with vLLM](#inference-with-vllm)
- [Model Weights](#model-weights)

---

## Environment Setup

We use [ms-swift](https://github.com/modelscope/ms-swift) for both fine-tuning (SFT) and inference. ms-swift provides a unified interface for training and serving various vision-language models with LoRA, full fine-tuning, and vLLM integration.

```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# Install vLLM for accelerated inference
pip install vllm
```

**Requirements:** Python >= 3.10, PyTorch >= 2.0 with CUDA or ROCm support, transformers, accelerate, peft.

---

## Training Data Format

The training data is in JSONL (JSON Lines) format, compatible with ms-swift's SFT pipeline. Each line is a JSON object:

```json
{
    "id": "fake__a73db74c266e9a0574c5f70b",
    "messages": [
        {
            "role": "user",
            "content": "<image>\nDetect whether the image is real or fake and provide reasoning for it.\n\nRespond in the following format:\n<reasoning>your reasoning here</reasoning>\n<answer>real or fake</answer>"
        },
        {
            "role": "assistant",
            "content": "<reasoning>The image looks fake due to several visual inconsistencies...</reasoning>\n<answer>fake</answer>"
        }
    ],
    "images": [
        "/path/to/image.png"
    ]
}
```

| Field | Description |
|-------|-------------|
| `id` | Unique identifier: `{label}__{image_hash}` |
| `messages` | Conversation with a `user` prompt (containing `<image>` placeholder) and an `assistant` response |
| `images` | List of absolute paths to the referenced image(s) |

The assistant response uses structured XML-like tags: `<reasoning>...</reasoning>` for the explanation and `<answer>real or fake</answer>` for the classification.

---

## Data Preparation

The full training set from the [XPlainVerse dataset](https://huggingface.co/datasets/Abhijeet8901/XPlainVerse) contains **450,000 samples** (130,000 real + 320,000 fake).

Since the dataset is heavily imbalanced toward fake samples, we **randomly sampled 130,000 fake images** (equal to the number of real images) to create a balanced training set of **260,000 samples** (130,000 real + 130,000 fake).

---

## Fine-Tuning

Both models were fine-tuned using **LoRA (Low-Rank Adaptation)** with ms-swift on 4 GPUs with identical hyperparameters:

| Hyperparameter | Value |
|----------------|-------|
| Training type | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | `all-linear` |
| Freeze LLM | `false` |
| Freeze ViT | `true` |
| Epochs | 1 |
| Per-device batch size | 8 |
| Gradient accumulation steps | 2 |
| Effective batch size | 8 × 4 GPUs × 2 = **64** |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Max sequence length | 2048 |
| Precision | bf16 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight decay | 0.1 |
| Max gradient norm | 1.0 |
| Seed | 42 |

Training produced **4,063 steps** for both models (260,000 samples / effective batch size 64).

### Qwen3-VL-8B

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --use_hf true \
    --dataset final_training_with_perturbations_equal.jsonl \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit true \
    --gradient_accumulation_steps 2 \
    --max_length 2048 \
    --max_new_tokens 2048 \
    --warmup_ratio 0.05 \
    --save_strategy steps \
    --save_steps 400 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 4 \
    --bf16 true \
    --output_dir ./output/qwen3-vl-8b_xplainverse
```
---

## LoRA Weight Merging

After training, the LoRA adapter weights are merged into the base model to create a standalone model for efficient inference with vLLM.

```bash
# Qwen3-VL-8B
swift export \
    --adapters ./output/qwen3-vl-8b_xplainverse/checkpoint-4063 \
    --merge_lora true \
    --output_dir ./Qwen3-VL-8B-XPlainVerse

```

---

## Inference with vLLM

After merging, we use vLLM as the inference backend through ms-swift for fast batch inference with continuous batching and PagedAttention.

| Hyperparameter | Value |
|----------------|-------|
| Inference backend | vLLM |
| Max new tokens | 2048 |
| Temperature | 0.0 (greedy) |
| Precision | bf16 |
| GPU memory utilization | 0.9 |
| Max model length | 4096 |
| Max batch size | 16 |

```bash
# Qwen3-VL-8B
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model ./Qwen3-VL-8B-XPlainVerse \
    --val_dataset val_data.jsonl \
    --infer_backend vllm \
    --max_new_tokens 2048 \
    --temperature 0.0 \
    --torch_dtype bfloat16 \
    --stream false \
    --use_hf true \
    --gpu_memory_utilization 0.9 \
    --max_model_len 4096 \
    --max_batch_size 16 \
    --result_path eval_qwen_output.jsonl

```

---

## Model Weights

The fine-tuned model weights (LoRA merged) are available on Hugging Face:

| Model | Hugging Face Link |
|-------|-------------------|
| Qwen3-VL-8B-XPlainVerse | [kartik060702/Qwen3-VL-8B-XPlainVerse](https://huggingface.co/kartik060702/Qwen3-VL-8B-XPlainVerse) |
| InternVL3.5-14B-XPlainVerse | [kartik060702/InternVL3_5-14B-XPlainVerse](https://huggingface.co/kartik060702/InternVL3_5-14B-XPlainVerse) |

```bash
# Download models
huggingface-cli download kartik060702/Qwen3-VL-8B-XPlainVerse --local-dir ./Qwen3-VL-8B-XPlainVerse
huggingface-cli download kartik060702/InternVL3_5-14B-XPlainVerse --local-dir ./InternVL3_5-14B-XPlainVerse
```

## Results



