# Ground Truth Folders

`evaluate_val.py` can load references and precomputed ground-truth entity/fact caches from these folders.

Expected files in each split folder:

- `reference.jsonl`
- `complex_ground_truth_entity_facts.jsonl`

Available local splits:

- `val/`: validation references and validation GT entity/fact cache.
- `test/`: final-test references and final-test GT entity/fact cache for local/private evaluation.

Use a split by name:

```bash
python ../evaluate_val.py \
  --submission /path/to/submission.zip \
  --ground-truth-split val \
  --output-dir /path/to/results
```

Or pass a folder explicitly:

```bash
python ../evaluate_val.py \
  --submission /path/to/submission.zip \
  --ground-truth-dir /path/to/ground_truth_folder \
  --output-dir /path/to/results
```

You can override only the cache with:

```bash
python ../evaluate_val.py \
  --submission /path/to/submission.zip \
  --ground-truth-split test \
  --gt-entity-facts /path/to/complex_ground_truth_entity_facts.jsonl \
  --output-dir /path/to/results
```

If a cache is present, the evaluator uses it and skips GT entity/fact extraction for matching rows. If rows are missing from the cache and Qwen/vLLM stages are enabled, the evaluator computes the missing GT rows and updates the cache unless `--no-update-gt-entity-facts` is set.
