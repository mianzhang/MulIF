# Evaluating instruction following

MulIF uses the IFVerify implementation under
`verl/utils/reward_score/ifverify/`. It supports strict, per-instruction
verification for IFBench-style rules, RECAST constraints, and optional
LLM-rubric constraints.

## Evaluation during training

The MulIF launchers pass evaluation parquet files through verl:

```text
data.val_files="[$DATA_DIR/IFBench.parquet]"
trainer.val_before_train=True
trainer.test_freq=25
reward_model.reward_manager=ifverify_sg_score
```

Depending on the recipe, validation may also include `RECAST_c5.parquet`,
`RECAST_c10.parquet`, or `AdvancedIF.parquet`.

The reward manager reports:

- `instruction_acc`: fraction of constraints passed by each response.
- `prompt_acc`: fraction of rollouts satisfying every prompt constraint.
- `icr`: fraction of instructions passed by at least one rollout.
- `via`: variance of nonzero per-instruction pass rates.
- `vsa`: variance of pairwise instruction success association.
- `inst_mixed_portion`: fraction of instructions with pass rate strictly
  between zero and one.

## Evaluate responses directly

Install the verifier dependencies first:

```bash
pip install -r requirement-if.txt
```

Use `compute_score_batch` for the same strict scoring path used by training:

```python
from verl.utils.reward_score.ifverify import compute_score_batch

ground_truth = {
    "prompt": "Write at least two sentences.",
    "instructions": [
        {
            "instruction_id": "length_constraints:number_sentences",
            "args": {"num_sentences": 2, "relation": "at least"},
            "mode": "rule",
        }
    ],
}

results = compute_score_batch(
    [
        ("This is sentence one. This is sentence two.", ground_truth),
        ("This response has only one sentence.", ground_truth),
    ],
    strict=True,
    return_verl_reward=False,
)

for result in results:
    print(result["accuracy"], result["follow_instruction_list"])
```

Each instruction dictionary must contain:

- `instruction_id`: a key registered in
  `verl/utils/reward_score/ifverify/instructions_registry.py`.
- `args`: keyword arguments expected by that verifier.
- `mode`: usually `rule`; use the mode required by the dataset.

The returned record includes the overall score, one Boolean per instruction,
the number of followed instructions, and whether all instructions passed.
Batch concurrency is controlled by `REWARDS_SCORE_MAX_WORKERS` (default: 32).

## Rubric-based checks

Rule-based checks run locally. Instructions whose IDs begin with `rubric:`
use an OpenAI-compatible model:

```bash
export OPENAI_API_KEY=...
export OPENAI_RUBRIC_MODEL=...
```

Azure OpenAI configuration is also supported through
`AZURE_OPENAI_BASE_URL` and `AZURE_OPENAI_API_KEY`. Keep credentials in your
shell or secret manager, never in committed files.

## Scope of this release

This repository does not ship the former `evaluate_if.py` or
`evaluate_logicif.py` command-line wrappers. It also does not include full
benchmark prompt JSONL files; released training and validation parquet files
are downloaded from
[`billmianz/MulIF`](https://huggingface.co/datasets/billmianz/MulIF) with:

```bash
python hf_data_download.py
```
