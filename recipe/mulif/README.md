# Training scripts

This directory contains the two scripts needed to compare the baseline with
MulIF:

- `train_baseline.sh`: baseline training configuration.
- `train_mulif.sh`: MulIF training configuration.

Both scripts use the same model, dataset, rollout settings, and two-GPU setup.
Run them from the repository root after configuring `.env`:

```bash
bash recipe/mulif/train_baseline.sh
bash recipe/mulif/train_mulif.sh
```

Adjust `CUDA_VISIBLE_DEVICES`, batch sizes, and `MODEL_PATH` for your
environment.
