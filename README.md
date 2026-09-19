# MulIF

Official research code for MulIF, built on
[verl](https://github.com/volcengine/verl) for reinforcement-learning
post-training of language models on multi-constraint instruction-following
tasks.

Paper: [Mitigating Exploration Bias in RL for Multi-Instruction Following](https://arxiv.org/pdf/2608.23830)

This repository contains the implementation, a baseline configuration, and
the MulIF training configuration used for comparison. Datasets and models are
hosted separately on Hugging Face.

## Installation

### Docker

```bash
docker pull mianz0122/ifmul:v1
docker run --shm-size=64g --gpus all -it \
  -v "$(pwd)":/workspace -w /workspace mianz0122/ifmul:v1
```

### Local environment

Follow the
[verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html)
for a supported CUDA and PyTorch environment. Python 3.10 or newer is
required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirement-if.txt
```

Install the vLLM or SGLang rollout backend required by your environment.

## Configuration

Create a local environment file:

```bash
cp .env.example .env
```

At minimum, configure:

```text
ROOT_DIR=/path/to/MulIF
DATA_DIR=/path/to/MulIF/verl_data
```

Then load it:

```bash
set -a
source .env
set +a
```

Use `wandb login` for experiment tracking. Do not commit credentials.

## Data and models

`hf_data_download.py` downloads the training dataset **IFTrain** and the main
validation dataset **IFBench** from the
[`billmianz/MulIF`](https://huggingface.co/datasets/billmianz/MulIF)
Hugging Face dataset repository:

```bash
python hf_data_download.py
```

The downloaded files used by the training scripts are:

```text
verl_data/IFTrain_3c_5c.parquet
verl_data/IFBench.parquet
```

Both training scripts use the official
[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) model directly
from Hugging Face. The model is downloaded automatically on first use. You
may change `MODEL_PATH` in both scripts to use another Hugging Face model or
local checkpoint.

## Training

Run commands from the repository root.

Baseline:

```bash
bash recipe/mulif/train_baseline.sh
```

MulIF:

```bash
bash recipe/mulif/train_mulif.sh
```

Both scripts use two GPUs by default. Update `CUDA_VISIBLE_DEVICES` and batch
sizes for your hardware. Logs are written to `log/`; checkpoints and W&B
locations follow the verl configuration and local environment.

## Evaluation

The training scripts evaluate on IFBench at the configured validation
interval. Direct evaluation with the included IFVerify API is documented in
[`README_EVALUATION.md`](README_EVALUATION.md).

## Acknowledgements and license

MulIF is based on
[verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl).
The repository is distributed under the Apache License 2.0. Substantial
framework code remains copyright ByteDance Ltd. and the verl contributors.
See [`LICENSE`](LICENSE) and [`Notice.txt`](Notice.txt).

If you find MulIF useful, please cite:

```bibtex
@article{zhang2026mitigating,
  title={Mitigating Exploration Bias in RL for Multi-Instruction Following},
  author={Zhang, Mian and Yin, Yueqin and He, Kaiyu and Wu, Peilin and Zhang, Xinlu and Zhou, Mingyuan and Chen, Zhiyu Zoey},
  journal={arXiv preprint arXiv:2608.23830},
  year={2026}
}
```
