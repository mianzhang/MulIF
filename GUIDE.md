

## Docker Setup

#### Step 1: Build the Docker Image

```bash
>>> docker pull mianz0122/ifmul:v1
```

#### Step 2: Run the Container
```bash
>>> docker run --shm-size=64g --gpus all -it -v $(pwd):/workspace -w /workspace mianz0122/ifmul:v1
```

- The `-v $(pwd):/workspace` flag mounts your current directory into the container
- The `--gpus all` flag enables GPU access inside the container
- If encountering `Error while creating shared memory segment`, try to increase `--shm-size` and renew the container.


#### Step 3: Environment Variables
Some environment variables are needed for the project. Please create a `.env`:
```
ROOT_DIR=/workspace
HF_CACHE_DIR=/workspace/hf_cache
RAY_LOG_DIR=/workspace/tmp
TMPDIR=/workspace/tmp
TEMP=/workspace/tmp
TMP=/workspace/tmp
CUDA_DEVICE_ORDER=PCI_BUS_ID
RAY_DEBUG=legacy
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_BASE_URL=
```
Then run `export $(grep -v '^#' .env | xargs)` to export them. 

#### Step 4: Login in to Huggingface and Wandb
```
>>> hf auth login
>>> wandb login
```

## Preparation
#### Step 1: Data 
Run `python hf_data_download.py` to download all the training and eval data into `verl_data/`.
#### Step 2: Base Models
Run `python hf_model_download.py` to download the base models from huggingface. We only train Qwen3 models at this stage.


## Training
Run three training jobs:

```
sh recipe/mulif_ms/rebuttal_SaR_qwen17b_random_sft.sh
sh recipe/mulif_ms/rebuttal_SaR_qwen17b_bestN_sft.sh
sh recipe/mulif_ms/rebuttal_SaR_qwen7b.sh
```
If OOM is encountered, considering decrease the value of `ppo_micro_batch_size` to 16.

- The training log is saved to `log/`
- The checkpoints are saved to `checkpoints/`

## Delivery
run `sh delivery.sh` to upload 12 checkpoints.


<!-- #### Step 2: Convert FSDP checkpoints to Huggingface format -->
<!-- (Skip, I will do this on my end) -->

<!-- #### Step 3: Upload the Trained Models. -->
<!-- Run `python hf_upload.py checkpoints` to upload the trained models. -->
