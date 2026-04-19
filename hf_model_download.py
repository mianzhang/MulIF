import os
from pathlib import Path

import huggingface_hub.file_download as hf_file_download
from huggingface_hub import snapshot_download


def configure_hf_downloads():
    # Azure/FUSE-backed mounts can report zero free space via statvfs even when writes succeed.
    if os.environ.get("HF_SKIP_DISK_CHECK", "1") == "1":
        hf_file_download._check_disk_space = lambda *args, **kwargs: None


configure_hf_downloads()

default_storage_root = os.environ.get("AZURE_STORAGE_ROOT")
model_cache_dir = os.environ.get("HF_CACHE_DIR")
if model_cache_dir is None:
    model_cache_dir = str(Path(default_storage_root) / "hf_cache") if default_storage_root else "hf_cache"

model_pool = [
    # 'Qwen/Qwen3-0.6B',
    # 'Qwen/Qwen3-1.7B',
    # 'billmianz/sg_baseline_rule_qwen17b_step125'
    # 'billmianz/IFTrain_baseline_qwen17b_step500'
    # 'billmianz/qwen17b_sft1'
    'billmianz/qwen17b_sft2'
    # 'billmianz/IFTrain_baseline_qwen17b_sft1_step200',
    # 'billmianz/IFTrain_baseline_qwen17b_sft1_step500'
    # 'billmianz/IFTrain_linear_base_qwen17b_sft1_step200'
    # 'Qwen/Qwen3-4B',
    # 'Qwen/Qwen3-8B'
    ]

for repo_id in model_pool:
    local_dir = os.path.join(model_cache_dir, repo_id)
    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"[hf_model_download] Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id,
        local_dir=local_dir,
    )
    # Verify download
    if Path(local_dir).exists():
        files = list(Path(local_dir).iterdir())
        print(f"[hf_model_download] Done. {len(files)} files in {local_dir}")
    else:
        print(f"[hf_model_download] WARNING: {local_dir} does not exist after download!")
