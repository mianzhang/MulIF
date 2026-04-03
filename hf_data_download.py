#!/usr/bin/env python3
"""
Script to download all data files from the MulIF Hugging Face repository.
Downloads all files from billmianz/MulIF into verl_data/ directory.
"""

import os
from pathlib import Path

import huggingface_hub.file_download as hf_file_download
from huggingface_hub import snapshot_download


def configure_hf_downloads():
    # Azure/FUSE-backed mounts can report zero free space via statvfs even when writes succeed.
    if os.environ.get("HF_SKIP_DISK_CHECK", "1") == "1":
        hf_file_download._check_disk_space = lambda *args, **kwargs: None


def main():
    configure_hf_downloads()
    repo_id = "billmianz/MulIF"
    default_storage_root = os.environ.get("AZURE_STORAGE_ROOT")
    data_dir = os.environ.get("DATA_DIR")
    if data_dir is None:
        data_dir = str(Path(default_storage_root) / "verl_data") if default_storage_root else "verl_data"
    local_dir = data_dir

    hf_token = os.environ.get("HF_TOKEN")
    
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        token=hf_token,
    )

if __name__ == "__main__":
    exit(main())
