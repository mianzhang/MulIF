import os
import argparse
from huggingface_hub import HfApi

REPO_TYPES = ("model", "dataset")

parser = argparse.ArgumentParser(
    description="Upload a folder to Hugging Face Hub (model or dataset repository)"
)
parser.add_argument("folder_path", type=str, help="Path to the folder to upload")
parser.add_argument(
    "--repo-id",
    type=str,
    default="billmianz/MulIF",
    help="Hugging Face repository ID (default: billmianz/MulIF)",
)
parser.add_argument(
    "--repo-type",
    type=str,
    choices=REPO_TYPES,
    default="dataset",
    help="Hub repo type: 'model' or 'dataset' (default: dataset)",
)
args = parser.parse_args()

token = os.getenv("HF_TOKEN")
api = HfApi(token=token)
api.create_repo(
    repo_id=args.repo_id,
    repo_type=args.repo_type,
    exist_ok=True,
    token=token,
)
api.upload_folder(
    folder_path=args.folder_path,
    repo_id=args.repo_id,
    repo_type=args.repo_type,
)
