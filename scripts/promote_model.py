"""
scripts/promote_model.py
Promote a new model version to production.

Usage:
    python scripts/promote_model.py model_v3.keras
    # Then: git commit -m "promote model_v3" && git push
    # Railway auto-redeploys with the new model.
"""
import sys
import os
from io import BytesIO

def promote(version: str):
    from huggingface_hub import HfApi
    repo_id = os.environ.get("HF_REPO_ID", "YOUR_USERNAME/esl-model")
    token   = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable first.")
        sys.exit(1)

    local_path = f"artifacts/{version}"
    if not os.path.exists(local_path):
        print(f"ERROR: {local_path} not found locally.")
        sys.exit(1)

    api = HfApi()
    print(f"Uploading {version} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=version,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Updating current.txt -> {version}")
    api.upload_file(
        path_or_fileobj=BytesIO(version.encode()),
        path_in_repo="current.txt",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Done. Production model is now: {version}")
    print("Trigger redeploy: git commit --allow-empty -m 'deploy: promote to {version}' && git push")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/promote_model.py <model_filename>")
        print("Example: python scripts/promote_model.py model_v3.keras")
        sys.exit(1)
    promote(sys.argv[1])