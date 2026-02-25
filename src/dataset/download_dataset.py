import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_files, login

DEFAULT_REPO_ID = "pixparse/cc12m-wds"


def download_cc12m(repo_id, data_dir, max_shards):
    shards_dir = os.path.join(data_dir, "cc12m-wds")
    os.makedirs(shards_dir, exist_ok=True)

    all_files = sorted(
        f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".tar")
    )

    print(f"Total shards available: {len(all_files)}")

    if max_shards > 0:
        all_files = all_files[:max_shards]
        print(f"Downloading {len(all_files)} shards")
    else:
        print(f"Downloading ALL {len(all_files)} shards (~175 GB)")

    for i, filename in enumerate(all_files):
        print(f"[{i+1}/{len(all_files)}] Downloading {filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=shards_dir,
            local_dir_use_symlinks=False,
        )

    print("Download complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--project-root", type=str, default=os.environ.get("PROJECT_ROOT"))
    parser.add_argument("--max-shards", type=int, default=5)
    parser.add_argument("--hf-token", type=str, required=True)
    args = parser.parse_args()

    if not args.project_root:
        parser.error("--project-root is required (or set PROJECT_ROOT)")

    project_root = os.path.abspath(os.path.expanduser(args.project_root))
    cache_dir = os.path.join(project_root, ".cache", "huggingface")
    data_dir = os.path.join(project_root, "datasets")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = f"{cache_dir}/datasets"
    os.environ["HF_HUB_CACHE"] = f"{cache_dir}/hub"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    login(token=args.hf_token)

    download_cc12m(args.repo_id, data_dir, args.max_shards)


if __name__ == "__main__":
    main()