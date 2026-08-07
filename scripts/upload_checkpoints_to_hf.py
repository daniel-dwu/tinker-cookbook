"""Download Tinker sampler checkpoints and upload them to a Hugging Face repo.

Each checkpoint archive is downloaded via a signed URL, extracted, and uploaded
to hf.co/{HF_REPO} under {run_name}/{checkpoint_name}/.
"""

import json
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import tinker
from huggingface_hub import HfApi

HF_REPO = "danieldwu/tinker-checkpoints"

SDF_LOGS = Path(
    "/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/tinker_cookbook/recipes/"
    "reward_hacking/SDF_Comparison/logs/cubic_gravity"
)
COOKBOOK_LOGS = Path("/Users/danielwu/Documents/Coding/SPAR/tinker-cookbook/logs")

# (run label used as HF folder, path to checkpoints.jsonl, checkpoint names or "LAST")
TARGETS: list[tuple[str, Path, list[str]]] = [
    (
        "cubic_gravity_sdf_c4_doctag",
        SDF_LOGS / "sdf_c4_doctag" / "checkpoints.jsonl",
        ["000050", "000100", "000200", "000500", "LAST"],
    ),
    (
        "cubic_gravity_user_sft_wildchat",
        SDF_LOGS / "user_sft_wildchat" / "checkpoints.jsonl",
        ["000050", "000100", "000200", "000500", "LAST"],
    ),
    (
        "428run_no_2",
        COOKBOOK_LOGS / "428run" / "no" / "2" / "checkpoints.jsonl",
        ["LAST"],
    ),
]


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_checkpoints(manifest: list[dict], wanted: list[str]) -> list[dict]:
    by_name = {row["name"]: row for row in manifest}
    resolved = []
    for name in wanted:
        row = by_name[manifest[-1]["name"]] if name == "LAST" else by_name[name]
        if row not in resolved:
            resolved.append(row)
    return resolved


def download_and_extract(url: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".archive", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"    downloading archive ...")
        urllib.request.urlretrieve(url, tmp_path)
        size_mb = tmp_path.stat().st_size / 1e6
        print(f"    downloaded {size_mb:.1f} MB, extracting ...")
        if tarfile.is_tarfile(tmp_path):
            with tarfile.open(tmp_path) as tf:
                tf.extractall(dest_dir, filter="data")
        elif zipfile.is_zipfile(tmp_path):
            with zipfile.ZipFile(tmp_path) as zf:
                zf.extractall(dest_dir)
        else:
            # Not an archive; keep raw bytes.
            shutil.move(tmp_path, dest_dir / "checkpoint.bin")
            return
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)

    staging_root = Path(tempfile.mkdtemp(prefix="tinker_hf_upload_"))
    print(f"Staging directory: {staging_root}")

    for run_label, manifest_path, wanted in TARGETS:
        manifest = load_manifest(manifest_path)
        for row in resolve_checkpoints(manifest, wanted):
            ckpt_name = row["name"]
            sampler_path = row["sampler_path"]
            folder = f"{run_label}/{ckpt_name}"
            print(f"[{folder}] {sampler_path}")

            resp = rest_client.get_checkpoint_archive_url_from_tinker_path(
                sampler_path
            ).result()
            dest = staging_root / run_label / ckpt_name
            download_and_extract(resp.url, dest)

            print(f"    uploading to {HF_REPO}/{folder} ...")
            api.upload_folder(
                repo_id=HF_REPO,
                repo_type="model",
                folder_path=str(dest),
                path_in_repo=folder,
                commit_message=f"Add {folder} (from {sampler_path})",
            )
            shutil.rmtree(dest)
            print(f"    done.")

    shutil.rmtree(staging_root, ignore_errors=True)
    print(f"\nAll checkpoints uploaded to https://huggingface.co/{HF_REPO}")


if __name__ == "__main__":
    main()
