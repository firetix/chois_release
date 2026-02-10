#!/usr/bin/env python3
from __future__ import annotations

"""
Run CHOIS demos on Modal GPUs.

Why Modal?
- macOS cannot provide CUDA GPUs to Docker, but Modal can run your code on an NVIDIA GPU.

What this script does
- Builds the same environment from this repo's Dockerfile using `modal.Image.from_dockerfile`.
- Uses a Modal Volume for large assets (`processed_data/`, `pretrained_models/`).
- Runs a smoke test (no Blender rendering) via `--compute_metrics --max_test_seqs=N`.

Setup (one time)
1) Install + authenticate Modal:
   - python3 -m pip install modal
   - modal token new

2) Upload assets from your laptop into a Modal Volume:
   - modal run tools/modal_chois_demo.py::upload_assets

Then run the smoke test:
   - modal run tools/modal_chois_demo.py --max-test-seqs 2

Results are written to the `chois-results` volume under:
  /chois_single_window_results
"""

import os
import subprocess
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parent.parent

# Volume names (override via env if desired)
ASSETS_VOLUME_NAME = os.environ.get("CHOIS_MODAL_ASSETS_VOL", "chois-assets")
RESULTS_VOLUME_NAME = os.environ.get("CHOIS_MODAL_RESULTS_VOL", "chois-results")

# GPU type (A10/A100 recommended for CUDA 11.3 + torch 1.11)
GPU_TYPE = os.environ.get("CHOIS_MODAL_GPU", "A10")

# Resource sizing; increase if you see OOMs.
CPU = float(os.environ.get("CHOIS_MODAL_CPU", "8"))
MEMORY_MIB = int(os.environ.get("CHOIS_MODAL_MEMORY_MIB", "32768"))  # 32 GiB
TIMEOUT_S = int(os.environ.get("CHOIS_MODAL_TIMEOUT_S", str(2 * 60 * 60)))  # 2 hours


image = modal.Image.from_dockerfile(
    path=str(REPO_ROOT / "Dockerfile"),
    context_dir=str(REPO_ROOT),
)

app = modal.App(name="chois-demo", image=image)

assets_vol = modal.Volume.from_name(ASSETS_VOLUME_NAME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)


def _as_dir(p: Path) -> str:
    # Modal's `put_directory` example uses a trailing slash on the local path.
    s = str(p)
    return s if s.endswith("/") else (s + "/")


@app.local_entrypoint()
def upload_assets(
    processed_data_dir: str = str(REPO_ROOT / "processed_data"),
    pretrained_models_dir: str = str(REPO_ROOT / "pretrained_models"),
) -> None:
    """
    Upload `processed_data/` and `pretrained_models/` into the assets volume.

    This can take a while (multi-GB upload).
    """
    processed_data = Path(processed_data_dir).expanduser().resolve()
    pretrained_models = Path(pretrained_models_dir).expanduser().resolve()

    if not processed_data.exists():
        raise SystemExit(f"processed_data_dir not found: {processed_data}")
    if not pretrained_models.exists():
        raise SystemExit(f"pretrained_models_dir not found: {pretrained_models}")

    with assets_vol.batch_upload(force=True) as batch:
        batch.put_directory(_as_dir(processed_data), "/processed_data")
        batch.put_directory(_as_dir(pretrained_models), "/pretrained_models")

    print(f"Uploaded to volume: {ASSETS_VOLUME_NAME}")
    print("Expected remote layout:")
    print("- /processed_data/...")
    print("- /pretrained_models/model-10.pt")


@app.function(
    gpu=GPU_TYPE,
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_S,
    volumes={
        "/data": assets_vol.read_only(),
        "/results": results_vol,
    },
)
def smoke_single_window(max_test_seqs: int = 2) -> str:
    """
    Runs a fast smoke test:
    - uses `--compute_metrics` to skip Blender rendering/mesh export
    - limits to N sequences via `--max_test_seqs`
    """
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")

    # Fail fast with a clear message if assets are not present on the volume.
    subprocess.run(
        [
            "python",
            "tools/check_demo_prereqs.py",
            "--demo",
            "single_window",
            "--data-root",
            "/data/processed_data",
            "--pretrained-model",
            "/data/pretrained_models/model-10.pt",
        ],
        check=True,
        cwd="/workspace",
        env=env,
    )

    cmd = [
        "python",
        "trainer_chois.py",
        "--window=120",
        "--batch_size=32",
        "--data_root_folder=/data/processed_data",
        "--pretrained_model=/data/pretrained_models/model-10.pt",
        "--save_res_folder=/results/chois_single_window_results",
        "--input_first_human_pose",
        "--use_random_frame_bps",
        "--add_language_condition",
        "--use_object_keypoints",
        "--add_semantic_contact_labels",
        "--loss_w_feet=1",
        "--loss_w_fk=0.5",
        "--loss_w_obj_pts=1",
        "--test_sample_res",
        "--use_guidance_in_denoising",
        "--compute_metrics",
        f"--max_test_seqs={max_test_seqs}",
    ]

    subprocess.run(cmd, check=True, cwd="/workspace", env=env)
    results_vol.commit()

    return "/results/chois_single_window_results"


@app.local_entrypoint()
def main(max_test_seqs: int = 2) -> None:
    """
    Entrypoint:
      modal run tools/modal_chois_demo.py --max-test-seqs 2
    """
    out_dir = smoke_single_window.remote(max_test_seqs=max_test_seqs)
    print(f"Smoke test complete. Results in volume '{RESULTS_VOLUME_NAME}' at: {out_dir}")

