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
   - (Recommended) smoke-test subset:
       modal run tools/modal_chois_demo.py::upload_assets_smoke_single_window --max-test-seqs 2
   - (Full) everything under processed_data/ (multi-GB, slow):
       modal run tools/modal_chois_demo.py::upload_assets

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

@app.local_entrypoint()
def upload_assets_smoke_single_window(
    max_test_seqs: int = 2,
    window: int = 120,
    processed_data_dir: str = str(REPO_ROOT / "processed_data"),
    pretrained_models_dir: str = str(REPO_ROOT / "pretrained_models"),
) -> None:
    """
    Upload a smaller subset of assets sufficient to run the `single_window` smoke test for the
    first `max_test_seqs` validation sequences.

    This avoids uploading the entire multi-GB `processed_data/` folder (which can time out).
    """
    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "joblib is required locally to compute the minimal upload set. "
            "Install it with: python3 -m pip install -U joblib"
        ) from e

    processed_data = Path(processed_data_dir).expanduser().resolve()
    pretrained_models = Path(pretrained_models_dir).expanduser().resolve()

    cano_test_pkl = processed_data / f"cano_test_diffusion_manip_window_{window}_joints24.p"
    if not cano_test_pkl.exists():
        raise SystemExit(f"Missing required pickle: {cano_test_pkl}")

    # Mirror CanoObjectTrajDataset filtering used by the demo:
    # - only windows starting at t=0
    # - only sequences with text annotations present
    # - only windows with length >= window
    window_data = joblib.load(cano_test_pkl)
    seq_names: list[str] = []
    objects: list[str] = []
    kept = 0
    for _, w in window_data.items():
        if int(w.get("start_t_idx", -1)) != 0:
            continue
        motion = w.get("motion", None)
        if motion is None or getattr(motion, "shape", (0,))[0] < window:
            continue
        seq_name = w.get("seq_name", "")
        if not seq_name:
            continue
        text_json = processed_data / "omomo_text_anno_json_data" / f"{seq_name}.json"
        if not text_json.exists():
            continue
        contact_npy = processed_data / "contact_labels_w_semantics_npy_files" / f"{seq_name}.npy"
        if not contact_npy.exists():
            continue

        seq_names.append(seq_name)
        objects.append(seq_name.split("_")[1])
        kept += 1
        if kept >= max_test_seqs:
            break

    if not objects:
        raise SystemExit(
            "Could not determine object names for the smoke test. "
            "Check that `omomo_text_anno_json_data/` exists and contains per-sequence JSON files."
        )

    unique_objects = sorted(set(objects))
    print(f"Smoke upload target objects ({len(unique_objects)}): {', '.join(unique_objects)}")
    print(f"Smoke upload target sequences ({len(seq_names)}): {', '.join(seq_names)}")

    # Base processed_data files/dirs required by `tools/check_demo_prereqs.py` + trainer in smoke mode.
    required_files = [
        "test_diffusion_manip_seq_joints24.p",
        f"cano_test_diffusion_manip_window_{window}_joints24.p",
        f"cano_min_max_mean_std_data_window_{window}_joints24.p",
    ]

    for rel in required_files:
        p = processed_data / rel
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    # Required SMPL models.
    smpl_all_models = processed_data / "smpl_all_models"
    smpl_required = [
        smpl_all_models / "smplx" / "SMPLX_MALE.npz",
        smpl_all_models / "smplx" / "SMPLX_FEMALE.npz",
        smpl_all_models / "smplh_amass" / "male" / "model.npz",
    ]
    for p in smpl_required:
        if not p.exists():
            raise SystemExit(f"Missing required SMPL model file: {p}")

    # Required Blender floor scene file (checked by prereq checker; not used in --compute_metrics mode).
    floor_blend = processed_data / "blender_files" / "floor_colorful_mat.blend"
    if not floor_blend.exists():
        raise SystemExit(f"Missing required Blender scene: {floor_blend}")

    # Required per-object assets for guidance and object reconstruction.
    sdf_dir = processed_data / "rest_object_sdf_256_npy_files"
    rest_geo_dir = processed_data / "rest_object_geo"
    captured_objects_dir = processed_data / "captured_objects"

    sdf_files: list[Path] = []
    rest_geo_files: list[Path] = []
    captured_obj_files: list[Path] = []
    for obj in unique_objects:
        for suffix in [".ply.npy", ".ply.json"]:
            p = sdf_dir / f"{obj}{suffix}"
            if not p.exists():
                raise SystemExit(f"Missing required object SDF file: {p}")
            sdf_files.append(p)
        for suffix in [".ply", ".npy", ".json"]:
            p = rest_geo_dir / f"{obj}{suffix}"
            if not p.exists():
                raise SystemExit(f"Missing required rest object geometry file: {p}")
            rest_geo_files.append(p)
        p = captured_objects_dir / f"{obj}_cleaned_simplified.obj"
        if not p.exists():
            raise SystemExit(f"Missing required captured object mesh file: {p}")
        captured_obj_files.append(p)

    ckpt = pretrained_models / "model-10.pt"
    if not ckpt.exists():
        raise SystemExit(f"Missing required checkpoint: {ckpt}")

    per_seq_contact_files: list[Path] = []
    per_seq_text_files: list[Path] = []
    for seq_name in seq_names:
        contact_npy = processed_data / "contact_labels_w_semantics_npy_files" / f"{seq_name}.npy"
        if not contact_npy.exists():
            raise SystemExit(f"Missing required contact labels file: {contact_npy}")
        per_seq_contact_files.append(contact_npy)

        text_json = processed_data / "omomo_text_anno_json_data" / f"{seq_name}.json"
        if not text_json.exists():
            raise SystemExit(f"Missing required text annotation file: {text_json}")
        per_seq_text_files.append(text_json)

    with assets_vol.batch_upload(force=True) as batch:
        # processed_data/ base
        for rel in required_files:
            batch.put_file(str(processed_data / rel), f"/processed_data/{rel}")

        # Blender file
        batch.put_file(
            str(floor_blend),
            "/processed_data/blender_files/floor_colorful_mat.blend",
        )

        # Per-sequence assets needed for the selected smoke sequences.
        for p in per_seq_contact_files:
            batch.put_file(str(p), f"/processed_data/contact_labels_w_semantics_npy_files/{p.name}")
        for p in per_seq_text_files:
            batch.put_file(str(p), f"/processed_data/omomo_text_anno_json_data/{p.name}")

        # Per-object assets needed for the selected smoke sequences.
        for p in captured_obj_files:
            batch.put_file(str(p), f"/processed_data/captured_objects/{p.name}")
        for p in rest_geo_files:
            batch.put_file(str(p), f"/processed_data/rest_object_geo/{p.name}")

        # Only the SDF npy+json files needed for this smoke test (avoid uploading *.obj).
        for p in sdf_files:
            batch.put_file(str(p), f"/processed_data/rest_object_sdf_256_npy_files/{p.name}")

        # SMPL models (minimal subset)
        batch.put_file(
            str(smpl_all_models / "smplx" / "SMPLX_MALE.npz"),
            "/processed_data/smpl_all_models/smplx/SMPLX_MALE.npz",
        )
        batch.put_file(
            str(smpl_all_models / "smplx" / "SMPLX_FEMALE.npz"),
            "/processed_data/smpl_all_models/smplx/SMPLX_FEMALE.npz",
        )
        batch.put_file(
            str(smpl_all_models / "smplh_amass" / "male" / "model.npz"),
            "/processed_data/smpl_all_models/smplh_amass/male/model.npz",
        )

        # Checkpoint
        batch.put_file(str(ckpt), "/pretrained_models/model-10.pt")

    print(f"Uploaded smoke-test subset to volume: {ASSETS_VOLUME_NAME}")


@app.function(
    gpu=GPU_TYPE,
    cpu=CPU,
    memory=MEMORY_MIB,
    timeout=TIMEOUT_S,
    volumes={
        "/data": assets_vol,
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
    # Override Dockerfile defaults: assets are mounted under /data in Modal.
    env.setdefault("SMPL_ALL_MODELS_DIR", "/data/processed_data/smpl_all_models")
    env.setdefault("SMPLH_PATH", "/data/processed_data/smpl_all_models/smplh_amass")

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
    assets_vol.commit()
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
