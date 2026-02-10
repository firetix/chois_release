#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _resolve_smpl_all_models_dir(data_root: Path) -> Path:
    env = os.environ.get("SMPL_ALL_MODELS_DIR")
    if env:
        return Path(env)

    root = _repo_root()
    candidates = [
        data_root / "smpl_all_models",
        root / "data" / "smpl_all_models",
        root / "smpl_all_models",
    ]
    return _first_existing(candidates) or candidates[0]


def _resolve_smplh_path(smpl_all_models_dir: Path) -> Path:
    env = os.environ.get("SMPLH_PATH")
    if env:
        return Path(env)
    return smpl_all_models_dir / "smplh_amass"


def _check_exists(missing: list[str], path: Path, desc: str) -> None:
    if not path.exists():
        missing.append(f"{desc}: {path}")


def _check_dir_nonempty(missing: list[str], path: Path, desc: str, glob_pat: str | None = None) -> None:
    if not path.exists() or not path.is_dir():
        missing.append(f"{desc}: {path}")
        return
    if glob_pat:
        if not any(path.glob(glob_pat)):
            missing.append(f"{desc} (empty): {path} (expected {glob_pat})")
    else:
        if not any(path.iterdir()):
            missing.append(f"{desc} (empty): {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CHOIS demo prerequisites (data/models) before running scripts.")
    parser.add_argument("--data-root", default="./processed_data", help="processed_data folder (mounted in docker).")
    parser.add_argument("--pretrained-model", default="./pretrained_models/model-10.pt", help="Path to model checkpoint.")
    parser.add_argument("--window", type=int, default=120, help="Window size used by the demo scripts.")
    parser.add_argument("--demo", choices=["single_window", "long_seq"], default="single_window")
    parser.add_argument("--scene-name", default="frl_apartment_4", help="Scene name for long_seq demo.")
    args = parser.parse_args()

    repo_root = _repo_root()
    data_root = Path(args.data_root).resolve()
    pretrained_model = Path(args.pretrained_model).resolve()

    missing: list[str] = []

    _check_exists(missing, data_root, "processed_data folder")
    _check_exists(missing, pretrained_model, "pretrained model checkpoint")

    # Core test dataset inputs.
    _check_exists(missing, data_root / "test_diffusion_manip_seq_joints24.p", "test sequence pickle")
    _check_exists(
        missing,
        data_root / f"cano_test_diffusion_manip_window_{args.window}_joints24.p",
        "canonicalized test windows pickle (avoids expensive preprocessing at runtime)",
    )
    _check_exists(
        missing,
        data_root / f"cano_min_max_mean_std_data_window_{args.window}_joints24.p",
        "min/max/mean/std stats pickle (required for test)",
    )

    # Contact labels and guidance data required by the provided scripts.
    _check_dir_nonempty(
        missing,
        data_root / "contact_labels_w_semantics_npy_files",
        "semantic contact labels folder",
        glob_pat="*.npy",
    )
    _check_dir_nonempty(
        missing,
        data_root / "rest_object_sdf_256_npy_files",
        "rest object SDF folder (guidance)",
        glob_pat="*.npy",
    )

    # Meshes + text annotations are used during sampling/vis.
    _check_dir_nonempty(missing, data_root / "captured_objects", "captured_objects folder")
    _check_dir_nonempty(missing, data_root / "omomo_text_anno_json_data", "text annotations folder", glob_pat="*.json")

    # Rest pose meshes are used for object reconstruction and for generating missing BPS caches on-demand.
    _check_dir_nonempty(
        missing,
        data_root / "rest_object_geo",
        "rest object geometry folder",
        glob_pat="*.ply",
    )

    # Blender assets for video rendering.
    _check_exists(missing, data_root / "blender_files" / "floor_colorful_mat.blend", "Blender floor scene .blend")

    # SMPL models.
    smpl_all_models_dir = _resolve_smpl_all_models_dir(data_root)
    smplh_path = _resolve_smplh_path(smpl_all_models_dir)
    _check_exists(missing, smpl_all_models_dir, "SMPL models root (smpl_all_models)")
    _check_exists(missing, smpl_all_models_dir / "smplx" / "SMPLX_MALE.npz", "SMPL-X male model")
    _check_exists(missing, smpl_all_models_dir / "smplx" / "SMPLX_FEMALE.npz", "SMPL-X female model")
    if args.demo == "long_seq":
        _check_exists(missing, smpl_all_models_dir / "smplx" / "SMPLX_NEUTRAL.npz", "SMPL-X neutral model")
    _check_exists(missing, smplh_path / "male" / "model.npz", "SMPL-H male model (kintree_table)")

    # Long-seq additional assets.
    if args.demo == "long_seq":
        _check_exists(
            missing,
            data_root / "replica_processed" / "scene_floor_height.json",
            "replica scene floor height json",
        )
        _check_exists(
            missing,
            data_root
            / "replica_processed"
            / "replica_fixed_poisson_sdfs_res256"
            / f"{args.scene_name}_sdf.npy",
            "scene SDF npy",
        )
        _check_exists(
            missing,
            data_root
            / "replica_processed"
            / "replica_fixed_poisson_sdfs_res256"
            / f"{args.scene_name}_sdf_info.json",
            "scene SDF info json",
        )
        _check_dir_nonempty(
            missing,
            data_root / "replica_processed" / "replica_single_object_long_seq_data_selected",
            "planned long sequence waypoint npys folder",
            glob_pat="**/*.npy",
        )
        _check_exists(
            missing,
            repo_root / "utils" / "create_eval_dataset" / "eval_dataset_response_text_idx.json",
            "eval dataset response text idx json (repo file)",
        )
        _check_exists(
            missing,
            repo_root / "utils" / "create_eval_dataset" / "selected_long_seq_names.json",
            "selected long seq names json (repo file)",
        )

    if missing:
        print("CHOIS demo prerequisites: FAIL")
        print(f"Repo: {repo_root}")
        print(f"data_root: {data_root}")
        print(f"pretrained_model: {pretrained_model}")
        print(f"smpl_all_models_dir: {smpl_all_models_dir}")
        print(f"smplh_path: {smplh_path}")
        print("Missing:")
        for item in missing:
            print(f"- {item}")
        print("")
        print("If you are using docker-compose, set these environment variables if your assets are elsewhere:")
        print("- PROCESSED_DATA_HOST=/abs/path/to/processed_data")
        print("- PRETRAINED_MODELS_HOST=/abs/path/to/pretrained_models")
        print("- SMPL_ALL_MODELS_HOST=/abs/path/to/smpl_all_models")
        return 2

    print("CHOIS demo prerequisites: PASS")
    print(f"Repo: {repo_root}")
    print(f"data_root: {data_root}")
    print(f"pretrained_model: {pretrained_model}")
    print(f"smpl_all_models_dir: {smpl_all_models_dir}")
    print(f"smplh_path: {smplh_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
