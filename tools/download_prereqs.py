#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

DEFAULT_PROCESSED_DATA_URL = "https://drive.google.com/file/d/1ZG-9--RfUWj5oWYnvcONNuRuxaH_Zpw1/view?usp=sharing"
DEFAULT_PRETRAINED_MODELS_URL = "https://drive.google.com/drive/folders/1gqw3EHiEMqw1OXqH92Axoc5FJntA_E5x?usp=sharing"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_dir_empty(p: Path) -> bool:
    return not any(p.iterdir())


def _ensure_gdown():
    try:
        import gdown  # type: ignore

        return gdown
    except Exception:
        print("[INFO] Installing gdown (needed for Google Drive downloads)...", file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "gdown"])
        import gdown  # type: ignore

        return gdown


def _detect_archive_type(path: Path) -> str | None:
    """
    Return one of: zip, tar, tar.gz, tar.bz2, tar.xz, or None.
    """
    with path.open("rb") as f:
        head = f.read(8)

    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06") or head.startswith(b"PK\x07\x08"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "tar.gz"
    if head.startswith(b"BZh"):
        return "tar.bz2"
    if head.startswith(b"\xfd7zXZ\0"):
        return "tar.xz"

    # tar magic "ustar" appears at offset 257
    try:
        with path.open("rb") as f:
            f.seek(257)
            if f.read(5) == b"ustar":
                return "tar"
    except Exception:
        pass

    return None


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    dest_dir_resolved = dest_dir.resolve()

    def _is_within_directory(directory: Path, target: Path) -> bool:
        try:
            directory = directory.resolve()
            target = target.resolve()
        except Exception:
            return False
        return str(target).startswith(str(directory))

    for member in tar.getmembers():
        member_path = dest_dir / member.name
        if not _is_within_directory(dest_dir_resolved, member_path):
            raise RuntimeError(f"Blocked path traversal in tar member: {member.name}")

    tar.extractall(path=dest_dir)


def _safe_extract_zip(z: zipfile.ZipFile, dest_dir: Path) -> None:
    dest_dir_resolved = dest_dir.resolve()

    for name in z.namelist():
        member_path = (dest_dir / name).resolve()
        if not str(member_path).startswith(str(dest_dir_resolved)):
            raise RuntimeError(f"Blocked path traversal in zip member: {name}")

    z.extractall(dest_dir)


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    kind = _detect_archive_type(archive_path)
    if kind is None:
        raise RuntimeError(f"Unrecognized archive type for: {archive_path}")

    if kind == "zip":
        with zipfile.ZipFile(archive_path) as z:
            _safe_extract_zip(z, dest_dir)
        return

    mode = {
        "tar": "r:",
        "tar.gz": "r:gz",
        "tar.bz2": "r:bz2",
        "tar.xz": "r:xz",
    }[kind]

    with tarfile.open(archive_path, mode) as t:
        _safe_extract_tar(t, dest_dir)


def _find_processed_data_root(extracted_dir: Path) -> Path:
    """
    Try to locate the processed_data directory inside an extracted archive.
    """
    if (extracted_dir / "processed_data").is_dir():
        return extracted_dir / "processed_data"

    # The file below is expected at processed_data root for the provided demos.
    hits = list(extracted_dir.rglob("test_diffusion_manip_seq_joints24.p"))
    if hits:
        return hits[0].parent

    # Fallback: if there's only one directory, assume it's the root.
    dirs = [p for p in extracted_dir.iterdir() if p.is_dir()]
    if len(dirs) == 1:
        return dirs[0]

    raise RuntimeError(
        "Could not find processed_data root inside the downloaded archive. "
        f"Extracted to: {extracted_dir}"
    )


def _install_dir(src_dir: Path, dest_dir: Path) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if dest_dir.exists() and not _is_dir_empty(dest_dir):
        print(f"[WARN] {dest_dir} is not empty; merging (this may take a while).")
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        return

    if dest_dir.exists() and _is_dir_empty(dest_dir):
        dest_dir.rmdir()

    shutil.move(str(src_dir), str(dest_dir))


def _maybe_flatten_single_subdir(dest_dir: Path) -> None:
    """
    If a download tool created a single nested folder under dest_dir, flatten it.
    Example: pretrained_models/pretrained_models/model-10.pt
    """
    if not dest_dir.is_dir():
        return
    entries = [p for p in dest_dir.iterdir() if p.name not in {".DS_Store"}]
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for child in nested.iterdir():
            target = dest_dir / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))
        try:
            nested.rmdir()
        except OSError:
            pass


def download_processed_data(gdown, url: str, repo_root: Path) -> None:
    downloads_dir = repo_root / ".downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = downloads_dir / "chois_processed_data_download"

    print(f"[INFO] Downloading processed_data archive from Google Drive: {url}")
    out = gdown.download(url, str(archive_path), quiet=False, fuzzy=True)
    if not out:
        raise RuntimeError("gdown failed to download processed_data (returned no output path).")

    archive_path = Path(out)
    print(f"[INFO] Downloaded to: {archive_path}")

    with tempfile.TemporaryDirectory(prefix="chois_processed_data_extract_") as td:
        extract_dir = Path(td)
        print(f"[INFO] Extracting archive to: {extract_dir}")
        _extract_archive(archive_path, extract_dir)
        src_processed_data = _find_processed_data_root(extract_dir)
        dest_processed_data = repo_root / "processed_data"
        print(f"[INFO] Installing processed_data to: {dest_processed_data}")
        _install_dir(src_processed_data, dest_processed_data)


def download_pretrained_models(gdown, url: str, repo_root: Path) -> None:
    dest_dir = repo_root / "pretrained_models"
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading pretrained_models folder from Google Drive: {url}")
    # gdown may create a nested folder; we flatten below.
    gdown.download_folder(url, output=str(dest_dir), quiet=False, use_cookies=True)
    _maybe_flatten_single_subdir(dest_dir)

    # Basic sanity check (the provided scripts expect this).
    expected = dest_dir / "model-10.pt"
    if expected.exists():
        print(f"[INFO] Found checkpoint: {expected}")
    else:
        print(f"[WARN] Did not find {expected}. The Drive folder may have a different layout.")
        pt_files = list(dest_dir.rglob("*.pt"))
        if pt_files:
            print("[INFO] Found these .pt files:")
            for p in pt_files[:20]:
                print(f"- {p}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CHOIS demo prerequisites (data + pretrained models).")
    parser.add_argument("--processed-data-url", default=DEFAULT_PROCESSED_DATA_URL)
    parser.add_argument("--pretrained-models-url", default=DEFAULT_PRETRAINED_MODELS_URL)
    parser.add_argument("--repo-root", default=str(_repo_root()), help="Repo root (default: auto-detected).")
    parser.add_argument("--skip-processed-data", action="store_true")
    parser.add_argument("--skip-pretrained-models", action="store_true")
    parser.add_argument("--run-check", action="store_true", help="Run tools/check_demo_prereqs.py at the end.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "trainer_chois.py").exists():
        print(f"[ERROR] {repo_root} does not look like the CHOIS repo root.", file=sys.stderr)
        return 2

    gdown = _ensure_gdown()

    try:
        if not args.skip_processed_data:
            download_processed_data(gdown, args.processed_data_url, repo_root)
        if not args.skip_pretrained_models:
            download_pretrained_models(gdown, args.pretrained_models_url, repo_root)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        return 2

    print("")
    print("[NOTE] SMPL models (SMPL-X and SMPL-H) cannot be auto-downloaded by this script because they require")
    print("       manual acceptance of the license on the SMPL-X website. Place them under:")
    print("       - processed_data/smpl_all_models/smplx/*.npz")
    print("       - processed_data/smpl_all_models/smplh_amass/male/model.npz")

    if args.run_check:
        print("")
        print("[INFO] Running prereq checker (single_window)...")
        subprocess.call([sys.executable, str(repo_root / "tools" / "check_demo_prereqs.py"), "--demo", "single_window"])

    print("")
    print("[DONE] Download step completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

