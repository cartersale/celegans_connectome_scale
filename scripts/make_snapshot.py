"""Archive the state behind a submission: code, results, figures, environment and manuscript.

The archive holds
  code.tar            `git archive` of HEAD
  results/, figures/  the untracked outputs as they are on disk
  manuscript/         the files passed with --manuscript
  data_manifest.sha256
  environment.txt     Python version and `pip freeze`
  SNAPSHOT.txt        name, date, commit, branch, and whether the tree was clean
  MANIFEST.sha256     SHA-256 of every file above

The working tree must be clean, so that the commit identifies the code exactly.
With --allow-dirty the archive also carries uncommitted.patch (`git diff HEAD`)
and untracked/ (untracked, non-ignored files), and SNAPSHOT.txt says so.

Inputs: the git repository, results/, figures/, data_manifest.sha256, and any
    --manuscript files.
Outputs: snapshots/<name>.tar.gz (snapshots/ is not tracked).
Usage: python scripts/make_snapshot.py <name> [--manuscript FILE ...] [--allow-dirty]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--manuscript", nargs="*", default=[], type=Path)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    status = git("status", "--porcelain")
    if status and not args.allow_dirty:
        sys.exit("working tree is not clean; commit first or pass --allow-dirty\n" + status)
    missing = [str(p) for p in args.manuscript if not p.is_file()]
    if missing:
        sys.exit(f"manuscript files not found: {missing}")

    out = ROOT / "snapshots" / f"{args.name}.tar.gz"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        sys.exit(f"{out} exists; choose another name")

    with tempfile.TemporaryDirectory() as tmp:
        top = Path(tmp) / args.name
        top.mkdir()
        (top / "code.tar").write_bytes(subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"], cwd=ROOT, check=True,
            capture_output=True).stdout)
        for d in ("results", "figures"):
            if (ROOT / d).is_dir():
                shutil.copytree(ROOT / d, top / d, ignore=shutil.ignore_patterns(".DS_Store"))
        if (ROOT / "data_manifest.sha256").exists():
            shutil.copy2(ROOT / "data_manifest.sha256", top)
        if args.manuscript:
            (top / "manuscript").mkdir()
            for p in args.manuscript:
                shutil.copy2(p, top / "manuscript")
        if status:
            (top / "uncommitted.patch").write_text(git("diff", "HEAD"))
            for rel in git("ls-files", "--others", "--exclude-standard").splitlines():
                dst = top / "untracked" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, dst)
        (top / "environment.txt").write_text(
            f"python {sys.version}\n\n" + subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
        (top / "SNAPSHOT.txt").write_text(
            f"name: {args.name}\n"
            f"created: {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"commit: {git('rev-parse', 'HEAD').strip()}\n"
            f"branch: {git('rev-parse', '--abbrev-ref', 'HEAD').strip()}\n"
            f"working tree: {'clean' if not status else 'DIRTY (see uncommitted.patch, untracked/)'}\n"
            f"manuscript files: {', '.join(p.name for p in args.manuscript) or 'none'}\n")
        files = sorted(p for p in top.rglob("*") if p.is_file())
        (top / "MANIFEST.sha256").write_text(
            "".join(f"{sha256(p)}  {p.relative_to(top)}\n" for p in files))
        with tarfile.open(out, "w:gz") as tar:
            tar.add(top, arcname=args.name)

    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files) + 1} files)")
    print(f"sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
