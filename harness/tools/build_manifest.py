"""Rebuild HARNESS_MANIFEST.json after editing anything under harness/.

usage: python harness/tools/build_manifest.py [--version VERSION] [--note TEXT]

The manifest lists a CRLF-normalized SHA-256 for every shipped file and one tree
digest over that list. The notebook compares the uploaded copy with it and records
the actual digests in every run, so a run can always be traced to exact code.
Keep the version label unique per uploaded revision (for example 2026.10.12-r2).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebook"))

import arc3nb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="new version label; default keeps the current one")
    parser.add_argument("--note", help="one-line description of this revision")
    args = parser.parse_args()
    path = ROOT / arc3nb.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    files = arc3nb.harness_files(ROOT)
    manifest.update(
        name="arc3-harness",
        version=args.version or manifest.get("version") or datetime.date.today().strftime("%Y.%m.%d") + "-r1",
        built_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        tree_sha256=arc3nb.tree_digest(files),
        file_count=len(files),
        files=files,
    )
    if args.note:
        manifest["note"] = args.note
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{path.name}: version {manifest['version']}, {len(files)} files, tree {manifest['tree_sha256'][:12]}")


if __name__ == "__main__":
    main()
