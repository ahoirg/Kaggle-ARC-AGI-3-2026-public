"""Run arc-agi-3-final-solution.ipynb end to end on CPU against a scripted fake model.

usage: python notebook_cpu_test.py NOTEBOOK SOURCE_BUNDLE ENV_FILES WORK_DIR
           [--games lf52,vc33] [--minutes 3] [--mode replay|edge] [--kernel python3]

Development check, not an evaluation. It builds a Kaggle-like input layout under
WORK_DIR, rewrites only the configuration cell (paths, games, budget) and the model
cell (start tools/fake_model/fake_llm.py instead of SGLang), and executes every other
cell unchanged with nbclient. The fake model replays actions recorded in a real run,
so level completions and deaths happen in the real offline engine.
The kernel must have the harness dependencies, arc-agi, nbclient and ipykernel.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]


def _replace_cell(cells, marker, new_source):
    for cell in cells:
        if cell["cell_type"] == "code" and "".join(cell["source"]).lstrip().startswith(marker):
            cell["source"] = new_source
            return
    raise KeyError(marker)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("source_bundle")
    parser.add_argument("env_files")
    parser.add_argument("work_dir")
    parser.add_argument("--games", default="lf52,vc33")
    parser.add_argument("--minutes", type=float, default=3.0)
    parser.add_argument("--mode", default="replay")
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    import nbformat
    from nbclient import NotebookClient

    work = Path(args.work_dir).resolve()
    if work.exists():
        shutil.rmtree(work)
    inputs = work / "input"
    (inputs / "competition" / "arc_agi_3_wheels").mkdir(parents=True)
    (inputs / "competition" / "environment_files").symlink_to(Path(args.env_files).resolve())
    for name in ("wheelhouse", "model", "draft_model"):
        (inputs / name).mkdir()
    output = work / "working"

    notebook = nbformat.read(args.notebook, as_version=4)
    config_cell = next(c for c in notebook.cells if c["cell_type"] == "code"
                       and "# ---------------------------------------------------------------- run configuration"
                       in "".join(c["source"]))
    config = "".join(config_cell["source"])
    paths = {
        "wheelhouse": str(inputs / "wheelhouse"), "model": str(inputs / "model"),
        "draft_model": str(inputs / "draft_model"), "source_bundle": str(Path(args.source_bundle).resolve()),
        "competition": str(inputs / "competition"),
    }
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    config += "\n".join([
        "", "# ---- CPU test overrides",
        f"KAGGLE_INPUTS = {paths!r}",
        f"KAGGLE_OUTPUT_DIR = {str(output)!r}",
        f"KAGGLE_BUNDLE_DIR = {str(work / 'bundle')!r}",
        f"HARNESS_DIR_KAGGLE = {str(HARNESS)!r}",
        f"PILOT_GAME_PREFIXES = {games!r}",
        f"PUBLIC_RUNTIME_MINUTES = {args.minutes!r}",
    ])
    config_cell["source"] = config
    fake = HARNESS / "tools" / "fake_model" / "fake_llm.py"
    _replace_cell(notebook.cells, "# arc3-step: model", "\n".join([
        "# arc3-step: model",
        "import subprocess as _sp",
        f"_fake = _sp.Popen([sys.executable, {str(fake)!r}, '{args.port}'], "
        f"env=dict(os.environ, FAKE_MODE={args.mode!r}, FAKE_LOG=str(WORKING_DIR / 'fake_requests.jsonl')))",
        f"Path({str(work / 'fake.pid')!r}).write_text(str(_fake.pid))",
        "time.sleep(1.5)",
        "CFG = {'fake_model': True}",
        "_manifest['serving_settings'] = dict(CFG)",
        "arc3nb.save_json(WORKING_DIR / 'experiment_manifest.json', _manifest)",
    ]))
    started = time.time()
    client = NotebookClient(notebook, timeout=int(args.minutes * 60 * 4 + 900), kernel_name=args.kernel,
                            resources={"metadata": {"path": str(work)}})
    status = "PASS"
    try:
        client.execute()
    except Exception as error:  # report and keep the executed notebook for inspection
        status = f"FAIL: {type(error).__name__}: {str(error)[-2000:]}"
    finally:
        nbformat.write(notebook, str(work / "executed.ipynb"))
        try:
            import os
            import signal
            os.kill(int((work / "fake.pid").read_text()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
    bundle = sorted(output.glob("ARC3_*_run_bundle.zip"))
    report = {"status": status, "seconds": round(time.time() - started, 1),
              "bundle": str(bundle[0]) if bundle else None,
              "summary": (output / "summary.txt").read_text() if (output / "summary.txt").is_file() else None}
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" and bundle else 1


if __name__ == "__main__":
    sys.exit(main())
