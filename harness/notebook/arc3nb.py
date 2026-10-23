"""Notebook support for arc-agi-3-final-solution.ipynb.

Runs inside the harness interpreter: the Kaggle kernel, or the Python 3.12 harness
process that the Colab adapter starts. Nothing here talks to the model or changes
how the agent plays; it installs the harness, records provenance, builds the game
list and packages the run outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
import zipfile
from pathlib import Path

MANIFEST_NAME = "HARNESS_MANIFEST.json"
HARNESS_TREE = "ARC3-Inference"


# --------------------------------------------------------------------------- harness

def normalized_digest(path: Path) -> str:
    """SHA-256 of a file with CRLF folded to LF, so a Windows checkout hashes the same."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def harness_files(root: Path) -> dict[str, str]:
    """Normalized digests of every shipped file under the harness root."""
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or "__pycache__" in path.parts or relative == MANIFEST_NAME:
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name in {".DS_Store", "desktop.ini", "Thumbs.db", "dataset-metadata.json"}:
            continue
        files[relative] = normalized_digest(path)
    return files


def tree_digest(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_harness(harness_dir) -> dict:
    """Compare the harness directory with its manifest; report rather than refuse.

    A mismatch usually means a file was edited without rebuilding the manifest
    (tools/build_manifest.py). The run still proceeds, and the actual digests are
    what the run manifest records, so provenance stays exact either way.
    """
    harness_dir = Path(harness_dir)
    manifest = json.loads((harness_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    actual = harness_files(harness_dir)
    expected = manifest.get("files", {})
    changed = sorted(k for k in expected if k in actual and actual[k] != expected[k])
    missing = sorted(k for k in expected if k not in actual)
    extra = sorted(k for k in actual if k not in expected)
    info = {
        "name": manifest.get("name"), "version": manifest.get("version"),
        "manifest_tree_sha256": manifest.get("tree_sha256"), "tree_sha256": tree_digest(actual),
        "matches_manifest": not (changed or missing or extra),
        "changed": changed, "missing": missing, "extra": extra, "path": str(harness_dir),
    }
    if missing:
        raise RuntimeError(f"Harness is incomplete; missing files: {missing[:10]}")
    if not info["matches_manifest"]:
        print("WARNING: harness differs from HARNESS_MANIFEST.json "
              f"(changed={changed[:10]}, extra={extra[:10]}). Rebuild it with tools/build_manifest.py. "
              "The run records the actual file digests.", flush=True)
    return info


def install_harness(source_bundle, harness_dir, bundle_dir) -> dict:
    """Copy the TAAF source bundle and replace its ARC3-Inference tree with ours."""
    source_bundle, harness_dir, bundle_dir = Path(source_bundle), Path(harness_dir), Path(bundle_dir)
    info = verify_harness(harness_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    shutil.copytree(source_bundle, bundle_dir, symlinks=True)
    target = bundle_dir / "src" / HARNESS_TREE
    if not target.is_dir():
        raise RuntimeError(f"Source bundle has no src/{HARNESS_TREE}: {source_bundle}")
    shutil.rmtree(target)
    shutil.copytree(harness_dir / HARNESS_TREE, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    compiled = 0
    for path in sorted(target.rglob("*.py")):
        py_compile.compile(str(path), doraise=True)
        compiled += 1
    info.update(installed_to=str(target), compiled_modules=compiled)
    print(f"Harness {info['version']} installed into {target} ({compiled} modules compiled; "
          f"tree {info['tree_sha256'][:12]}, matches manifest: {info['matches_manifest']}).", flush=True)
    return info


def source_path_entries(bundle_dir) -> list[Path]:
    """Each bundled repo exposes its importable tree at <repo>/src or <repo>."""
    entries = []
    for repo in sorted((Path(bundle_dir) / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries


def expose_sources(bundle_dir) -> Path:
    """Make the bundled repos importable here (sys.path) and in child processes (.pth)."""
    entries = source_path_entries(bundle_dir)
    for entry in entries:
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
    pth_path.write_text("".join(f"{entry}\n" for entry in entries))
    print(f"Source roots: {len(entries)} -> {pth_path}", flush=True)
    return pth_path


def install_arc_runtime(competition_dir) -> None:
    """Install arc-agi from the competition's bundled wheels (offline)."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-warn-conflicts",
         "--disable-pip-version-check", "--find-links", str(Path(competition_dir) / "arc_agi_3_wheels"),
         "arc-agi"],
        stdout=subprocess.DEVNULL,
    )


def install_harness_dependencies(bundle_dir, competition_dir, wheelhouse_dir, allow_pypi: bool) -> None:
    """Install the bundled projects' missing base dependencies (Colab harness process only).

    Serving extras stay out: GPU packages belong to the model server environment.
    """
    import importlib.metadata as metadata
    import tomllib
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    projects = []
    for repo in (Path(bundle_dir) / "src").iterdir():
        descriptor = repo / "pyproject.toml"
        if descriptor.is_file():
            with descriptor.open("rb") as handle:
                data = tomllib.load(handle)
            if isinstance(data.get("project"), dict):
                projects.append((repo, data["project"]))
    local = {canonicalize_name(p["name"]) for _, p in projects if p.get("name")} | {"arc-agi", "arcengine"}
    installed = {canonicalize_name(d.metadata["Name"]) for d in metadata.distributions() if d.metadata.get("Name")}
    gpu = {"torch", "torchvision", "torchaudio", "vllm", "sglang", "triton", "flash-attn", "flashinfer-python"}
    missing = []
    for repo, project in projects:
        locked = {}
        lock = repo / "uv.lock"
        if lock.is_file():
            with lock.open("rb") as handle:
                for package in tomllib.load(handle).get("package", []):
                    if package.get("name") and package.get("version"):
                        locked.setdefault(canonicalize_name(package["name"]), set()).add(package["version"])
        for text in project.get("dependencies", []):
            req = Requirement(text)
            name = canonicalize_name(req.name)
            if req.marker and not req.marker.evaluate():
                continue
            if name == "arc-agi-3-local" or name in local or name in installed:
                continue  # the share bundle excludes re-arc-3; games come from environment_files
            if req.url or name in gpu or name.startswith(("nvidia-", "cuda-", "sgl-")):
                raise RuntimeError("Unexpected non-base harness dependency: " + name)
            versions = locked.get(name, set())
            if len(versions) == 1:
                version = next(iter(versions))
                if req.specifier.contains(version, prereleases=True):
                    extra = "[" + ",".join(sorted(req.extras)) + "]" if req.extras else ""
                    text = req.name + extra + "==" + version
            if text not in missing:
                missing.append(text)
    if not missing:
        print("Bundled harness base dependencies are present.", flush=True)
        return
    args = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
            "--only-binary=:all:", "--find-links", str(Path(competition_dir) / "arc_agi_3_wheels"),
            "--find-links", str(Path(wheelhouse_dir) / "wheels")]
    if not allow_pypi:
        args.append("--no-index")
    print("Installing bundled harness base dependencies:", missing, flush=True)
    subprocess.run(args + missing, check=True, timeout=600)


# --------------------------------------------------------------------------- page cache

def precache(*paths, times=1, delay=0, threads=16) -> None:
    """Read files once so later loads hit the page cache (Kaggle input disks are slow)."""
    from concurrent.futures import ThreadPoolExecutor

    time.sleep(delay)
    print("Precaching: started...", flush=True)
    start = time.time()
    chunk_size = 32 * 1024 * 1024
    chunks, total, count = [], 0, 0
    for index, root in enumerate(paths):
        try:
            files = [root] if os.path.isfile(root) else [
                os.path.join(d, f) for d, _, names in os.walk(root) for f in names]
            for path in files:
                if not os.path.isfile(path):
                    continue
                size = os.path.getsize(path)
                total += size
                count += 1
                for offset in range(0, max(size, 1), chunk_size):
                    chunks.append((index, path, offset))
        except OSError:
            pass
    chunks.sort()
    local = threading.local()

    def read(chunk):
        buffer = getattr(local, "buffer", None)
        if buffer is None:
            buffer = local.buffer = bytearray(chunk_size)
        _, path, offset = chunk
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                handle.readinto(buffer)
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for _ in range(times):
            list(pool.map(read, chunks))
    print(f"Precaching: loaded {count} files ({total / 1e9:.2f} GB) in {time.time() - start:.2f} s.", flush=True)


# --------------------------------------------------------------------------- monitoring

def sysline(paths=("/",)) -> str:
    """One used/total census line. RAM uses MemAvailable, the figure that predicts an OOM."""
    gb = 1024 ** 3
    out, mem, status = [], {}, {}
    try:
        for line in open("/proc/meminfo"):
            key, _, rest = line.partition(":")
            if rest.split():
                mem[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    if mem.get("MemTotal") and mem.get("MemAvailable") is not None:
        out.append(f"RAM {(mem['MemTotal'] - mem['MemAvailable']) / gb:.1f}/{mem['MemTotal'] / gb:.1f}")
    out.append(f"swap {(mem['SwapTotal'] - mem['SwapFree']) / gb:.1f}/{mem['SwapTotal'] / gb:.1f}"
               if mem.get("SwapTotal") else "swap none")
    try:
        for line in open("/proc/self/status"):
            key, _, rest = line.partition(":")
            if key in ("VmRSS", "Threads") and rest.split():
                status[key] = int(rest.split()[0])
    except OSError:
        pass
    proc = []
    if "VmRSS" in status:
        proc.append(f"{status['VmRSS'] * 1024 / gb:.1f} GB")
    proc.append(f"{status.get('Threads', threading.active_count())} threads")
    try:
        proc.append(f"{len(os.listdir('/proc/self/fd'))} fds")
    except OSError:
        pass
    out.append("proc " + ", ".join(proc))
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=False)
        gpus = []
        for index, line in enumerate(result.stdout.splitlines()):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2:
                gpus.append(f"GPU{index} {float(parts[0]) / 1024:.1f}/{float(parts[1]) / 1024:.1f}")
        out.append(" ".join(gpus) if gpus else "GPU unavailable")
    except Exception:
        out.append("GPU unavailable")  # nvidia-smi hangs on a wedged GPU
    disks = []
    for path in paths:
        try:
            stat = os.statvfs(path)
        except OSError:
            continue
        size = stat.f_blocks * stat.f_frsize
        disks.append(f"{path} {(size - stat.f_bavail * stat.f_frsize) / gb:.1f}/{size / gb:.1f}")
    if disks:
        out.append("disk " + ", ".join(disks))
    return "[sys] " + " | ".join(out)


def start_monitor(interval_minutes=10.0, paths=("/",)) -> threading.Event:
    """Daemon census thread: keeps reporting even if the run loop deadlocks."""
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                print(sysline(paths), flush=True)
            except Exception as exc:
                print(f"[sys] census failed: {exc!r}", flush=True)
            stop.wait(interval_minutes * 60.0)

    threading.Thread(target=loop, name="system-stats", daemon=True).start()
    return stop


# --------------------------------------------------------------------------- readiness and games

def wait_for_model(base_url: str, timeout_s: float) -> dict:
    """Wait for /health before the benchmark so startup cannot consume a public game budget."""
    from urllib.request import urlopen

    url = base_url.rstrip("/")
    url = (url[:-3] if url.endswith("/v1") else url) + "/health"
    start = time.monotonic()
    last_report = -30.0
    while time.monotonic() - start < timeout_s:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return {"status": "READY", "ready_epoch": time.time(),
                            "wait_seconds": time.monotonic() - start, "health_url": url}
        except (OSError, TimeoutError):
            pass
        elapsed = time.monotonic() - start
        if elapsed - last_report >= 30:
            print(f"Waiting for model readiness outside the game budget: {elapsed:.0f}s", flush=True)
            last_report = elapsed
        time.sleep(3)
    raise TimeoutError("Model not ready; the benchmark was NOT started. Inspect serve.log.")


def wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    """The Kaggle gateway boots asynchronously; poll until it answers."""
    from urllib.request import urlopen

    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}api/games", timeout=10) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")


def competition_games(base_url: str) -> list:
    """Live competition games served by the Kaggle gateway (real submission)."""
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.COMPETITION,
                                    arc_base_url=base_url, environments_dir="")
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION,
                            arc_base_url=spec.arc_base_url, environments_dir="")
    game_ids = [info.game_id for info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


def offline_games(env_dir, *, excluded=(), prefixes=None, expected_count=None) -> list:
    """Offline games from environment_files; `prefixes` selects and orders a pilot subset."""
    import arc_agi
    import taaf.game_api

    env_dir = str(env_dir)
    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [info.game_id for info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {env_dir}.")
    game_ids = [g for g in game_ids if not any(g.startswith(x) for x in excluded)]
    if prefixes:
        selected = []
        for prefix in prefixes:
            matches = [g for g in game_ids if g.startswith(prefix + "-")]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one game for {prefix}, found {len(matches)}")
            selected.append(matches[0])
        game_ids = selected
    if expected_count is not None and len(game_ids) != expected_count:
        raise RuntimeError(f"Expected {expected_count} games, found {len(game_ids)}")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


# --------------------------------------------------------------------------- provenance and outputs

def save_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str))
    temporary.replace(path)


def write_prompt_artifacts(working_dir) -> dict:
    """Save the exact system prompt and level-start text the model will see.

    Call after the runtime settings are in os.environ: the agent reads them at import.
    """
    from inference.agent.prompts import LEVEL_START_USER_PROMPT
    from inference.agent.tool_agent import effective_system_prompt

    system = effective_system_prompt()
    folder = Path(working_dir) / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "effective-system-prompt.txt").write_text(system)
    (folder / "level-start-user-prompt.txt").write_text(LEVEL_START_USER_PROMPT)
    return {"effective_system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "effective_system_prompt_chars": len(system),
            "level_start_user_prompt_sha256": hashlib.sha256(LEVEL_START_USER_PROMPT.encode()).hexdigest()}


TRANSCRIPT_MARKERS = {
    "no_visible_change_stops": "stop_reason: no_visible_change",
    "border_only_stops": "stop_reason: border_only_change",
    "animation_only_stops": "stop_reason: animation_only",
    "stale_state_refusals": "changed nothing visible, so the rest of this snippet",
    "stall_notes": "on this turn without executing an action",
    "truncation_recoveries": "reached the output limit before calling the tool",
}


def run_summary(working_dir) -> dict:
    """Observation totals plus per-game transcript counts (counts can repeat one event)."""
    from inference.agent.observation_recorder import run_summary as observation_summary

    summary = observation_summary()
    audit = {}
    for path in sorted((Path(working_dir) / "transcripts").glob("*_p*.txt")):
        text = path.read_text(errors="replace")
        row = {key: text.count(marker) for key, marker in TRANSCRIPT_MARKERS.items()}
        row["output_limit_responses"] = len(re.findall(r"^finish_reason: length$", text, re.M))
        audit[path.stem] = row
    summary["transcript_audit"] = audit
    summary["audit_note"] = "Text counts can repeat an event; they are not unique action counts."
    return summary


BUNDLE_FILES = ("experiment_manifest.json", "run_summary.json", "benchmark.json", "summary.txt",
                "git_status.txt", "serve.log", "model_load_report.json", "colab_run_report.json")
BUNDLE_FOLDERS = ("observations", "transcripts", "artifacts", "server_recording")


def export_bundle(working_dir, bundle_name: str) -> Path:
    """Zip the run outputs that the analysis needs into one archive."""
    working_dir = Path(working_dir)
    path = working_dir / bundle_name
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in BUNDLE_FILES:
            source = working_dir / name
            if source.is_file():
                archive.write(source, name)
        for folder in BUNDLE_FOLDERS:
            for source in sorted((working_dir / folder).rglob("*")):
                if source.is_file():
                    archive.write(source, str(source.relative_to(working_dir)))
    print("Bundle:", path.name, "| MiB:", round(path.stat().st_size / 2 ** 20, 2), flush=True)
    export_request_logs(working_dir)
    return path


REQUESTS_ARCHIVE_NAME = "requests.zip"


def export_request_logs(working_dir) -> Path | None:
    """Zip the per-request logs (exact model inputs and outputs) into a separate archive.

    They exist only when the solver runs with save_request_logs (off for real submissions)."""
    working_dir = Path(working_dir)
    sources = sorted(p for p in working_dir.rglob("*requests.jsonl") if p.is_file())
    if not sources:
        return None
    path = working_dir / REQUESTS_ARCHIVE_NAME
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            archive.write(source, str(source.relative_to(working_dir)))
    print("Request logs:", path.name, "| files:", len(sources), "| MiB:",
          round(path.stat().st_size / 2 ** 20, 2), flush=True)
    return path
