# ARC3 harness

Everything `arc-agi-3-final-solution.ipynb` runs besides the model and the TAAF framework.
Upload this whole directory, unchanged, as a Kaggle dataset and to Google Drive, then set
`HARNESS_DIR_KAGGLE` and `HARNESS_DIR_COLAB` in the notebook's configuration cell.

| Path | Contents |
| --- | --- |
| `HARNESS_MANIFEST.json` | Version label, per-file SHA-256 (CRLF-normalized), tree digest. |
| `ARC3-Inference/` | Complete replacement for `src/ARC3-Inference` of the TAAF source bundle (`dfranzen/taaf-kaggle-source-bundle-copy`). Contains Daniel Franzen's Milestone-2 harness changes and all of our changes, applied in place. No patch files are needed. |
| `notebook/arc3nb.py` | Notebook helpers: harness install and verification, page-cache warmup, census monitor, readiness, game lists, provenance, run summary, bundle export. |
| `notebook/kaggle_sglang_launcher.py` | Kaggle model server launch (SGLang from the Pennyroyal wheelhouse). |
| `notebook/colab_adapter.py`, `colab_model_loader.py`, `colab_bridge_child.py` | Colab runtime: Drive inputs, model loading, the Python 3.12 harness process and cell forwarding. |
| `tools/build_manifest.py` | Rebuilds the manifest after an edit. |
| `tools/notebook_cpu_test.py` | Runs the notebook on CPU against a scripted fake model (development check). |

## Editing workflow

1. Edit files here (the agent lives in `ARC3-Inference/inference`).
2. Run the CPU self-test: `cd harness/ARC3-Inference && python -m inference.agent.selftest`.
3. Rebuild the manifest with a new version label:
   `python harness/tools/build_manifest.py --version 2026.10.13-r1 --note "what changed"`.
4. Upload the directory to Kaggle (new dataset version) and Drive.

The notebook warns, and still runs, if the uploaded files differ from the manifest; every run records the
actual file digests, the effective system prompt and a copy of the agent sources in its bundle.

## Provenance of the base

- TAAF source bundle: `dfranzen/taaf-kaggle-source-bundle-copy` v1 (framework and pickled benchmark/target).
- `ARC3-Inference` base: Tufa Labs' Duck harness ([github.com/Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness),
  commit `7652836`) with Daniel Franzen's Milestone-2 changes
  ([github.com/da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution), Apache-2.0), applied from his
  `harness-changes.patch` (SHA-256 `80aef535…`), then our Action Evidence v2 changes.
- Revision `2026.10.12-r1` consolidates all of the above into source and adds the unified prompt contract
  (single system prompt, dynamic-only openers, truncation recovery, stall note, time cues) and removes dead
  features.
