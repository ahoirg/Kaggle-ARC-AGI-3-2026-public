# Notices and third-party licenses

This repository mixes our own code with code we adapted from other participants. The ARC Prize 2026 rules ask
authors to release their own code under a permissive public-domain-style license and allow third-party code under
its original open-source license. This file records which is which.

## Our own contributions — MIT-0

Code and documentation we wrote are released under the MIT No Attribution license (`LICENSE`).
That covers our changes inside the files listed below as well as the files we wrote from scratch
(for example `harness/notebook/colab_*.py`, `harness/tools/`, and the agent modules
`action_observation.py`, `death_context.py`, `exploration.py`, `goal_check.py`, `ledger.py`,
`observation_recorder.py`, `selftest.py`, `sim_check.py` and `sim_contract.py`).

## Third-party code included here

| Path | Origin | License |
| --- | --- | --- |
| `harness/ARC3-Inference/` | Tufa Labs, *The Duck* ([github.com/Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness), commit `7652836`), author Jeroen Cottaar and Tufa Labs | MIT, as declared in `ARC3-Inference/pyproject.toml` (the upstream repository has no separate LICENSE file) |
| `harness/ARC3-Inference/` (changes on top of the Duck) | Daniel Franzen, ARC-AGI-3 Milestone 2 solution ([github.com/da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution)) | Apache-2.0, see `LICENSES/Apache-2.0.txt` |
| `harness/notebook/kaggle_sglang_launcher.py`, parts of `harness/notebook/arc3nb.py`, and the structure of `arc-agi-3-final-solution.ipynb` | Daniel Franzen's [competition notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution) | Apache-2.0, see `LICENSES/Apache-2.0.txt` |

**Modification notice (Apache-2.0 §4b).** We modified the files under `harness/ARC3-Inference/`, the launcher
and the notebook listed above. The upstream portions stay under their original licenses; our modifications are MIT-0.

## Not redistributed here

These are attached as Kaggle inputs at run time and are not part of this repository. Their own terms apply.

- TAAF source bundle (`dfranzen/taaf-kaggle-source-bundle-copy`): Tufa Labs' framework snapshot with a pickled benchmark.
- Pennyroyal SGLang wheelhouse (`dfranzen/pennyroyal-v253`).
- Model weights: Intel AutoRound W4A16 Qwen3.8-Flash-Next and the albucino MTP drafter (Kaggle mirrors by `dfranzen`).
- ARC-AGI-3 competition data and game environments (`arc-prize-2026-arc-agi-3`).
