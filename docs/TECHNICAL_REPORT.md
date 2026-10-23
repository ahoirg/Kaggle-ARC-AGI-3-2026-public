# Technical report: ARC-AGI-3 2026 solution

The full write-up behind the [README](../README.md): credits, results, how the solution works, how to run it,
the experiment log and what we learned.

> Status: r23 harness. Final leaderboard scores are added after the competition closes (2 November 2026).
> <!-- TODO after 2 Nov: final private/public LB scores and the exact submitted harness version. -->

## Contents

1. [Credits and prior work](#credits-and-prior-work)
2. [Results](#results)
3. [How the solution works](#how-the-solution-works)
4. [How to run it](#how-to-run-it)
5. [Repository layout](#repository-layout)
6. [Experiment log](#experiment-log)
7. [What we learned](#what-we-learned)
8. [Limitations](#limitations)
9. [License](#license)

## Credits and prior work

Most of what makes this solution run was built and shared by others. We are grateful to all of them.

- **Daniel Franzen: ARC-AGI-3 Milestone 2 solution.**
  Our notebook was developed on the basis of his
  [competition notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution)
  ([source and write-up](https://github.com/da-fr/arc-agi-3-solution), Apache-2.0).
  - His work: the model choice, the optimized SGLang serving setup, the priority scheduler and context management.
  - Also his: the perception improvements (board-difference, death and animation images), executable undo,
    retained Python functions and the stale-state / no-op guards.
  - The Kaggle inputs we attach are his published copies.
- **Jeroen Cottaar and Tufa Labs: The Duck harness and TAAF.**
  The tool-using agent (`ARC3-Inference`, "The Duck") and the game execution framework (TAAF) come from Tufa
  Labs' [Duck harness](https://github.com/Tufalabs/duck-harness). It was the winning solution of ARC-AGI-3
  Milestone 1 and is described in their
  [technical write-up](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133).
  Franzen's fork, and so ours, is based on upstream commit `7652836`.
- **Serving.** The offline SGLang runtime builds on the work of
  [John Pezzulli (Pennyroyal)](https://github.com/jpezzulli/sglang-rtxpro6000),
  [Mamy Ratsimbazafy](https://github.com/mratsim/sglang-qwen38fn-sm120-turbo),
  [Gabriel Olympie](https://github.com/gabrielolympie/sglang-flashnext-sm120) and the
  [SGLang](https://github.com/sgl-project/sglang) project.
- **Models.** Thanks to the Qwen team for
  [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Intel for the
  [W4A16 AutoRound quantization](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound), and the
  [albucino](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) authors for the MTP drafter.
- **ARC Prize Foundation**, for ARC-AGI-3, its public games and this competition.

The same credits appear at the top of the notebook. [`NOTICE.md`](../NOTICE.md) lists which files come from which
project and under which license.

## Results

### Hidden set (Kaggle leaderboard)

The public leaderboard scores about half of the 110 hidden games. Each line below is a single submission, so
one draw says little. For scale, the copies of Franzen's notebook submitted by many teams gave
**mean 25.8, SD 3.9**
([discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/745062)).

| Submission | Public LB |
| --- | ---: |
| An early version (Action Evidence era) | 31.07 |
| r6 harness | 26 |
| r15 harness | 22.0 |
| Another intermediate version | ~23 |
| **Final submission** | *TBD after 2 Nov* <!-- TODO --> |

**Honest summary.** No version we built showed a hidden-set gain we could separate from the variance of the
Franzen baseline. The top of the leaderboard was around 60–69 during the same period.

### Public games (offline runs)

The 25 public games run offline in the notebook. `RUN_STAGE="FULL"` plays all 25 games, 120 minutes each.
`RUN_STAGE="PILOT"` plays 10 games, 45 minutes each.

| Run | Games | Mean score | Levels completed |
| --- | ---: | ---: | ---: |
| r23 FULL, Kaggle | 25 | 35.35 | 88 / 183 |
| r23 FULL, Colab run 1 | 25 | 43.26 | 95 / 183 |
| r23 FULL, Colab run 2 | 25 | 45.52 | 104 / 183 |
| 45-minute PILOT runs, r4–r10 (11 runs) | 10 | 62.3 ± 6.0 | — |
| r15 PILOT, Colab / Kaggle (identical code) | 10 | 56.6 / 70.4 | 48 / 55 of 76 |

The pilot is **not** a proxy for the hidden set:

- Every change after 10 October was developed while watching these 10 games.
- In the same FULL runs, the 10 pilot games scored about twice the other 15. For example, r13 FULL scored
  62.9 on the pilot games and 31.8 on the rest.
- Identical code differed by 7 levels between two runs (r15).

## How the solution works

```
Kaggle / Colab notebook (arc-agi-3-final-solution.ipynb)
 ├─ configuration: runtime, RUN_STAGE (PILOT | FULL), harness location, Kaggle inputs
 ├─ harness install: TAAF source bundle copied, its src/ARC3-Inference replaced by harness/ARC3-Inference
 ├─ runtime settings: every agent / scheduler / framework option as an environment variable
 ├─ CPU self-test + provenance: prompt contract and sandbox checks, manifest, exact system prompt, file digests
 ├─ model server: SGLang (Pennyroyal wheelhouse), Qwen3.8-Flash-Next W4A16 + albucino MTP drafter
 ├─ benchmark: TAAF deployment target and benchmark restored; budgets and concurrency set
 ├─ run: offline public games, or the live competition games through the Kaggle gateway
 └─ export: run_summary.json and a run bundle (transcripts, observations, artifacts, request logs)
```

### Model and serving (Franzen's setup, unchanged)

- **Hardware.** One RTX PRO 6000 (96 GB), no internet, 9-hour notebook budget.
- **Model.** Qwen3.8-Flash-Next W4A16 (Intel AutoRound): a MoE with about 6B active parameters. It is served by
  SGLang with an INT4 MTP drafter (NEXTN, 3 draft steps), an FP8 KV cache and up to 10 concurrent requests.
- **Context and sampling.**
  - Context: 128k window, 12k output cap.
  - Sampling: temperature 1.0, top-p 0.95, top-k 20 (the Qwen card's thinking-mode values). Franzen used 0.7.
  - Thinking is on, and reasoning is kept in history.
- **Submission budget.** All 110 games are admitted, 10 streams decode at once, and the priority scheduler
  decides which game gets the next turn.

### Agent

The Duck plays each game through a single Python tool. Each turn the model sees an upscaled image of the board,
a difference image, animation frames, and a text opener. It then calls `python` to inspect the frame and send
actions. Functions it writes are retained for the rest of the game.

### What we added on top of Franzen's notebook

| Feature (setting) | Since | What it does |
| --- | --- | --- |
| Single-source notebook + `harness/` dataset | r1 | One notebook for Kaggle and Colab; all agent code in one versioned directory with a SHA-256 manifest; no patch files. |
| Action Evidence v2 (`ARC3_ACTION_EVIDENCE`) | pre-r1 | Truthful action results: interior changes vs border-only vs no change; inconclusive cases are never stored as no-ops; truthful guard messages. |
| Unified prompt contract | r1–r3 | One system prompt; dynamic-only openers; truncation recovery (`ARC3_TRUNCATION_RECOVERY`); stall note (`ARC3_STALL_SECONDS`); time cues (`ARC3_TIME_CUES`). |
| Exploration note (`ARC3_EXPLORATION_NOTE`) | r3 | Lists actions never tried on the level and state-specific no-ops. |
| Goal check (`ARC3_GOAL_CHECK`) | r4 | Runs the model's own retained `goal_violations(frame)` each turn, or asks for one once per level. |
| Simulator contract (`ARC3_SIM_CHECK`, `ARC3_SIM_GUARD`) | r13–r15 | If the model writes `observe(frame)` and `step(state, action)`, they are replayed over the level's real transitions. `search(goal)` plans over them, and a batch stops at the first mispredicted action. |
| Death context (`ARC3_DEATH_CONTEXT`) | r14 | On a game-over turn: the fatal animation, object tracks and what the vanished object touched. |
| Scheduler token scale (`ARC3_PRIORITY_TOKEN_SCALE=250000`) | r16 | Tuned from a replay of logged runs: the completion hazard stays flat after ~40k tokens on a level. |
| Context compaction (`ARC3_CONTEXT_COMPACTION`) | r17–r18 | When the context fills, the model writes notes that replace the dropped turns, instead of silent truncation. The notes request reuses the prefix cache. |
| Exact request logs | r23 | Every model request and response is exported per game (`requests.zip`), for analysis and SFT data; logging only. |
| Colab runtime | r1+ | A pinned Python 3.12 harness process on Colab with checksummed Drive inputs; the notebook cells are identical on both platforms. |

## How to run it

### On Kaggle

1. **Inputs.** Join the competition and attach these inputs (see `kernel-metadata.example.json`):
   - datasets `dfranzen/pennyroyal-v253` and `dfranzen/taaf-kaggle-source-bundle-copy`;
   - models `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` and `dfranzen/albucino-qwen3-8-flash-next-drafter`;
   - the `arc-prize-2026-arc-agi-3` competition data.
2. **Harness.** Upload `harness/` unchanged as a Kaggle dataset (the public copy is `ahoirg/arc3-harness`). Set
   `HARNESS_DIR_KAGGLE` in the configuration cell. If the path is wrong, the notebook searches `/kaggle/input`.
3. **GPU.** Select the **RTX Pro 6000** GPU and turn internet off.
4. **Run.** Choose `RUN_STAGE` (`PILOT` or `FULL`) and run all cells in a fresh session.
   - Outside a competition rerun, the notebook plays the offline public games.
   - When Kaggle reruns it for scoring (`KAGGLE_IS_COMPETITION_RERUN`), it plays the hidden games through the
     gateway with minimal diagnostics.
5. **Pushing with the Kaggle CLI.** Copy `kernel-metadata.example.json` to `kernel-metadata.json` and set your
   kernel id. Then run `push_to_kaggle.bat`, or `kaggle kernels push -p .`.

### On Colab (GPU test runs)

1. **Copy inputs to Drive.** Copy the wheelhouse, the TAAF source bundle and the competition data to Google Drive.
   Upload `harness/` to Drive as well.
2. **Write the Colab config.** Create `/content/drive/MyDrive/ARC3/colab_config.json` with `schema_version: 1`,
   plus:
   - `inputs`: `main_model` with the 50 checkpoint files, each with its size, SHA-256 and Drive file id;
     `wheelhouse`, `draft_model` and `source_bundle`, each with its handle, version 1 and a Drive `local_path`;
     `competition`;
   - `outputs`: `local_work_dir` and `drive_results_dir`.

   The loader rejects any checkpoint that differs from the original files. The full validation is in
   `harness/notebook/colab_adapter.py`.
3. **Run.** Set `RUNTIME_ENVIRONMENT = "colab"` and run all cells. A GPU with enough memory for the model is
   required; we used an RTX PRO 6000 class runtime.

### Without a GPU

- CPU self-test of the agent's prompt contract and sandbox:
  `cd harness/ARC3-Inference && python -m inference.agent.selftest`
- Run the whole notebook on CPU against a scripted fake model that replays recorded actions:
  `python harness/tools/notebook_cpu_test.py NOTEBOOK SOURCE_BUNDLE ENV_FILES WORK_DIR [--games lf52,vc33] [--minutes 3]`
- After editing anything in `harness/`, rebuild the manifest:
  `python harness/tools/build_manifest.py --version <label> --note "<what changed>"`

### Reading a run

Each run writes a bundle zip with these contents:

- `experiment_manifest.json`: settings, harness digests and the exact system prompt;
- `run_summary.json`: per-game levels and score;
- per-game transcripts, per-action observations and `requests.zip`.

Tufa's viewer renders a run directory: `cd harness/ARC3-Inference && make view VIEW_RUN_DIR=/path/to/run`.

## Repository layout

| Path | Contents |
| --- | --- |
| `arc-agi-3-final-solution.ipynb` | The notebook (Kaggle and Colab). |
| `harness/HARNESS_MANIFEST.json` | Version label, per-file SHA-256, tree digest. |
| `harness/ARC3-Inference/` | The agent: Tufa's Duck with Franzen's changes and ours (`inference/agent/` holds the tool agent, prompts, scheduler and our modules). |
| `harness/notebook/` | Notebook helpers (`arc3nb.py`), Kaggle SGLang launcher, Colab adapter and model loader. |
| `harness/tools/` | Manifest builder, CPU notebook test and fake model. |
| `kernel-metadata.example.json`, `push_to_kaggle.bat` | Kaggle CLI push template. |
| `docs/TECHNICAL_REPORT.md` | This report. |
| `LICENSE`, `NOTICE.md`, `LICENSES/` | Licenses and third-party notices. |

## Experiment log

**Protocol.**

- PILOT runs: 10 public games, 45 minutes per game, usually two runs per change.
- FULL runs: 25 games, 120 minutes per game.
- Leaderboard submissions were used sparingly.
- One intervention at a time. Rejected code was deleted, not hidden behind flags.

| Version | Date | Change | What we saw |
| --- | --- | --- | --- |
| base | 9 Oct | Franzen's Milestone 2 notebook; harness patch loaded from a dataset; serving pinned to Python 3.12. | Reference. |
| pre-r1 | 10–12 Oct | Action Evidence v2; recovery guidance, target-ID helpers, shared planner, evidence workspace (most removed again in r1). | Prose and tool additions did not move scores; the base was simplified. |
| r1–r3 | 12 Oct | Slim notebook driving the versioned `harness/`; unified prompt contract; exploration note. | Cleaner provenance; no measurable score change. |
| r4 | 13 Oct | Goal check; pilot budget 30 → 45 min. | 30 → 45 min added ~22 points to the pilot. This was budget, not code. |
| r5–r7 | 13 Oct | View-shift registration and corrections for scrolling games. | Mechanism worked in logs; no score effect. r6 submitted: **26** on LB. |
| r9–r10 | 13–14 Oct | Never-pressed-key escalation; motion summary, predict check, level ledger. | Pilot 52–72 range, no ordering a 6-point SD supports. A medium `reasoning_effort` trial was much worse; we kept the template default. |
| r12 | 14 Oct | "Lean" harness: deleted escalation, predict check, in-batch line; temperature 1.0. | Same range with fewer moving parts. |
| r13 | 14 Oct | Auto-probe (harness presses an untried key once); simulator check. | FULL run analysis: net effect of auto-probe ≈ 0. |
| r14 | 14 Oct | Death context. | 43/76 levels; within noise. |
| r15 | 14 Oct | Simulator contract (`observe`/`step`, `search`, batch guard). | Best pilot (70.4 on Kaggle), but the gain came from games that never used the contract. Same code on Colab scored 56.6. LB **22.0**. The model sometimes "blinded" `observe` to pass the guard. |
| r16 | 15 Oct | Auto-probe removed; scheduler token scale 62k → 250k. | Scheduler replay: +0.8 (25 games) / +1.1 (110 games) simulated. |
| r17–r18 | 15 Oct | Context compaction (r17); notes request reuses the prefix cache (r18). | 58 compactions, 0 failures; notes accurate; no visible pilot gain (44.4 / 57.0). r17 notes requests missed the prefix cache; r18 sends them with the turn's own settings. |
| r19, r22 | 15–16 Oct | Qwen3.8-27B-FP8 instead of Flash-Next. | Serving failed in r19, worked in r22 (~437 vs ~640 tok/s); pilot 20/76 levels vs ~48. Dropped. |
| r20–r20b | 16 Oct | Level-start "checkpoint" evaluation, replaying known prefixes to start mid-game. | Cold start solved 15%; starting one level earlier solved 25%. Earlier-level experience carries most of the work. Reverted. |
| r21 | 16 Oct | Prompt patch from contrast analysis of successful vs failed runs. | Prepared but never committed; the main line was rolled back to r18. |
| r23 | 17 Oct | r18 harness + exact per-request logs. | Three FULL runs: 35.35 / 43.26 / 45.52 (this release). |

**Side tracks (not part of this release).**

- **Action-effect / RL helper.** A small model that predicts which click changes the board. A memory table beat
  an online CNN on unseen games, but the LLM already wastes only ~2.3% of its clicks, so this targets a
  non-bottleneck. Not integrated.
- **LoRA SFT on Flash-Next.** We built an exact-render rejection-SFT dataset from r23 request logs: 15 train,
  3 val and 7 test games. The CPU checks of the collator and the gradients pass.
  - Problem 1: no published recipe fits a single 96 GB GPU.
  - Problem 2: stock SGLang cannot serve a runtime LoRA on this architecture, so an adapter would have to be
    merged into the 16-bit tensors.
  - As of 17 October, no model had been trained.
- **Bounded action-outcome oracle.** A pre-registered three-arm pilot protocol: does correct action-outcome
  feedback help the agent? In progress at the time of writing; excluded from this release.

## What we learned

1. **The loss is levels never reached, not action efficiency.**
   - On completed levels we are at or near the human action baseline.
   - Almost all of the lost score is unfinished levels. Later levels weigh more: the last level of a 6-level
     game is 28.6% of the game.
2. **Time per game moved scores more than any feature.**
   - 30 → 45 minutes per pilot game: about +22 points.
   - On 110 hidden games, the same 10 decode streams are spread over 4.4× more games.
3. **Mechanism effects are not score effects.**
   - Every feature after r3 did what it was built to do in the transcripts: goal check, exploration note,
     view shift, escalation, ledger, simulator contract, compaction.
   - None showed a score effect we could measure with two runs.
4. **The pilot overfits.** Ten games, watched for two weeks, scored about twice the other public games. We should
   have held out more games earlier and used FULL runs and the leaderboard as the judge.
5. **Direct pressure on the model's reasoning backfired.**
   - More prose rules, forced probes and a lower `reasoning_effort` all hurt or did nothing.
   - In the failed runs, the model held wrong hypotheses too long and repeated failing plan families. Most of
     the rules against this were already in the prompt.
6. **Failure taxonomy (r13 FULL, 18 unfinished games, judged from transcripts).**
   - model reasoning ≈ 41%;
   - starved by the scheduler ≈ 40%;
   - correct plan cut by time ≈ 12%;
   - harness got in the way ≈ 6%.
7. **What others report agrees.** Perception gains (upscaling, animations) mattered more than added machinery
   for Franzen, for Tufa and for other write-ups. Our additions were mostly machinery.

## Limitations

- Leaderboard numbers are single submissions, reported as we saw them. The public LB covers about half of the
  hidden games.
- Offline numbers come from 1–3 runs per version. Run-to-run spread on identical code is ±6–7 levels on the pilot.
- Much of the experiment log above was summarized from internal notes; the raw run bundles are not published
  (they are several GB).
- The harness keeps some unused options inherited from upstream (for example the OpenRouter config and the
  Slurm targets).

## License

- Our own code and documentation: [MIT No Attribution](../LICENSE) (MIT-0), as required by the ARC Prize 2026 rules.
- Code adapted from Tufa Labs (MIT) and Daniel Franzen (Apache-2.0) keeps its original license. See
  [`NOTICE.md`](../NOTICE.md) and [`LICENSES/`](../LICENSES/).
- Model weights, the SGLang wheelhouse, the TAAF bundle and the competition data are not included. They are
  attached from Kaggle and have their own terms.
