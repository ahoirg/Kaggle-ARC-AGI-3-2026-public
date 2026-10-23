# TODO before making this repository public

Working checklist for the public release (planned ~6 November 2026, after the 2 November final deadline and
before official private scores). Delete this file, or move it out of the repository, before publishing.

## 1. After the final deadline (2 Nov): sync to the final version
- [ ] Decide which harness version was the final submission (currently r23 here; the private repo also has the
      uncommitted r26 oracle experiment, which is excluded).
- [ ] If the final version is not r23: copy the final notebook and `databases/harness` again, re-apply the public
      edits (English console messages, `harness/` paths, credits cells) and rebuild the manifest with a new label
      (`python harness/tools/build_manifest.py --version <final>-public1 --note "..."`).
- [ ] Run the CPU self-test: `cd harness/ARC3-Inference && python -m inference.agent.selftest`.
- [ ] Tag the source commit in the private repo (for example `public-release-v1`).

## 2. Fill in the placeholders (marked `<!-- TODO -->`)
- [ ] `README.md` line 13: status line, add final LB scores and the submitted harness version.
- [ ] `README.md` line 50: "Final submission" row of the results table.
- [ ] `docs/TECHNICAL_REPORT.md` line 7: status block, final private/public LB scores and the submitted version.
- [ ] `docs/TECHNICAL_REPORT.md` line 68: "Final submission" row of the hidden-set table.
- [ ] Remove the `<!-- TODO -->` comments once filled.

## 3. Facts to confirm in docs/TECHNICAL_REPORT.md
- [ ] Name of the version that scored 31.07 on the LB (now "An early version (Action Evidence era)").
- [ ] The "~23" LB entry: which version.
- [ ] LoRA SFT status: the report says "As of 17 October, no model had been trained"; update if B300 training ran.
- [ ] r21: result of the prompt patch, if it was ever run (the report says "prepared but never committed").
- [ ] Bounded oracle experiment (r26): add the outcome or keep it as "in progress / excluded".
- [ ] Experiment log: add any versions after r23.

## 4. Licenses and credits
- [ ] `LICENSE` copyright line says "ahoirg"; change to your full name if preferred.
- [ ] Duck harness has no LICENSE file upstream; MIT is taken from its `pyproject.toml`. Re-check before
      publishing (or ask Tufa Labs).
- [ ] Re-read the credits cell in the notebook and the README credits paragraph.

## 5. Pre-publication checks
- [ ] Secret scan of the whole tree (for example `gitleaks detect --no-git -s .`); no Kaggle tokens, Drive file
      ids of private files, or personal paths.
- [ ] Notebook outputs empty (`execution_count` null, no outputs).
- [ ] Fresh clone + CPU notebook test with the fake model (`harness/tools/notebook_cpu_test.py`).
- [ ] Check that every link in `README.md`, `docs/TECHNICAL_REPORT.md` and `NOTICE.md` opens.

## 6. Kaggle
- [ ] Upload `harness/` as a new version of `ahoirg/arc3-harness` and make the dataset public.
- [ ] Make the final Kaggle notebook public (or publish a public copy) and link it from the README.
- [ ] Optional: post a short write-up in the competition discussion with links to this repo.

## 7. Publish
- [ ] Commit everything except this file (first public commit), push, switch the GitHub repo to public.
- [ ] Optional: ARC Prize paper award deadline is 8 November; the technical report can serve as its basis.
