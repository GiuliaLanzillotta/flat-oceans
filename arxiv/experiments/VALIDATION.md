# Cleanup validation

The cleanup preserves the submitted arXiv manuscript and its scientific conventions. The current paper source is `../`. Original submission and rebuttal material is kept locally outside the public repository.

## Checks completed

- All 110 retained regression tests passed at the final repository path (254.33 seconds), including small training/replay tests. The reproduction runner additionally checks read-only planning, stopping on failure, command-aware resumption, and relocated saved paths.
- All 887 experiment recipes have existing entry points and pass static argument checks. Dependency ordering includes ray measurements before their gain-bound consumers. The catalog includes 155 recorded training runs and the subsequent measurement and validation commands.
- All 71 published figure assets were regenerated using only the retained source and measurement files.
- The clean manuscript compiled to 109 pages. Its extracted text is byte-for-byte identical to the submitted PDF; approved figure assets are preserved.
- The Python wheel builds successfully as `flat-oceans` version 0.1.0.
- The exact-ocean coverage sweep reproduces every saved three-seed Hessian/GGN coverage value at all 11 thresholds exactly. Independent coordinate/projection checks pass.
- Regenerated curvature-control directions and recovered visualization inputs match the saved arrays exactly.
- Removed 153 obsolete Python files and the old launch pipelines/manuscript variants. Result-file pruning removes 12,146 obsolete files (126.45 GB); the retained factors, bases, checkpoints, and measurements occupy approximately 925.37 GB locally. Large generated arrays are excluded from Git.

## Limits

The complete 887-step experiment suite has **not** been retrained and recomputed from scratch during this cleanup. The checks above validate the retained implementation, saved-data replay, recipe syntax, and paper build; they do not establish a fresh end-to-end scientific reproduction on different hardware. Full reproduction requires downloading the datasets and pretrained weights, substantial compute and storage, and running the catalog in order. Install this repository in editable mode as documented in `README.md`.

The pre-cleanup Git history was preserved locally. A pre-cleanup source snapshot and the original working-tree diff were retained outside the repository for recovery; no commit or push was made.

## Two-folder layout (26 November 2026)

The Git root is the workspace root. Only `arxiv/` and the root project documentation are published; `aistats/` is an ignored local archive. The manuscript starts at `arxiv/main.tex`; included sections are grouped under `arxiv/sections/`; code and measurements live under `arxiv/experiments/`. AISTATS contains documents only. Old outer workspace directories and the unused environment were removed; the working environment remains locally under `.tools/venv/`.

After relocation, the manuscript builds with unchanged extracted text. Its ZIP contains exactly the 120 declared manuscript dependencies and excludes all code and results. The curvature figure recipe also runs successfully from the new code location.

All 110 regression tests also pass after this relocation. The working Python environment and reproduction CLI were checked at their new paths.
