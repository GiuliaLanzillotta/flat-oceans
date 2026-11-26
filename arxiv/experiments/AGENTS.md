# Working on this repository

The current paper is `../main.tex`. Read `../CONTRIBUTION_FRAMING.md` before changing its scientific framing and `../WRITING_STYLE.md` before editing text or figures. If present, `../../aistats/` is private local material. Preserve it, keep it ignored, and never commit or push it.

Keep experiment settings in `configs/`, shared numerical routines in `flatocean/`, and experiment-specific entry points in `scripts/analysis/`. Use the existing models, operators, rank rules, probe sampling, and validation gates. Do not silently substitute approximations or change thresholds. Failed gates are scientific outcomes, not permission to retune.

Generated plots go to `build/figures/`; paper build intermediates go to `../.build/`. Plotting must never rewrite manuscript sources. `../figures/` contains the approved published assets. Use the common plotting style and square appendix panels; preserve the explicitly approved wide CNN-unit and LMC-plane panels.

Every retained experiment must have a recipe in `configs/experiments.json`. Add dependencies before consumers. A new scientific experiment requires the user's authorization; reproducing existing measurements and running regression tests are ordinary validation.

On the shared analysis server: inspect load, memory, and GPU use before launching long work. Heavy matrix assembly/factorization uses 16 CPU threads, at most two concurrent heavy jobs, and at least 100 GB free-memory headroom beyond the job's estimated peak. Do not launch when one-minute load exceeds 120. Use `scripts/jobs.sh` for admission and logging. Never run two writers against one result file. Do not weaken provenance or numerical gates to continue a pipeline.

Run relevant tests after changes. `python -m flatocean figures` replays the published plots from saved measurements; `python scripts/build_paper.py` compiles and packages the current paper. Full retraining is expensive and must not be claimed as validation unless it actually ran.
