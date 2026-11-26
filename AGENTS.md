# Repository layout

The public repository contains `arxiv/` for the current paper and reproduction code. A local `aistats/` folder may contain private submission and rebuttal documents: preserve it locally, keep it ignored, and never commit or push it. Do not add another parallel manuscript or experiment tree.

Read `arxiv/CONTRIBUTION_FRAMING.md` and `arxiv/WRITING_STYLE.md` before editing the paper. Experiment instructions are in `arxiv/experiments/AGENTS.md`. The AISTATS folder is a document archive and must contain no experiment or helper code.

Build the paper with `python arxiv/experiments/scripts/build_paper.py`. Reproduction commands run from `arxiv/experiments/`. Local tool installations are hidden under `.tools/`; they are not repository content. Do not commit or push without authorization.
