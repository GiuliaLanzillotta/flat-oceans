# Flat oceans

Reproduction code for **Flat oceans: vast convex invariance at the end of training**, by Giulia Lanzillotta and Thomas Hofmann.

The current manuscript is `../main.tex`.

Run the commands below from `arxiv/experiments/`.

## Install

Use Python 3.10 or newer. The recorded environment used Python 3.10, PyTorch 2.5.1, CUDA 12.4, and float64 for the geometry experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,resisc]'
python scripts/download_data.py --resisc
```

MNIST and SVHN are downloaded through TorchVision. RESISC45 uses a pinned revision of the public `timm/resisc45` parquet dataset. Downloads are checked against the SHA-256 hashes of the experimental inputs. Data defaults to `data/`; set `FLATOCEAN_DATA_ROOT` to use an existing dataset directory. Individual locations can be set with `FLATOCEAN_MNIST_ROOT`, `FLATOCEAN_SVHN_ROOT`, and `FLATOCEAN_RESISC45_ROOT`.

## Hardware

The reference server had two AMD EPYC 9554 CPUs (128 physical cores), about 377 GiB RAM, and four 48 GB NVIDIA RTX A6000 GPUs. The serial catalog does not require four GPUs. For a full reproduction, plan for roughly 384 GiB host RAM, 16 available CPU threads, one 48 GB CUDA GPU, and at least 2 TB free storage; these are provisioning budgets, not validated minimums. Paper compilation and the regression tests require no GPU.

See [HARDWARE.md](HARDWARE.md) for task-specific requirements, precision, storage, and shared-server limits.

## Reproduce the experiments

The executable catalog is `configs/experiments.json`. It records commands in dependency order; `configs/training.json` contains the original training settings, including the sibling runs. `configs/finetuning.json` records full-network ViT and ResNet fine-tuning.

```bash
python reproduce.py list                 # experiment groups
python reproduce.py all                  # inspect every command without running it
python reproduce.py all --execute        # run serially; stop at the first failure
python reproduce.py all --execute --resume
```

A group or a single step can be selected:

```bash
python reproduce.py curvature
python reproduce.py curvature --execute
python reproduce.py training --only train:mnist_mlp_w64/seed_0 --execute
```

Selecting a group does not run its prerequisites. Follow the printed group order for a fresh reproduction. The complete suite includes 500-epoch training runs, dense Jacobian factorizations, and large saved eigensystems; it requires a research server, substantial disk space, and sustained compute. `--threads` controls the runner's CPU thread cap. On a shared server, use the admission wrapper in `scripts/jobs.sh` and the resource rules in `AGENTS.md`.

Results are written under `results/`. Large checkpoints, factors, training histories, and raw probe arrays are generated locally and excluded from Git. Scalar reference records remain alongside them. The source and recipes are sufficient to regenerate these artifacts; cloning alone does not supply the large arrays needed to redraw every figure immediately.

The numerical gates are part of the protocol. A failed rank, residual, or provenance check stops its command; do not change thresholds to obtain a pass. Floating-point factorizations and training trajectories may differ on another hardware/software stack. Reference hashes establish which inputs produced the reported measurements, rather than promising bitwise training equivalence across platforms.

## Figures and paper

Once the measurements exist:

```bash
python -m flatocean figures --list
python -m flatocean figures
python -m flatocean figures --only curvature
python scripts/build_paper.py
```

The figure command regenerates the 71 published assets into `build/figures/`, without editing the manuscript or its approved figures. `configs/figures.json` maps each published asset to its plotting recipe. The paper build uses Tectonic by default; `--pdflatex` selects an installed TeX Live distribution. It writes `../flat_oceans.pdf` and `../arxiv.zip`. Building the paper requires neither datasets nor experiment outputs: the approved figures and bibliography are included in `../`.

## Layout

| Path | Purpose |
|---|---|
| `flatocean/` | Network models, frozen responses, subspace algebra, analyses, and plotting |
| `scripts/analysis/` | Experiment-specific measurement programs |
| `paper_figures/` | Slice measurements and the remaining small figure recipes |
| `configs/` | Recorded training settings, experiment commands, datasets, and figure catalog |
| `tests/` | Numerical, provenance, CLI, and small training regression tests |
| `../` | Current LaTeX source, tables, bibliography, and published figures |
| `results/` | Measurement records and locally generated large artifacts |
| `build/`, `../.build/` | Disposable figure and LaTeX build intermediates |

## Validation

```bash
pytest -q
python scripts/download_data.py --check --resisc
```

Tests use small synthetic problems and temporary directories; the training smoke tests use the downloaded MNIST data. `VALIDATION.md` records the cleanup checks and their limits. Scientific protocol identifiers in saved records are preserved verbatim, including historical memo names; those strings identify the measurement convention and are not runtime file dependencies.
