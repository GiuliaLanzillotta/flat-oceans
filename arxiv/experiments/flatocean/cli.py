"""Common command-line options and the explicit process configuration of a driver."""
import argparse
import os

import torch

from flatocean import runs

DECLARED_STAGES = ["high_accuracy", "mse_interpolation", "epoch_250", "late_endpoint"]


def parser(description, stages=DECLARED_STAGES, seeds=(0, 1, 2)):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--model", default="mlp"); ap.add_argument("--tag", default="", help="run tag ('' for ReLU; leaky, gelu, identity, w64, ...)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(seeds)); ap.add_argument("--stages", nargs="+", default=list(stages))
    ap.add_argument("--match-stages-of", default=None, help="take the stages of another tag's run of the same seed (identity control: '' for ReLU)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--overwrite", action="store_true")
    return ap


def configure(args):
    """Process-wide settings, set here and nowhere else: BLAS threads and the torch thread count."""
    torch.set_num_threads(args.threads)
    return args


def stages_of(args, seed):
    """(stage, checkpoint epoch, requested epoch, label) for the requested stages of one seed; matched stages when asked."""
    rd = runs.run_dir(args.dataset, args.model, seed, args.tag)
    if args.match_stages_of is not None:
        src = runs.run_dir(args.dataset, args.model, seed, args.match_stages_of); st = runs.matched_stages(rd, src)
        return [(s, st[s]["checkpoint_epoch"], st[s]["requested_epoch"], st[s]["label"]) for s in args.stages if s in st]
    st = runs.stage_epochs(rd)                       # in the order requested on the command line
    return [(s, st[s]["checkpoint_epoch"], st[s]["checkpoint_epoch"], s) for s in args.stages if s in st and st[s]["checkpoint_epoch"] is not None]
