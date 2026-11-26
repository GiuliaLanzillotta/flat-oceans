"""One entry point for every analysis and every figure, table or summary:  python -m flatocean <command> [arguments]

Only the selected analysis module is imported; its arguments are forwarded unchanged to its own parser, so every
command keeps the argument list of the former exp_*.py script (the table below gives the old name of each command).
`python -m flatocean` alone lists the commands; an unknown command exits with status 2.
"""
import importlib
import sys

COMMANDS = {                       # command -> (module under flatocean, former script); "analyses." or "reporting." prefix
    "atomic-stages": ("atomic", "exp_atomic_stages.py"),
    "kernel-persistence": ("persistence", "exp_kernel_persistence.py"),
    "kernel-persistence-siblings": ("persistence_siblings", "persistence_siblings.py"),
    "transition-residual": ("transitions", "exp_transition_residual.py"),
    "multilayer-branch": ("branch", "exp_multilayer_branch.py"),
    "cnn-patch-spaces": ("cnn_patch_spaces", "exp_cnn_patch_spaces.py"),
    "cnn-branch": ("cnn_branch", "exp_cnn_branch.py"),
    "cnn-channels": ("cnn_channels", "exp_cnn_channels.py"),
    "cnn-background": ("cnn_background", "exp_cnn_background.py"),
    "neuronwise": ("neuronwise", "exp_neuronwise.py"),
    "clusters": ("clusters", "exp_clusters.py"),
    "lmc": ("lmc", "new on 2026-09-25 (spawn-state, segment)"),
    "soft-selectivity": ("soft_selectivity", "new on 2026-09-26 (GELU line item (a): per-unit profiles and tau-activity)"),
    "epsilon-flat": ("epsilon_flat", "new on 2026-09-26 (GELU line item (c): survival curves and the smooth transition bound)"),
    "neural-collapse": ("neural_collapse", "exp_neural_collapse.py"),
    "new-batch-exposure": ("new_batch_exposure", "exp_new_batch_exposure.py"),
    "branch-rays": ("branch_rays", "exp_branch_rays.py"),
    "symmetries": ("symmetries", "exp_symmetries.py"),
    "structured-directions": ("structured_directions", "exp_structured_directions.py"),
    "persistence-geometry": ("persistence_geometry", "exp_persistence_geometry.py"),
    "gated-null": ("gated_null", "exp_gated_null.py"),
    "figures": ("reporting.paper", "all current arXiv figures"),
}


NO_ARGUMENT_COMMANDS = set()


def usage(stream=sys.stdout):
    print("usage: python -m flatocean <command> [arguments]\n\ncommands (former script):", file=stream)
    for cmd, (mod, old) in COMMANDS.items():
        print(f"  {cmd:24s} {old}", file=stream)
    print("\n`python -m flatocean <command> --help` shows the arguments of a command.", file=stream)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        usage(); return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"flatocean: unknown command '{cmd}'\n", file=sys.stderr); usage(sys.stderr); return 2
    if cmd in NO_ARGUMENT_COMMANDS:
        if len(argv) > 1:
            if argv[1:] in (["--help"], ["-h"]):
                print(f"usage: python -m flatocean {cmd}\n\nGenerate this figure group from saved measurements.")
                return 0
            print(f"{cmd}: unrecognized arguments: {' '.join(argv[1:])}", file=sys.stderr)
            return 2
    mod = COMMANDS[cmd][0]; module = importlib.import_module(f"flatocean.{mod}" if "." in mod else f"flatocean.analyses.{mod}")
    sys.argv = [f"python -m flatocean {cmd}"] + argv[1:]            # the analysis parses sys.argv exactly as the old script did
    result = module.main()
    return 0 if result is None else result


if __name__ == "__main__":
    sys.exit(main())
