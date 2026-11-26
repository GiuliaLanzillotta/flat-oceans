"""Along an LMC segment, do the units that keep their gate agree with A because they are selective?  For a spawned pair
(child A, child B) compute, per hidden unit / conv channel: (i) its activity fraction at A (fraction of sample-position
pairs on which it is active; selective = far from 1/2), and (ii) its gate agreement with A at each alpha (fraction of
sample-position pairs whose gate equals A's).  Writes results/paper_slices/selective_<model>.npz."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean import frozen_lib as FL
from flatocean import runs
import cnn_slices as CS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(runs.HERE, "results", "paper_slices"); os.makedirs(OUT, exist_ok=True)
ALPHAS = np.linspace(0, 1, 11)


@torch.no_grad()
def mlp_gates(net, X):
    _, Z, _ = net.forward(X, return_states=True)
    return [(z > 0) for z in Z]                                  # list of [S, n_l]


@torch.no_grad()
def cnn_gates(net, X):
    _, Z, _, _ = CS._states(net, X)
    return [(z > 0) for z in Z]                                  # list of [S, C_l, H, W]


def run(model):
    if model == "mlp":
        tA, tB = "lmc_s0_e250_c0", "lmc_s0_e250_c1"; gates = mlp_gates; flat = FL.flat_params; setf = FL.set_flat
        r = runs.load_run("mnist", "mlp", 0, tag=tA, epoch=500, device=DEV); net = r["net"]; X = r["data"]["analysis"][0]
        chdim = None                                             # per-unit
    else:
        tA, tB = "lmc_cnn_s0_e250_c0", "lmc_cnn_s0_e250_c1"; gates = cnn_gates; flat = CS.flat_params; setf = CS.set_flat
        r = runs.load_run("mnist", "cnn", 0, tag=tA, epoch=500, device=DEV); net = r["net"]; X = r["data"]["analysis"][0][:2000]
        chdim = (0, 2, 3)                                        # per-channel: mean over sample and space
    thA = flat(net).clone()
    thB = flat(runs.load_run("mnist", model, 0, tag=tB, epoch=500, device=DEV)["net"]).clone()
    gA = gates(net, X)
    if chdim is None:
        act = [g.float().mean(0).cpu().numpy() for g in gA]                     # [n_l]
    else:
        act = [g.float().mean(dim=chdim).cpu().numpy() for g in gA]             # [C_l]
    nblk = len(gA); agree = [np.zeros((len(ALPHAS), a.shape[0])) for a in act]
    for k, al in enumerate(ALPHAS):
        setf(net, thA + float(al) * (thB - thA)); g = gates(net, X)
        for l in range(nblk):
            eq = (g[l] == gA[l]).float()
            agree[l][k] = (eq.mean(0) if chdim is None else eq.mean(dim=chdim)).cpu().numpy()
        setf(net, thA)
    np.savez(os.path.join(OUT, f"selective_{model}.npz"), alphas=ALPHAS,
             **{f"act_{l}": act[l] for l in range(nblk)}, **{f"agree_{l}": agree[l] for l in range(nblk)}, nblk=nblk)
    print(f"wrote selective_{model}.npz  blocks={nblk} units={[a.shape[0] for a in act]}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(6)
    run("mlp"); run("cnn")
