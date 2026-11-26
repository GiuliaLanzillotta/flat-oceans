"""2D loss and function landscape of an LMC pair, for several spawn epochs.  Plane through the two sibling endpoints
A, B (children of one spawn, epoch 500) and their common spawn parent P (the base model at the spawn epoch).  Over a grid
on that plane, with the ReLU gates fixed at A, evaluate MSE loss, function change ||F-F_A||_X, frozen part C and
activation transition T.  Also project the two real training trajectories P->A and P->B onto the plane.
Writes results/paper_slices/landscape_<model>_e<E>.npz.  Run: CUDA_VISIBLE_DEVICES=0 python paper_figures/lmc_landscape_compute.py"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flatocean import frozen_lib as FL
from flatocean import runs
import cnn_slices as CS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(runs.HERE, "results", "paper_slices"); os.makedirs(OUT, exist_ok=True)
SPEC = {"mlp": (["", "lmc_s0_e{E}_c0", "lmc_s0_e{E}_c1"], FL.flat_params, FL.set_flat, [100, 150, 250], 55, 45, None),
        "cnn": (["", "lmc_cnn_s0_e{E}_c0", "lmc_cnn_s0_e{E}_c1"], CS.flat_params, CS.set_flat, [100, 150, 250], 45, 37, 2000)}


def xn(V):
    return float(V.pow(2).sum(1).mean().sqrt())


def onehot(y, d):
    o = torch.zeros(y.shape[0], d, dtype=torch.float64, device=y.device); o[torch.arange(y.shape[0]), y] = 1.0; return o


def load_net_data(model, tag, epoch, nimg):
    r = runs.load_run("mnist", model, 0, tag=tag, epoch=epoch, device=DEV)
    X, Y = r["data"]["analysis"][0], r["data"]["analysis"][1]
    if nimg:
        X, Y = X[:nimg], Y[:nimg]
    return r, X, Y


def traj_coords(net, flat, setf, cdir, thA, e1, e2):
    ck = sorted(int(f[6:10]) for f in os.listdir(cdir) if f.startswith("epoch_") and "recovery" not in f)
    xs, ys = [], []
    for e in ck:
        st = torch.load(os.path.join(cdir, f"epoch_{e:04d}.pt"), map_location=DEV, weights_only=False)["model"]
        net.load_state_dict({k: v.double() for k, v in st.items()})
        th = flat(net)
        xs.append(float((th - thA) @ e1)); ys.append(float((th - thA) @ e2))
    return np.array(xs), np.array(ys)


def fwd_fns(model, net, X, masks):
    idx = torch.arange(X.shape[0], device=DEV)
    if model == "mlp":
        @torch.no_grad()
        def fwd(theta, thA, setf):
            setf(net, theta); f = net(X); frz = FL.frozen_forward(net, masks, X, idx); setf(net, thA); return f, frz
        return fwd
    @torch.no_grad()
    def fwd(theta, thA, setf):
        CS.set_flat(net, theta); out, _, _, _ = CS._states(net, X); frz = CS.frozen_forward(net, masks, X, idx); CS.set_flat(net, thA); return out, frz
    return fwd


def run(model, E):
    tags, flat, setf, _, nx, ny, nimg = SPEC[model]
    rA, X, Y = load_net_data(model, tags[1].format(E=E), 500, nimg); net = rA["net"]
    thA = flat(net).clone()
    thB = flat(load_net_data(model, tags[2].format(E=E), 500, nimg)[0]["net"]).clone()
    thP = flat(load_net_data(model, "", E, nimg)[0]["net"]).clone()
    masks = FL.base_masks(net, X) if model == "mlp" else CS.base_masks(net, X)
    with torch.no_grad():
        F_A = net(X) if model == "mlp" else CS._states(net, X)[0]
    d = F_A.shape[1]; oh = onehot(Y, d)
    u = thB - thA; R = float(u.norm()); e1 = u / R
    w = thP - thA; proj = float(w @ e1); vp = w - proj * e1; nv = float(vp.norm()); e2 = vp / nv
    # trajectories (reload checkpoints into net, then restore A)
    tAx, tAy = traj_coords(net, flat, setf, os.path.join(rA["dir"], "checkpoints"), thA, e1, e2)
    tBx, tBy = traj_coords(net, flat, setf, os.path.join(load_net_data(model, tags[2].format(E=E), 500, nimg)[0]["dir"], "checkpoints"), thA, e1, e2)
    setf(net, thA)
    fwd = fwd_fns(model, net, X, masks)
    px = np.array([0.0, R, proj]); py = np.array([0.0, 0.0, nv]); mx = 0.16 * R      # frame the A,B,P triangle
    xs = np.linspace(px.min() - mx, px.max() + mx, nx); ys = np.linspace(py.min() - 0.30 * R, py.max() + mx, ny)
    fields = {k: np.zeros((ny, nx)) for k in ("mse", "err", "fun", "froz", "trans")}
    t0 = time.time()
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            f, frz = fwd(thA + float(xs[ix]) * e1 + float(y) * e2, thA, setf)
            fields["mse"][iy, ix] = float(((f - oh) ** 2).mean()); fields["err"][iy, ix] = float((f.argmax(1) != Y).float().mean())
            fields["fun"][iy, ix] = xn(f - F_A); fields["froz"][iy, ix] = xn(frz - F_A); fields["trans"][iy, ix] = xn(f - frz)
    np.savez(os.path.join(OUT, f"landscape_{model}_e{E}.npz"), xs=xs, ys=ys, R=R, Px=proj, Py=nv, spawn=E,
             tAx=tAx, tAy=tAy, tBx=tBx, tBy=tBy, F_A_norm=xn(F_A), **fields)
    print(f"  {model} e{E}: R={R:.1f} P=({proj:.1f},{nv:.1f}) grid {nx}x{ny} {time.time()-t0:.0f}s", flush=True)


def main():
    torch.set_num_threads(6)
    for model in ("mlp", "cnn"):
        for E in SPEC[model][3]:
            run(model, E)


if __name__ == "__main__":
    main()
