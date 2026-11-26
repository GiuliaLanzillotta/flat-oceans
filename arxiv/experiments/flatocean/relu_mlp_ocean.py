"""
Flat Ocean running example: depth-L ReLU MLP on cluster-gated data.

Stage 1 (this script):
  1. train a bias-free ReLU MLP on G Gaussian clusters (full-batch Adam);
  2. check that it freezes: gate flips per epoch, gate margins, dead units, feature ranks;
  3. at the final parameter theta0, compute the Level-1 (single-layer) quantities of the theory:
        H_{l-1}, r_h, backprop maps D_{l,i}, r_d, J_l = stack_i(h_i^T kron D_{l,i}),
        rank J_l, dim K_l, GPM dim, gain over GPM = r_h (n_l - r_d) + delta_joint,
        dim ker J_X (all layers);
  4. gate radius r_gate along random directions in K_l, verified on the TRUE network,
     compared with GPM directions, full ker J_X directions, and random directions.

Notation follows PLAN_concrete_rewrite.md:  f(x) = V S_L W_L ... S_1 W_1 x   (*)
"""
import argparse, json, math, os, time
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- data
def make_clusters(G, d_in, s, noise, seed):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(G, d_in, generator=g) * 2.0
    labels = torch.arange(s) % G
    X = centers[labels] + noise * torch.randn(s, d_in, generator=g)
    return X, labels, centers


# ----------------------------------------------------------------------------- model
class ReLUMLP(torch.nn.Module):
    """h_0 = x, z_l = W_l h_{l-1}, h_l = relu(z_l), f = V h_L.  No biases."""

    def __init__(self, d_in, widths, d_out, seed):
        super().__init__()
        torch.manual_seed(seed)
        dims = [d_in] + list(widths)
        self.W = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(dims[i + 1], dims[i]) * math.sqrt(2.0 / dims[i]))
             for i in range(len(widths))])
        self.V = torch.nn.Parameter(torch.randn(d_out, dims[-1]) * math.sqrt(1.0 / dims[-1]))
        self.L = len(widths)

    def forward(self, X, return_states=False):
        h = X
        Z, H = [], [X]
        for W in self.W:
            z = h @ W.T
            h = torch.relu(z)
            Z.append(z); H.append(h)
        out = h @ self.V.T
        return (out, Z, H) if return_states else out

    def gates(self, X):
        with torch.no_grad():
            _, Z, _ = self.forward(X, return_states=True)
        return [(z > 0) for z in Z]

    def flat_params(self):
        return torch.cat([W.detach().reshape(-1) for W in self.W] + [self.V.detach().reshape(-1)])

    def set_flat_params(self, vec):
        i = 0
        with torch.no_grad():
            for W in list(self.W) + [self.V]:
                n = W.numel()
                W.copy_(vec[i:i + n].reshape(W.shape)); i += n


def numrank(M, rtol=1e-9):
    if M.numel() == 0:
        return 0, torch.zeros(0)
    sv = torch.linalg.svdvals(M)
    if sv.numel() == 0 or sv[0] == 0:
        return 0, sv
    return int((sv > rtol * sv[0]).sum()), sv


# ----------------------------------------------------------------------------- training
def train(model, X, y, epochs, lr, log_every, final_gates=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    hist = []
    prev = model.gates(X)
    for ep in range(epochs + 1):
        out = model(X)
        loss = lossf(out, y)
        acc = (out.argmax(1) == y).double().mean().item()
        cur = model.gates(X)
        flips_prev = sum((a != b).sum().item() for a, b in zip(cur, prev)) / sum(a.numel() for a in cur)
        _, Z, H = model.forward(X, return_states=True)
        margin = min(z.abs().min().item() for z in Z)
        dead = [int(((z <= 0).all(0)).sum()) for z in Z]
        active_frac = [float((z > 0).double().mean()) for z in Z]
        ranks = [numrank(h.detach())[0] for h in H]  # H_0..H_L
        rec = dict(epoch=ep, loss=loss.item(), acc=acc, flip_frac_prev=flips_prev,
                   min_margin=margin, dead=dead, active_frac=active_frac, feat_rank=ranks)
        hist.append(rec)
        if ep % log_every == 0:
            print(f"ep {ep:5d} loss {loss.item():.3e} acc {acc:.3f} flips {flips_prev:.2e} "
                  f"min|z| {margin:.2e} dead {dead} rank(H) {ranks}")
        if ep == epochs:
            break
        opt.zero_grad(); loss.backward(); opt.step()
        prev = cur
    return hist


# ----------------------------------------------------------------------------- frozen structure at theta0
def frozen_structure(model, X):
    """Returns per-sample states and the exact downstream maps D_{l,i} = V S_L W_L ... W_{l+1} S_l."""
    with torch.no_grad():
        out, Z, H = model.forward(X, return_states=True)
    s = X.shape[0]; L = model.L; d = model.V.shape[0]
    S = [(z > 0).double() for z in Z]  # s x n_l
    W = [w.detach() for w in model.W]; V = model.V.detach()
    # D[l] : s x d x n_l,   D_{L,i} = V S_L^i ;  D_{l,i} = D_{l+1,i} W_{l+1} S_l^i
    D = [None] * L
    D[L - 1] = V.unsqueeze(0) * S[L - 1].unsqueeze(1)                       # V diag(S_L^i)
    for l in range(L - 2, -1, -1):
        D[l] = (D[l + 1] @ W[l + 1]) * S[l].unsqueeze(1)                    # (D_{l+1} W_{l+1}) diag(S_l)
    return dict(out=out, Z=Z, H=H, S=S, D=D, W=W, V=V, s=s, L=L, d=d)


def check_D_against_autograd(model, X, st, l, i):
    """D_{l,i} must equal d f(x_i) / d z_l  (autograd)."""
    x = X[i:i + 1]
    h = x
    for k, W in enumerate(model.W):
        z = h @ W.T
        if k == l:
            z = z.detach().requires_grad_(True); zl = z
        h = torch.relu(z)
    out = (h @ model.V.T)[0]
    J = torch.stack([torch.autograd.grad(out[c], zl, retain_graph=True)[0][0] for c in range(out.shape[0])])
    return (J - st["D"][l][i]).abs().max().item()


def layer_jacobian(st, l):
    """J_l : (s d) x (n_l * n_{l-1}) with rows  h_{l-1,i}^T kron D_{l,i}  (column-major vec of A)."""
    Hprev = st["H"][l]          # s x n_{l-1}   (H[0]=X)
    Dl = st["D"][l]             # s x d x n_l
    rows = [torch.kron(Hprev[i].unsqueeze(0), Dl[i]) for i in range(st["s"])]  # d x (n_{l-1} n_l)
    return torch.cat(rows, 0)


def readout_jacobian(st):
    HL = st["H"][st["L"]]
    d = st["d"]
    return torch.cat([torch.kron(HL[i].unsqueeze(0), torch.eye(d)) for i in range(st["s"])], 0)


def level1_table(st):
    L, s, d = st["L"], st["s"], st["d"]
    rows = []
    Jblocks = []
    for l in range(L):
        Hprev = st["H"][l]; Dl = st["D"][l]
        n_l, n_prev = Dl.shape[2], Hprev.shape[1]
        p_l = n_l * n_prev
        r_h, _ = numrank(Hprev)
        r_d, _ = numrank(Dl.reshape(s * d, n_l))
        Jl = layer_jacobian(st, l); Jblocks.append(Jl)
        rJ, svJ = numrank(Jl)
        dimK = p_l - rJ
        gpm = n_l * (n_prev - r_h)
        delta_joint = r_h * r_d - rJ
        rows.append(dict(layer=l + 1, n_in=n_prev, n_out=n_l, p_l=p_l, r_h=r_h, r_d=r_d,
                         rank_J=rJ, bound_sd=s * d, bound_rhrd=r_h * r_d, dimK=dimK, dimGPM=gpm,
                         gain=dimK - gpm, gain_marginal=r_h * (n_l - r_d), delta_joint=delta_joint,
                         sv_tail=[float(v) for v in svJ[max(0, rJ - 3):rJ + 3]]))
    # readout block
    HL = st["H"][L]; r_L, _ = numrank(HL); n_L = HL.shape[1]
    JV = readout_jacobian(st); Jblocks.append(JV)
    rV, _ = numrank(JV)
    rows.append(dict(layer="V", n_in=n_L, n_out=d, p_l=d * n_L, r_h=r_L, r_d=d, rank_J=rV,
                     bound_sd=s * d, bound_rhrd=r_L * d, dimK=d * n_L - rV, dimGPM=d * (n_L - r_L),
                     gain=(d * n_L - rV) - d * (n_L - r_L), gain_marginal=0, delta_joint=r_L * d - rV,
                     sv_tail=[]))
    JX = torch.cat(Jblocks, 1)
    rX, svX = numrank(JX)
    return rows, JX, rX


# ----------------------------------------------------------------------------- directions and radius
class NullProjector:
    """Orthogonal projector onto ker J via SVD (J is N x p); handles N > p or N < p."""

    def __init__(self, J, rtol=1e-9):
        U, sv, Vh = torch.linalg.svd(J, full_matrices=False)
        r = int((sv > rtol * sv[0]).sum()) if sv.numel() else 0
        self.Vr = Vh[:r].T                      # p x r, orthonormal basis of row space
        self.dim = J.shape[1] - r

    def __call__(self, r):
        return r - self.Vr @ (self.Vr.T @ r)


def block_slices(model):
    sl, i = [], 0
    for W in list(model.W) + [model.V]:
        sl.append((i, i + W.numel(), W.shape)); i += W.numel()
    return sl


def vec_col_to_mat(v, shape):
    """inverse of column-major vec: A = reshape(v, (n_in, n_out)).T"""
    n_out, n_in = shape
    return v.reshape(n_in, n_out).T


def colvec_to_param(u, slices):
    """convert a vector in the column-major convention of J_X (blockwise vec_col) to the
    row-major flat parameter convention of set_flat_params."""
    parts = []
    for (a, b, shape) in slices:
        parts.append(vec_col_to_mat(u[a:b], shape).reshape(-1))
    return torch.cat(parts)


def direction_in_layer_kernel(proj, shape, gen):
    r = torch.randn(shape[0] * shape[1], generator=gen)
    u = proj(r)
    A = vec_col_to_mat(u, shape)
    return A / A.norm()


def direction_gpm(st, l, shape, gen):
    """A with A h_{l-1,i} = 0 for all i: rows in the orthogonal complement of the feature span."""
    Hprev = st["H"][l]
    proj = NullProjector(Hprev)                      # ker of (s x n_prev) matrix = feature-null vectors
    if proj.dim == 0:
        return None
    A = torch.stack([proj(torch.randn(shape[1], generator=gen)) for _ in range(shape[0])])
    return A / A.norm()


def full_direction(model, layer_mats):
    """assemble a flat parameter direction from per-block matrices (None = zero)."""
    parts = []
    for W, A in zip(list(model.W) + [model.V], layer_mats):
        parts.append(torch.zeros_like(W).reshape(-1) if A is None else A.reshape(-1))
    return torch.cat(parts)


def gate_crossings(model, X, st, u_flat):
    """Distances |z| / |dz/dt| (dz/dt at t=0; exact when a single layer moves) for every
    sample-unit gate whose preactivation moves.  Returns sorted crossing distances."""
    theta0 = model.flat_params()
    eps = 1e-6
    model.set_flat_params(theta0 + eps * u_flat)
    with torch.no_grad():
        _, Zp, _ = model.forward(X, return_states=True)
    model.set_flat_params(theta0)
    ratios = []
    for z0, zp in zip(st["Z"], Zp):
        slope = (zp - z0) / eps
        mask = slope.abs() > 1e-9
        if mask.any():
            ratios.append((z0.abs() / slope.abs())[mask])
    if not ratios:
        return torch.tensor([float("inf")])
    return torch.sort(torch.cat(ratios)).values


def crossing_summary(cr):
    q = lambda p: cr[min(int(p * (cr.numel() - 1)), cr.numel() - 1)].item()
    return dict(n_moving=int(cr.numel()), first=cr[0].item(), q001=q(0.001), q01=q(0.01), q10=q(0.10), median=q(0.5))


def visible_radius(model, X, u_flat, ts, F0, rel_eps):
    """first t at which ||F(theta0+tu)-F(theta0)|| > rel_eps * ||F0||, per rel_eps."""
    vals, flips = scan_ray(model, X, u_flat, ts)
    out = {}
    for e in rel_eps:
        idx = np.nonzero(vals > e * F0)[0]
        out[e] = float(ts[idx[0]]) if idx.size else float("inf")
    return out, vals, flips


def scan_ray(model, X, u_flat, ts):
    theta0 = model.flat_params()
    with torch.no_grad():
        F0 = model(X)
    vals, flips = [], []
    g0 = model.gates(X)
    for t in ts:
        model.set_flat_params(theta0 + t * u_flat)
        with torch.no_grad():
            F = model(X)
        vals.append((F - F0).norm().item())
        flips.append(sum((a != b).sum().item() for a, b in zip(model.gates(X), g0)))
    model.set_flat_params(theta0)
    return np.array(vals), np.array(flips)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_in", type=int, default=20)
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 64, 64])
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--s", type=int, default=256)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_dirs", type=int, default=10)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, centers = make_clusters(args.G, args.d_in, args.s, args.noise, args.seed)
    model = ReLUMLP(args.d_in, args.widths, args.G, args.seed)
    p = sum(w.numel() for w in model.parameters())
    print(f"p = {p} parameters, N = s*d = {args.s * args.G}, widths {args.widths}, depth L = {len(args.widths)}")

    # ---- 1-2. train and check freezing
    t0 = time.time()
    hist = train(model, X, y, args.epochs, args.lr, log_every=max(1, args.epochs // 15))
    print(f"training done in {time.time() - t0:.1f}s")
    late = hist[-max(2, args.epochs // 10):]
    print(f"late-training gate flips per epoch (mean over last {len(late)} epochs): "
          f"{np.mean([h['flip_frac_prev'] for h in late]):.2e}")

    # ---- 3. Level-1 table
    st = frozen_structure(model, X)
    err = max(check_D_against_autograd(model, X, st, l, i) for l in range(st["L"]) for i in [0, 7, 100])
    print(f"max |D_{{l,i}} - autograd| = {err:.2e}")
    rows, JX, rX = level1_table(st)
    print("\nLevel-1 (single-layer) quantities:")
    hdr = ["layer", "p_l", "r_h", "r_d", "rank_J", "bound_sd", "bound_rhrd", "dimK", "dimGPM", "gain", "gain_marginal", "delta_joint"]
    print(" ".join(f"{h:>13}" for h in hdr))
    for r in rows:
        print(" ".join(f"{str(r[h]):>13}" for h in hdr))
    print(f"\nfull Jacobian: rank J_X = {rX}, dim ker J_X = {p - rX}  (p - N = {p - args.s * args.G})")
    print(f"sum_l dim K_l = {sum(r['dimK'] for r in rows)}   (direct sum is NOT a branch in general)")
    print(f"sum_l dim GPM_l = {sum(r['dimGPM'] for r in rows)}  (this IS an exact ocean, infinite radius)")

    # ---- 4. gate crossings and visible radius along kernel directions (true network)
    gen = torch.Generator().manual_seed(args.seed + 1)
    slices = block_slices(model)
    Wnorms = [W.norm().item() for W in st["W"]] + [st["V"].norm().item()]
    L = st["L"]
    F0 = st["out"].norm().item()
    Z_all = torch.cat([z.reshape(-1) for z in st["Z"]]).abs()
    print("\npreactivation magnitudes |z| over all sample-units: "
          + ", ".join(f"frac<{c:g}: {(Z_all < c).double().mean().item():.3f}" for c in [1e-6, 1e-3, 1e-2, 1e-1, 1.0]))
    t_max = 0.5 * max(Wnorms)
    ts = np.concatenate([[0.0], np.logspace(-4, np.log10(t_max), 60)])
    rel_eps = [1e-6, 1e-3, 1e-2]
    projs = [NullProjector(layer_jacobian(st, l)) for l in range(L)]
    radii = {}
    print(f"\nalong unit directions in K_l  (||W_l||_F ~ {np.mean(Wnorms[:-1]):.1f}, ||F0|| = {F0:.1f}):")
    print("   first gate crossing vs visible radius r_eps (first t with ||dF|| > eps*||F0||)")
    for l in range(L):
        if projs[l].dim == 0:
            print(f"  K_{l + 1}: empty"); continue
        shape = slices[l][2]
        recs = []
        for _ in range(args.n_dirs):
            A = direction_in_layer_kernel(projs[l], shape, gen)
            u = full_direction(model, [A if k == l else None for k in range(L + 1)])
            cr = crossing_summary(gate_crossings(model, X, st, u))
            vr, _, _ = visible_radius(model, X, u, ts, F0, rel_eps)
            recs.append(dict(cross=cr, visible=vr))
        radii[f"K_{l + 1}"] = recs
        med = lambda key: np.median([r["cross"][key] for r in recs])
        medv = lambda e: np.median([r["visible"][e] for r in recs])
        print(f"  K_{l + 1}: crossings: first {med('first'):.2e}  q1% {med('q01'):.2e}  q10% {med('q10'):.2e}  median {med('median'):.2e}"
              f"   | r_eps: 1e-6: {medv(1e-6):.2e}  1e-3: {medv(1e-3):.2e}  1e-2: {medv(1e-2):.2e}   (medians over {args.n_dirs} dirs)")

    # ---- ray scans: four kinds of direction
    lstar = max(range(L), key=lambda l: projs[l].dim)
    lgpm = max(range(L), key=lambda l: rows[l]["dimGPM"])
    dirs = {}
    A = direction_in_layer_kernel(projs[lstar], slices[lstar][2], gen)
    dirs[f"K_{lstar + 1} (single-layer kernel)"] = full_direction(model, [A if k == lstar else None for k in range(L + 1)])
    A = direction_gpm(st, lgpm, slices[lgpm][2], gen)
    if A is not None:
        dirs[f"GPM_{lgpm + 1} (feature-null)"] = full_direction(model, [A if k == lgpm else None for k in range(L + 1)])
    projX = NullProjector(JX)
    u = colvec_to_param(projX(torch.randn(p, generator=gen)), slices); dirs["ker J_X (all layers, first-order only)"] = u / u.norm()
    r = torch.randn(p, generator=gen); dirs["random"] = r / r.norm()
    scans = {}
    print(f"\nray scans, ||F(theta0 + t u) - F(theta0)|| / ||F0||:")
    for name, u in dirs.items():
        vr, vals, flips = visible_radius(model, X, u, ts, F0, rel_eps)
        cr = crossing_summary(gate_crossings(model, X, st, u))
        scans[name] = dict(t=ts.tolist(), dF=vals.tolist(), flips=flips.tolist(), cross=cr, visible=vr)
        idx = [10, 25, 40, 50, 60]
        print(f"  {name:40s} 1st crossing {cr['first']:.1e} | r_1e-3 {vr[1e-3]:.2e} | "
              + " ".join(f"t={ts[i]:.2g}:{vals[i] / F0:.1e}({flips[i]})" for i in idx))

    # ---- plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ep = [h["epoch"] for h in hist]
        ax[0].semilogy(ep, [max(h["flip_frac_prev"], 1e-7) for h in hist], label="gate flips / epoch")
        ax[0].semilogy(ep, [h["loss"] for h in hist], label="train loss")
        ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("freezing during training")
        for l in range(L + 1):
            ax[1].plot(ep, [h["feat_rank"][l] for h in hist], label=f"rank H_{l}")
        ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("feature ranks")
        for name, sc in scans.items():
            ax[2].plot(sc["t"][1:], sc["dF"][1:], label=name)
        ax[2].set_yscale("symlog", linthresh=1e-8); ax[2].set_xscale("log"); ax[2].set_xlabel("t (parameter distance)")
        ax[2].set_ylabel("||F(theta0+tu)-F(theta0)||"); ax[2].legend(fontsize=7); ax[2].set_title("true-network flatness along rays")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"stage1_{args.tag}.png"), dpi=130)
        print(f"\nfigure saved to {os.path.join(args.out, f'stage1_{args.tag}.png')}")
    except Exception as e:
        print("plotting failed:", e)

    json.dump(dict(args=vars(args), p=p, hist=hist, level1=rows, rank_JX=rX, radii=radii, scans=scans),
              open(os.path.join(args.out, f"stage1_{args.tag}.json"), "w"), indent=1)
    torch.save(dict(state=model.state_dict(), X=X, y=y, centers=centers, args=vars(args)),
               os.path.join(args.out, f"model_stage1_{args.tag}.pt"))
    print(f"results saved to {args.out}")


if __name__ == "__main__":
    main()
