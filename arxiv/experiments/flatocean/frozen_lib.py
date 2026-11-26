"""Frozen forward passes and the activation-transition residual.

Protocol: memos/2026-09-10_protocol_stages_atomic_transition.md, Sections 4
and 6 (amendment of 2026-09-10).  Everything in float64 on the analysis
representation.

Definitions.  At a checkpoint theta_0 with base gates S (the derivative
masks 1{z > 0} per example, unit, and position), the frozen forward
F^frz(w) applies the edited affine maps of theta_0 + w and the fixed masks S
in place of ReLU; hidden values are recomputed, derivatives are not.  With
    dF_true(w) = F(theta_0 + w) - F(theta_0),
    dF_frz(w)  = F^frz(w) - F^frz(0),
the activation-transition residual is A_tr(w) = dF_true(w) - dF_frz(w).

Independent route (theory file (7.2)-(7.3b)).  Let z~_l be the TRUE
preactivations at theta_0 + w and rho_l = ReLU(z~_l) - S_l z~_l the local
activation residual (zero at gates that kept their sign).  Then
    F(theta_0 + w) - F^frz(w) = sum_l D^post_l(w) rho_l,
where D^post_l(w) is the linear part of the frozen tail after layer l (edited
weights, base masks, no biases).  `residual_propagation` computes this sum
without any subtraction of full outputs; its discrepancy with the
subtraction route is the "identity residual" stored with every measurement.

Norms: ||V||_X = (s^-1 sum_i ||v_i||^2)^(1/2) over the stacked outputs, the
same scaling as J / sqrt(s).

Other activations (2026-09-22).  The frozen network is the linearization of every activation at its base
preactivation: h = sigma(z0) + sigma'(z0) (z - z0) =: S0 z + c0.  For ReLU and leaky ReLU c0 = 0 and S0 is the
gate (1{z0 > 0}, or 0.1/1), so this is the frozen-gate network above and it is exact until a gate changes; for
GELU S0 = sigma'(z0) and c0 = sigma(z0) - S0 z0, the frozen map is affine in each layer's edit (still multi-affine
across layers), and the residual rho_l = sigma(z~) - sigma(z0) - sigma'(z0)(z~ - z0) is the smooth curvature
residual, nonzero for every edit.  `base_masks` returns the sign masks 1{z0 > 0} as before (state diagnostics)
and caches S0 and c0 on the network for the smooth case; `gate` and `local_residual` are the single points where
the activation enters.
"""
import torch
import torch.nn.functional as F

from flatocean import tier1_train as T # ----------------------------------------------------------------------------- base masks and states
def activation_of(net):
    return getattr(net, "activation", "relu")


class Masks(list):
    """List of per-layer sign masks (bool, per sample) that also carries the base linearization of a smooth
    activation (`base` = {"S": [sigma'(z0)], "c": [sigma(z0) - sigma'(z0) z0]}, else None), so that masks
    computed on different sample sets never share state."""
    base = None


@torch.no_grad()
def base_masks(net, X, block=1000):
    """Sign masks 1{z_l > 0} of every gated layer on X (bool, per sample).  For a smooth activation the base
    linearization (S0 = sigma'(z0), c0 = sigma(z0) - S0 z0) is computed here as well and cached on the network."""
    masks = None; S0 = None; c0 = None; smooth = activation_of(net) not in ("relu", "leaky")
    for i in range(0, X.shape[0], block):
        _, Z, _ = net.forward(X[i:i + block], return_states=True)
        g = [(z > 0) for z in Z]
        masks = g if masks is None else [torch.cat([a, b]) for a, b in zip(masks, g)]
        if smooth:
            s_ = [net.act_prime(z) for z in Z]; c_ = [net.act(z) - sp * z for z, sp in zip(Z, s_)]
            S0 = s_ if S0 is None else [torch.cat([a, b]) for a, b in zip(S0, s_)]
            c0 = c_ if c0 is None else [torch.cat([a, b]) for a, b in zip(c0, c_)]
    masks = Masks(masks); masks.base = {"S": S0, "c": c0} if smooth else None
    return masks


def bn_affine(net, l):
    """Frozen eval-mode BatchNorm affine of gated MLP layer l applied to the PRE-BN preactivation z_l:
    u_l = a_l * z_l + off_l with a_l = gamma_l / sqrt(running_var_l + eps), off_l = beta_l - a_l * running_mean_l.
    Returns (a_l, off_l) as 1-D tensors, or (None, None) when the network has no BatchNorm.  The gate acts on u_l, so
    the frozen forward and the frozen tail must carry a_l (and off_l, in the affine forward) explicitly."""
    if not getattr(net, "batchnorm", False) or getattr(net, "bn", None) is None:
        return None, None
    m = net.bn[l]
    a = m.weight.detach() / torch.sqrt(m.running_var.detach() + m.eps)
    off = m.bias.detach() - a * m.running_mean.detach()
    return a, off


def gate(net, masks, l, idx, dtype):
    """(S0, c0) of gated layer l on samples idx: the slope and offset of the base linearization.
    ReLU: (mask, None); leaky ReLU: (0.1 or 1, None); smooth: cached (sigma'(z0), sigma(z0) - sigma'(z0) z0)."""
    act = activation_of(net)
    if act == "relu":
        return masks[l][idx].to(dtype), None
    if act == "leaky":
        m = masks[l][idx]
        return torch.where(m, torch.ones((), dtype=dtype, device=m.device), torch.full((), 0.1, dtype=dtype, device=m.device)), None
    base = getattr(masks, "base", None)
    if base is None:
        raise ValueError("smooth activation: the masks must come from frozen_lib.base_masks (they carry the base linearization)")
    return base["S"][l][idx], base["c"][l][idx]


def local_residual(net, masks, l, idx, z):
    """rho_l = sigma(z) - (S0 z + c0): the activation residual of the edited preactivation z against the base
    linearization (zero at unchanged gates for ReLU and leaky ReLU; the curvature residual for a smooth activation)."""
    S, c = gate(net, masks, l, idx, z.dtype)
    act = getattr(net, "act", torch.relu)
    lin = S * z if c is None else S * z + c
    return act(z) - lin


def x_norm(V):
    """||V||_X for V of shape (s, d): root mean over samples of the squared row norm."""
    return (V.double().pow(2).sum(1).mean()).sqrt().item()


# ----------------------------------------------------------------------------- frozen forward
@torch.no_grad()
def frozen_forward(net, masks, X, idx, return_states=False):
    """F^frz on samples idx: edited maps of `net`, fixed masks `masks` (indexed by idx).

    Returns the output (and, if asked, the masked hidden values H_frz per gated
    layer, i.e. the frozen postactivations).  Biases are included; pooling and
    the readout are the network's own linear maps."""
    x = X[idx]
    if isinstance(net, T.MLP):
        h = x; Hs = []
        for l, lin in enumerate(net.layers):
            z = lin(h)
            a, off = bn_affine(net, l)
            if a is not None: z = a * z + off                 # frozen eval-mode BN affine before the gate
            S, c = gate(net, masks, l, idx, z.dtype); h = S * z if c is None else S * z + c; Hs.append(h)
        out = net.readout(h)
    else:
        h = x; Hs = []; ci = 0
        for a in net.arch:
            if a[0] == "conv":
                z = net.convs[ci](h); h = z * masks[ci][idx].to(z.dtype); Hs.append(h); ci += 1
            else:
                h = F.avg_pool2d(h, 2)
        out = net.readout(h.flatten(1))
    return (out, Hs) if return_states else out


@torch.no_grad()
def frozen_tail_linear(net, masks, idx, l, v):
    """Linear part of the frozen tail after gated layer l applied to v (same
    shape as that layer's postactivation): edited weights, base masks, no
    biases.  Returns the output-space vector (b x d)."""
    if isinstance(net, T.MLP):
        h = v
        for k in range(l + 1, len(net.layers)):
            h = F.linear(h, net.layers[k].weight)
            a, _ = bn_affine(net, k)
            if a is not None: h = h * a                       # frozen BN slope (offset is constant -> not in the linear tail)
            h = h * gate(net, masks, k, idx, h.dtype)[0]
        return F.linear(h, net.readout.weight)
    # CNN: walk the architecture from the pooling/conv that follows gated layer l
    h = v; ci = 0; started = False
    for a in net.arch:
        if a[0] == "conv":
            if started:
                h = F.conv2d(h, net.convs[ci].weight, padding=1) * masks[ci][idx].to(h.dtype)
            if ci == l:
                started = True
            ci += 1
        else:
            if started:
                h = F.avg_pool2d(h, 2)
    return F.linear(h.flatten(1), net.readout.weight)


@torch.no_grad()
def true_states(net, X, idx):
    """True preactivations of `net` (already edited) on samples idx."""
    out, Z, _ = net.forward(X[idx], return_states=True)
    return out, Z


@torch.no_grad()
def residual_propagation(net, masks, X, idx):
    """sum_l D^post_l rho_l on samples idx, with rho_l from the true preactivations
    of the edited net and D^post_l the frozen linear tail.  Also returns the
    per-layer gate disagreement counts against the base masks."""
    out_true, Z = true_states(net, X, idx)
    total = torch.zeros_like(out_true); flips = []
    for l, z in enumerate(Z):
        S = masks[l][idx]
        rho = local_residual(net, masks, l, idx, z)
        flips.append(int(((z > 0) != S).sum()))
        if rho.abs().max() > 0:
            total = total + frozen_tail_linear(net, masks, idx, l, rho)
    return out_true, total, flips


# ----------------------------------------------------------------------------- direction scaling protocols (2026-09-22, plan Section 0.4 items 5-6)
def block_slices(net):
    """(name, start, end) of every W/b parameter block in the flat vector (order of net.parameters()).

    Each block's weight and bias are contiguous in that order.  BatchNorm affine parameters (gamma/beta) are NOT a
    block of the conditional analysis: they are held fixed, and in `net.parameters()` they come AFTER the readout while
    `param_groups()` bundles them into each hidden group -- so their sizes and positions must not enter the block
    slices.  The slice of each block is therefore taken from the ACTUAL flat positions of only its weight/bias tensors,
    excluding any BN parameters, giving correct contiguous [W_l; b_l] spans for both plain and BatchNorm models."""
    pos = {}; i = 0
    for p in net.parameters():
        pos[id(p)] = (i, i + p.numel()); i += p.numel()
    bn_ids = set()
    if getattr(net, "bn", None) is not None:
        for m in net.bn:
            for p in m.parameters():
                bn_ids.add(id(p))
    out = []
    for name, ps in net.param_groups():
        spans = [pos[id(p)] for p in ps if id(p) not in bn_ids]           # weight/bias only, at their true flat positions
        a = min(s for s, _ in spans); b = max(e for _, e in spans)
        assert b - a == sum(e - s for s, e in spans), f"block {name} W/b not contiguous in the flat vector"
        out.append((name, a, b))
    return out


def scale_direction(w, theta0, slices, mode):
    """Return (w', info) with w' the direction scaled for a unit step of the protocol:
    'block-balanced': every nonzero block component set to ||theta_l|| w_l / ||w_l|| (a step alpha then moves every
        perturbed block by the fraction alpha; membership in a layerwise direct sum is preserved);
    'weighted': the whole direction divided by its weighted relative norm sqrt(sum_l ||w_l||^2 / ||theta_l||^2)
        (for coupled subspaces such as ker J_X, whose membership independent block scaling can break);
    'absolute': the whole direction divided by its Euclidean norm.
    info records the protocol and the block fractions ||w'_l|| / ||theta_l|| of the unit step."""
    w = w.clone()
    if mode == "block-balanced":
        for name, a, b in slices:
            n = float(w[a:b].norm())
            if n > 0:
                w[a:b] *= float(theta0[a:b].norm()) / n
    elif mode == "weighted":
        rel = sum(float(w[a:b].norm() ** 2 / theta0[a:b].norm() ** 2) for _, a, b in slices) ** 0.5; w /= rel
    elif mode == "absolute":
        w /= w.norm()
    else:
        raise ValueError(mode)
    info = {"protocol": mode, "unit_step_euclidean_norm": float(w.norm()),
            "unit_step_block_fractions": {name: float(w[a:b].norm() / theta0[a:b].norm()) for name, a, b in slices}}
    return w, info


# ----------------------------------------------------------------------------- parameter edits
def flat_params(net):
    return torch.cat([p.reshape(-1) for p in net.parameters()])


@torch.no_grad()
def set_flat(net, vec):
    i = 0
    for p in net.parameters():
        n = p.numel(); p.copy_(vec[i:i + n].reshape(p.shape)); i += n


def param_slices(net):
    """(name, start, end) of every parameter tensor in the flat vector."""
    out = []; i = 0
    for name, p in net.named_parameters():
        out.append((name, i, i + p.numel())); i += p.numel()
    return out


# ----------------------------------------------------------------------------- one probe along one ray
@torch.no_grad()
def measure_ray(net, theta0, masks, X, u, r, F0, block=1000):
    """At theta_0 + r u: true drift, frozen drift, transition residual by
    subtraction and by propagation, identity residual, per-layer flips.
    All norms are ||.||_X (absolute); the caller stores them."""
    set_flat(net, theta0 + r * u)
    s = X.shape[0]
    dtrue = torch.zeros_like(F0); dfrz = torch.zeros_like(F0); prop = torch.zeros_like(F0)
    flips = None
    for i in range(0, s, block):
        idx = torch.arange(i, min(i + block, s), device=X.device)
        f_frz = frozen_forward(net, masks, X, idx)
        f_true, p, fl = residual_propagation(net, masks, X, idx)
        dtrue[idx] = f_true - F0[idx]; dfrz[idx] = f_frz - F0[idx]; prop[idx] = p
        flips = fl if flips is None else [a + b for a, b in zip(flips, fl)]
    set_flat(net, theta0)
    atr = dtrue - dfrz
    return {"true": x_norm(dtrue), "frozen": x_norm(dfrz), "transition": x_norm(atr),
            "transition_propagated": x_norm(prop), "identity_residual": x_norm(atr - prop),
            "flips": flips}, atr


# ----------------------------------------------------------------------------- validation (protocol Section 6)
TOL = {"base_point": 0.0, "base_point_smooth": 1e-12, "no_flip_transition_rel": 1e-12, "identity_rel": 1e-10, "single_layer_linearity_rel": 1e-11,
       "batch_size_agreement_rel": 1e-13}


def _validate(net, X, name, device, probe_seed=100):
    """Runs the declared checks on a float64 model; raises on failure, prints the measured errors."""
    theta0 = flat_params(net).clone(); p = theta0.numel()
    masks = base_masks(net, X)
    F0 = net(X)
    rep = {}
    # 1. base-point equality
    f = torch.cat([frozen_forward(net, masks, X, torch.arange(i, min(i + 500, X.shape[0]), device=device)) for i in range(0, X.shape[0], 500)])
    rep["base_point_max_abs"] = (f - F0).abs().max().item()
    # gated activations reproduce the base output bit for bit; the smooth linearization sigma(z0) + S0 z0 - S0 z0
    # carries one rounding of S0 z0, so it agrees to the float64 level of the outputs
    assert rep["base_point_max_abs"] <= (TOL["base_point"] if activation_of(net) in ("relu", "leaky") else TOL["base_point_smooth"]), rep
    # 2. batch-size agreement of the frozen forward
    f2 = torch.cat([frozen_forward(net, masks, X, torch.arange(i, min(i + 127, X.shape[0]), device=device)) for i in range(0, X.shape[0], 127)])
    rep["batch_agreement_rel"] = ((f2 - f).norm() / f.norm()).item()
    assert rep["batch_agreement_rel"] <= TOL["batch_size_agreement_rel"], rep
    g = torch.Generator().manual_seed(probe_seed)
    u = torch.randn(p, generator=g, dtype=torch.float64).to(device); u /= u.norm()
    smooth = activation_of(net) not in ("relu", "leaky")
    if not smooth:
        # 3. tiny edit with no gate change: transition residual at rounding level
        r = 1e-9
        m, _ = measure_ray(net, theta0, masks, X, u, r, F0)
        assert sum(m["flips"]) == 0, ("expected no flips at r=1e-9", m["flips"])
        rep["no_flip_transition_rel"] = m["transition"] / max(m["true"], 1e-300)
        rep["no_flip_true"] = m["true"]
        assert rep["no_flip_transition_rel"] <= TOL["no_flip_transition_rel"], rep
    else:
        # 3'. smooth activation: the curvature residual is second order in the edit
        ma, _ = measure_ray(net, theta0, masks, X, u, 1e-4, F0); mb, _ = measure_ray(net, theta0, masks, X, u, 1e-3, F0)
        rep["curvature_order"] = (mb["transition"] / ma["transition"]) / 10.0     # 10 for exactly quadratic
        assert 8.0 <= rep["curvature_order"] <= 12.0, rep
    # 4. generic multilayer edit: identity residual (propagation vs subtraction), flips present
    m1, _ = measure_ray(net, theta0, masks, X, u, 1.0, F0)
    rep["r1"] = m1
    rep["identity_rel_r1"] = m1["identity_residual"] / max(m1["true"], 1e-300)
    assert (smooth or sum(m1["flips"]) > 0) and rep["identity_rel_r1"] <= TOL["identity_rel"], rep
    m3, _ = measure_ray(net, theta0, masks, X, u, 3.0, F0)
    rep["identity_rel_r3"] = m3["identity_residual"] / max(m3["true"], 1e-300)
    assert rep["identity_rel_r3"] <= TOL["identity_rel"], rep
    # 5. the frozen map is exactly affine in a single-layer edit and not affine in a multilayer edit
    sl = param_slices(net)
    u1 = torch.zeros_like(u); a, b = sl[0][1], sl[0][2]; u1[a:b] = u[a:b]; u1 /= u1.norm()
    def dfrz(v, r):
        set_flat(net, theta0 + r * v)
        out = torch.cat([frozen_forward(net, masks, X, torch.arange(i, min(i + 500, X.shape[0]), device=device)) for i in range(0, X.shape[0], 500)]) - F0
        set_flat(net, theta0); return out
    d1, d2 = dfrz(u1, 1.0), dfrz(u1, 2.0)
    rep["single_layer_linearity_rel"] = ((d2 - 2 * d1).norm() / d2.norm()).item()
    assert rep["single_layer_linearity_rel"] <= TOL["single_layer_linearity_rel"], rep
    d1, d2 = dfrz(u, 1.0), dfrz(u, 2.0)
    rep["multilayer_nonlinearity_rel"] = ((d2 - 2 * d1).norm() / d2.norm()).item()
    assert rep["multilayer_nonlinearity_rel"] > 1e-3, ("frozen map looks linear in a multilayer edit: J w instead of the polynomial?", rep)
    # 6. exact branch control: outgoing weights of always-inactive units (feature-null) leave both outputs unchanged
    dead_dirs = []
    if isinstance(net, T.MLP):
        for l in range(len(net.layers) - 1):
            dead = (~masks[l]).all(0).nonzero().flatten()
            if dead.numel():
                nxt = f"layers.{l + 1}.weight"
                _, a, b = next(s for s in sl if s[0] == nxt)
                W = torch.zeros_like(net.layers[l + 1].weight); W[:, dead] = torch.randn(W.shape[0], dead.numel(), dtype=W.dtype, device=device)
                v = torch.zeros_like(u); v[a:b] = W.reshape(-1); v /= v.norm(); dead_dirs.append((l, dead.numel(), v))
    for l, nd, v in (dead_dirs[:1] if activation_of(net) == "relu" else []):
        mb, _ = measure_ray(net, theta0, masks, X, v, 10.0, F0)
        rep[f"feature_null_control_layer{l + 2}_dead{nd}"] = mb
        assert mb["true"] <= 1e-12 and mb["frozen"] <= 1e-12 and sum(mb["flips"]) == 0, mb
    print(f"[{name}] validation passed:", {k: (v if not isinstance(v, dict) else {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in rep.items()})
    return rep


def validate(device="cuda"):
    from flatocean import ocean_lib as O
    torch.set_num_threads(4)
    for tag in ("", "leaky", "gelu"):
        run = O.load_run("mnist", "mlp", 0, tag=tag, epoch=500, device=device)
        _validate(run["net"], run["data"]["analysis"][0][:2000], f"MNIST MLP {tag or 'relu'}, seed 0, epoch 500", device)
    run = O.load_run("mnist", "cnn", 0, epoch=500, device=device)
    _validate(run["net"], run["data"]["analysis"][0][:400], "MNIST CNN, seed 0, epoch 500", device)
    print("frozen_lib: all validation checks passed at the declared tolerances", TOL)


if __name__ == "__main__":
    validate()
