"""The Jacobian block of one layer behind one interface, for the MLP and the tied CNN.

Conventions (theory file (3.6), (3.10)).  MLP layer l with weight W (n x m) and bias b: one parameter block A = [W b] of
shape n x (m+1), vectorized column-major, entry (j, k) at index k*n + j.  Rows of J_l are indexed by (sample i, output a):
row (i, a) = [h_i; 1]^T kron (D_i^T e_a), so J_l vec(A) = stack_i(D_i A [h_i; 1]); D_i = D_{l,i} is the downstream map from
the layer-l preactivation to the output, including the gate S_{l,i}.  The readout block [V b_V] (d x (n_L+1)) has index
k*d + c.  Tied conv block (cnn_jacobian): column index k*C + c with k the patch coordinate (ci*9 + kh*3 + kw, then the bias)
and c the output channel; the row of a sample sums the patch contributions over positions before it is formed.

`Block` exposes what the analyses need without materializing one dense downstream tensor for every architecture:
    p                 number of columns
    rows(idx)         (b d) x p rows for the samples idx (a torch index tensor on the block's device)
    unit_groups()     list of column-index tensors, one per unit (MLP: the m+1 columns of row j of A; CNN: the K columns of
                      filter c), the groups of the neuron-wise / channel-wise decompositions
    n_units, unit_kind ("unit" or "channel"), name
"""
import numpy as np
import torch


# ----------------------------------------------------------------------------- MLP per-sample structure
@torch.no_grad()
def bn_slopes(net):
    """Eval-mode BatchNorm slope a_l = gamma_l / sqrt(running_var_l + eps) per hidden layer (double), or None.
    In evaluation mode BN is the frozen diagonal affine map u_l = a_l * z_l + (beta_l - a_l * running_mean_l) applied to
    the pre-BN preactivation z_l = W_l h_{l-1} + b_l; a_l is the slope that carries a parameter/preactivation perturbation
    through BN.  A zero slope (gamma_l = 0) makes that feature constant, so its layer-l parameter directions do not reach
    the output (a full kernel direction) -- handled by folding a_l (the zero passes through to D)."""
    bn = getattr(net, "bn", None)
    if not getattr(net, "batchnorm", False) or bn is None:
        return None
    return [(m.weight.double() / torch.sqrt(m.running_var.double() + m.eps)) for m in bn]


@torch.no_grad()
def mlp_structure(net, X):
    """Per-sample states of an MLP with biases.  Returns a dict with
    H[l]  : s x (n_{l-1}+1) augmented input of gated layer l (l = 0..L-1), plus H[L] for the readout,
    Z[l]  : s x n_l preactivations (POST-BN when BatchNorm is present -- the value the gate acts on),
    S[l]  : gates sigma'(z_l) (double; the gate for ReLU, the derivative for a smooth activation),
    D[l]  : s x d x n_l downstream map from the layer-l PRE-BN preactivation z_l (what the layer-l parameters produce),
            including the frozen BN slope a_l of this layer and of every downstream layer, and the gate S_l, out, L, d, activation.
    With BatchNorm in eval mode the block Jacobian is thus reported in the ORIGINAL affine coordinates [W_l b_l]: folding
    a_l into W is an invertible reparametrization when a_l != 0 (rank preserved), but the kernel coordinates, angles and
    Euclidean radii live in these original coordinates, so a_l must be carried explicitly rather than absorbed into W."""
    out, Z, Hin = net.forward(X, return_states=True)
    L = len(Z)
    prime = getattr(net, "act_prime", None) or (lambda z: (z > 0).to(z.dtype))
    S = [prime(z).double() for z in Z]                          # gate on the POST-BN preactivation Z[l]
    Ws = [lin.weight for lin in net.layers]; V = net.readout.weight
    a = bn_slopes(net)                                          # a[l] : n_l  (eval-mode BN slope) or None
    M = [None] * L                                              # M[l] = dF/d u_l  (post-BN preactivation -> output)
    M[L - 1] = V.unsqueeze(0) * S[L - 1].unsqueeze(1)           # readout has no BN
    for l in range(L - 2, -1, -1):
        Weff = Ws[l + 1] if a is None else a[l + 1].unsqueeze(1) * Ws[l + 1]   # diag(a_{l+1}) W_{l+1}
        M[l] = (M[l + 1] @ Weff) * S[l].unsqueeze(1)
    # D[l] = dF/d z_l^pre = M[l] @ diag(a_l): the layer-l parameters produce the pre-BN preactivation z_l.
    D = M if a is None else [M[l] * a[l].view(1, 1, -1) for l in range(L)]
    ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
    H = [torch.cat([h, ones], 1) for h in Hin]
    H.append(torch.cat([getattr(net, "act", torch.relu)(Z[-1]), ones], 1))
    return dict(out=out, Z=Z, S=S, H=H, D=D, L=L, d=V.shape[0], activation=getattr(net, "activation", "relu"))


def mlp_rows(st, l, idx):
    """Rows (i, a) of J_l for samples idx: (b d) x n(m+1), index k*n + j."""
    h = st["H"][l][idx]; Dl = st["D"][l][idx]; b, d, n = Dl.shape
    return torch.einsum("ik,iaj->iakj", h, Dl).reshape(b * d, h.shape[1] * n)


def readout_rows(st, idx):
    """Rows of the readout block [V b_V] (d x (n_L+1)): (b d) x d(n_L+1), index k*d + c."""
    h = st["H"][st["L"]][idx]; b, m1 = h.shape; d = st["d"]
    eye = torch.eye(d, dtype=h.dtype, device=h.device)
    return torch.einsum("ik,ac->iakc", h, eye).reshape(b * d, m1 * d)


def unit_states(st, l, s):
    """Unit-level states of gated layer l on the first s samples (dead units: derivative exactly zero on every sample)."""
    S = st["S"][l][:s]; Z = st["Z"][l][:s]; act = st.get("activation", "relu")
    return {"activation": act, "dead_units": int((S.sum(0) == 0).sum()),
            "dead_units_meaning": "zero feature (ReLU)" if act == "relu" else "derivative underflows to zero on every sample; numerical, not analytic",
            "units_never_positive": int((Z.max(0).values <= 0).sum()),
            "units_derivative_below_1e-9": int((S.abs() < 1e-9).all(0).sum().item()),
            "exact_zero_derivatives": int((S == 0).sum())}


# ----------------------------------------------------------------------------- the block interface
class MLPLayerBlock:
    """Hidden layer l (0-based) of an MLP; rows from the stored per-sample structure `st`."""
    unit_kind = "unit"

    def __init__(self, st, l, s=None):
        self.st = st; self.l = l; self.s = st["H"][l].shape[0] if s is None else s
        self.H = st["H"][l][: self.s]; self.D = st["D"][l][: self.s]; self.Z = st["Z"][l][: self.s]
        self.m1 = self.H.shape[1]; self.d, self.n = self.D.shape[1], self.D.shape[2]; self.p = self.n * self.m1
        self.n_units = self.n; self.name = str(l + 1); self.device = self.H.device

    def rows(self, idx):
        return mlp_rows(self.st, self.l, idx)

    def unit_groups(self):
        """Columns of row j of A = [W b]: indices k*n + j for k = 0..m."""
        return [torch.arange(self.m1, device=self.device) * self.n + j for j in range(self.n)]

    def unit_rows(self, j, Hd=None):
        """The (s d) x (m+1) block J_lj of unit j: a_j -> ((D_i e_j)(a_j^T h_i))_i.  Hd: precomputed H.repeat_interleave(d, 0)."""
        Hd = self.H.repeat_interleave(self.d, 0) if Hd is None else Hd
        return Hd * self.D[:, :, j].reshape(self.s * self.d, 1)


class MLPReadoutBlock:
    unit_kind = "unit"

    def __init__(self, st, s=None):
        self.st = st; self.s = st["H"][st["L"]].shape[0] if s is None else s
        self.H = st["H"][st["L"]][: self.s]; self.m1 = self.H.shape[1]; self.d = st["d"]; self.p = self.m1 * self.d
        self.n_units = self.d; self.name = "readout"; self.device = self.H.device

    def rows(self, idx):
        return readout_rows(self.st, idx)

    def unit_groups(self):
        return [torch.arange(self.m1, device=self.device) * self.d + c for c in range(self.d)]


class CNNBlock:
    """Tied conv block "convN" or the readout of the MNIST/CIFAR CNN (cnn_jacobian coordinates, masks fixed at the base point)."""
    unit_kind = "channel"

    def __init__(self, net, masks, X, blk):
        from flatocean import cnn_jacobian as CJ
        self.net, self.masks, self.X, self.blk = net, masks, X, str(blk); self.s = X.shape[0]; self.device = X.device
        self.p = CJ.block_size(net, self.blk); self.name = self.blk; self._CJ = CJ
        if self.blk == "readout":
            self.C = net.readout.out_features; self.K = net.readout.in_features + 1
        else:
            conv = net.convs[int(self.blk[4:]) - 1]; self.C, self.K = conv.out_channels, 9 * conv.in_channels + 1
        self.n_units = self.C

    def rows(self, idx):
        return self._CJ.cnn_rows(self.net, self.masks, self.X, idx, self.blk)

    def unit_groups(self):
        """Columns of filter c: indices k*C + c for k = 0..K-1."""
        return [torch.arange(self.K, device=self.device) * self.C + c for c in range(self.C)]


def make_block(run, blk, device, st=None, masks=None):
    """Block by name for a loaded run: MLP "1", "2", ..., "readout" (needs st) or CNN "conv1", ..., "readout" (needs masks)."""
    from flatocean import tier1_train as T
    net = run["net"]; X = run["data"]["analysis"][0]
    if isinstance(net, T.MLP):
        st = mlp_structure(net, X) if st is None else st
        return MLPReadoutBlock(st) if str(blk) == "readout" else MLPLayerBlock(st, int(blk) - 1)
    from flatocean import frozen_lib as FL
    masks = FL.base_masks(net, X) if masks is None else masks
    return CNNBlock(net, masks, X, blk)
