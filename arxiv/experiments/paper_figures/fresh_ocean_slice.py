"""A fresh, reproducible inactive-neuron plane for the Figure 1 selection.

Same w64 checkpoint, analysis examples and physical grid as the gallery.
For a last-hidden-layer row edit, the true response can be evaluated exactly
from cached incoming features, its ReLU preactivation and the readout column.
Check that reduction against original-network forwards before saving.
"""
import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from flatocean import frozen_lib as FL
from flatocean import runs


@torch.no_grad()
def shared_window(ggn=False):
    """Remeasure generic/symmetry controls on option 03's shifted window."""
    import sym_compute as SY
    torch.set_num_threads(4)
    run = runs.load_run('mnist', 'mlp', 0, tag='w64', epoch=305, device='cpu')
    net = run['net']
    X, _, indices = run['data']['analysis']
    F0, Z, _ = net(X, return_states=True)
    theta0 = FL.flat_params(net).clone()
    root = Path(runs.RESULTS_ROOT) / 'paper_slices'
    reference = np.load(root / 'hero_fresh24/ocean_03_centered.npz')
    assert np.array_equal(indices, reference['analysis_indices'])
    # Include the reference point exactly even though it is off the window center.
    x = np.unique(np.r_[np.linspace(*reference['x_axis'][[0, -1]], 81), 0.0])
    y = np.unique(np.r_[np.linspace(*reference['y_axis'][[0, -1]], 81), 0.0])
    aa, bb = np.meshgrid(x, y, indexing='ij')
    coordinates = torch.tensor(np.stack([aa.ravel(), bb.ravel()], axis=1))
    g = torch.Generator().manual_seed(11)
    u = torch.randn(theta0.numel(), generator=g, dtype=torch.float64); u /= u.norm()
    v = torch.randn(theta0.numel(), generator=g, dtype=torch.float64)
    v -= (v @ u) * u; v /= v.norm()
    ei, eo, ni, no = SY.build_rescaling(net, theta0, SY.J)
    families = [('generic', u, v, {}),
                ('symmetry', ei / ni, eo / no, dict(norm_in=ni, norm_out=no))]
    slices = FL.param_slices(net)
    if ggn:
        kp = Path(runs.RESULTS_ROOT) / 'branches/mnist_mlp_w64_seed0_epoch0305_kerJX.pt'
        saved = torch.load(kp, map_location='cpu', weights_only=False)
        basis = saved['basis'].double()
        singular = np.load(kp.with_name('mnist_mlp_w64_seed0_epoch0305_JX_singular_values.npy'))
        threshold = 1e-9 * float(singular[0])
        assert basis.shape[1] == basis.shape[0] - int((singular > threshold).sum())
        g = torch.Generator().manual_seed(2026112501)
        plane, _ = torch.linalg.qr(basis @ torch.randn(basis.shape[1], 2, generator=g, dtype=torch.float64))
        # Saved basis uses row-major augmented [W|b] matrices, not PyTorch order.
        parts = []; offset = 0
        for layer in list(net.layers) + [net.readout]:
            n, m = layer.weight.shape
            block = plane[offset:offset+n*(m+1)].reshape(n, m+1, 2)
            parts.extend([block[:, :-1].reshape(-1, 2), block[:, -1]])
            offset += n * (m+1)
        mapped = torch.cat(parts)
        assert offset == len(theta0)
        assert torch.linalg.matrix_norm(mapped.T @ mapped - torch.eye(2)) < 1e-12
        parameters = {k: p.detach().clone() for k, p in net.named_parameters()}
        rates = []
        for vdir in mapped.T:
            tangent = {k: vdir[a:b].reshape(parameters[k].shape) for k, a, b in slices}
            _, jv = torch.func.jvp(lambda p: torch.func.functional_call(net, p, (X,)),
                                  (parameters,), (tangent,))
            rates.append(float(jv.square().sum(1).mean().sqrt()))
        rate_cut = threshold / np.sqrt(len(X))
        assert max(rates) <= rate_cut * 1.01 + 1e-13
        record = dict(kernel_dimension=basis.shape[1], parameters=basis.shape[0],
                      relative_J_singular_cut=1e-9, absolute_J_rate_cut=rate_cut,
                      sampled_JVP_rates=rates, seed=2026112501,
                      sampling='Gaussian coefficients in full saved numerical kernel; no ocean removal',
                      basis_identity_digest=saved.get('identity_digest'),
                      basis_sha256=hashlib.sha256(kp.read_bytes()).hexdigest(),
                      coordinate_map='augmented row-major [W|b] blocks to PyTorch weight-then-bias order',
                      loss='MSE; G = 2 J^T J / N')
        (root / 'hero_fresh24/shared_ggn_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2), flush=True)
        families = [('ggn', mapped[:, 0], mapped[:, 1], dict(jvp_rates=np.asarray(rates)))]
        del basis, plane, saved
    for name, u, v, extra in families:
        U = {key: u[a:b].reshape(dict(net.named_parameters())[key].shape) for key, a, b in slices}
        V = {key: v[a:b].reshape(dict(net.named_parameters())[key].shape) for key, a, b in slices}
        zu = X @ U['layers.0.weight'].T + U['layers.0.bias']
        zv = X @ V['layers.0.weight'].T + V['layers.0.bias']

        def forward(ab):
            alpha, beta = ab[:, 0, None, None], ab[:, 1, None, None]
            h = (Z[0][None] + alpha * zu[None] + beta * zv[None]).relu()
            for key, layer, activate in [(f'layers.{i}', net.layers[i], True)
                                         for i in range(1, len(net.layers))] + [('readout', net.readout, False)]:
                w = layer.weight[None] + alpha * U[key+'.weight'][None] + beta * V[key+'.weight'][None]
                b = layer.bias[None, None] + alpha * U[key+'.bias'][None, None] + beta * V[key+'.bias'][None, None]
                h = torch.bmm(h, w.transpose(1, 2)) + b
                if activate:
                    h = h.relu()
            return h

        errors = []
        checks = torch.tensor([[0., 0.], [x[0], y[0]], [x[-1], y[-1]], [1.5, 1.5]], dtype=torch.float64)
        predicted = forward(checks)
        for k, ab in enumerate(checks):
            FL.set_flat(net, theta0 + ab[0] * u + ab[1] * v)
            actual = net(X)
            errors.append(float((actual - predicted[k]).square().sum(1).mean().sqrt()))
        FL.set_flat(net, theta0)
        assert max(errors) < 1e-12
        print(name, 'original-network vector error', max(errors), flush=True)
        values = []
        for j, chunk in enumerate(coordinates.split(8)):
            values.append((forward(chunk) - F0[None]).square().sum(2).mean(1).sqrt())
            if j % 100 == 0:
                print(name, 'points', min((j+1)*8, len(coordinates)), '/', len(coordinates), flush=True)
        cart = torch.cat(values).reshape(len(x), len(y)).numpy()
        np.savez(root / f'hero_fresh24/shared_{name}.npz', x_axis=x, y_axis=y,
                 cart_true=cart, directions=torch.stack([u, v], 1).numpy(),
                 theta_norm=float(theta0.norm()), original_network_vector_error_max=max(errors),
                 analysis_indices=indices, **extra)
        print(name, 'saved', flush=True)


@torch.no_grad()
def centered(numbers):
    """Measure shifted windows enclosing the full inactive-region polygon."""
    from scipy.spatial import HalfspaceIntersection, ConvexHull
    torch.set_num_threads(4)
    run = runs.load_run('mnist', 'mlp', 0, tag='w64', epoch=305, device='cpu')
    net = run['net']
    X, _, indices = run['data']['analysis']
    F0, Z, H = net(X, return_states=True)
    features = torch.cat([H[-1], torch.ones(len(X), 1, dtype=X.dtype)], dim=1)
    directory = Path(runs.RESULTS_ROOT) / 'paper_slices/hero_fresh24'
    specs = []
    for number in numbers:
        old = np.load(directory / f'ocean_{number:02d}.npz')
        assert np.array_equal(indices, old['analysis_indices'])
        neuron = int(old['neuron'])
        directions = torch.from_numpy(old['directions'])
        projected = features @ directions
        baseline = Z[-1][:, neuron]
        assert baseline.max() < 0
        constraints = np.column_stack([projected.numpy(), baseline.numpy()])
        vertices = HalfspaceIntersection(constraints, np.zeros(2)).intersections
        vertices = vertices[ConvexHull(vertices).vertices]
        low, high = vertices.min(0), vertices.max(0)
        center = (low + high) / 2
        assert np.max(constraints[:, :2] @ center + constraints[:, 2]) <= 1e-12
        specs.append((number, neuron, directions, projected, baseline, center, vertices))
    # Both previews share one physical scale; show a margin around both polygons.
    width = max(3.0, max(np.ptp(s[-1], axis=0).max() * 1.3 for s in specs))
    for number, neuron, directions, projected, baseline, center, vertices in specs:
        axis = np.linspace(-width / 2, width / 2, 151)
        aa, bb = np.meshgrid(axis + center[0], axis + center[1], indexing='ij')
        coordinates = torch.tensor(np.stack([aa.ravel(), bb.ravel()], axis=1))
        column = net.readout.weight[:, neuron].clone()
        responses = []
        for chunk in coordinates.split(128):
            responses.append((baseline[:, None] + projected @ chunk.T).relu()
                             .square().mean(0).sqrt() * column.norm())
        cart = torch.cat(responses).reshape(151, 151).numpy()
        row = net.layers[-1].weight[neuron].clone()
        bias = net.layers[-1].bias[neuron].clone()
        error = []
        for ab in [center, np.array([aa[0, 0], bb[0, 0]]),
                   np.array([aa[-1, -1], bb[-1, -1]]), np.zeros(2)]:
            ab = torch.tensor(ab)
            edit = directions @ ab
            net.layers[-1].weight[neuron].copy_(row + edit[:-1])
            net.layers[-1].bias[neuron].copy_(bias + edit[-1])
            predicted = (baseline + projected @ ab).relu()[:, None] * column[None, :]
            actual = net(X) - F0
            error.append(float((actual - predicted).square().sum(1).mean().sqrt()))
        net.layers[-1].weight[neuron].copy_(row)
        net.layers[-1].bias[neuron].copy_(bias)
        assert max(error) < 1e-12
        np.savez(directory / f'ocean_{number:02d}_centered.npz',
                 axis=axis, x_axis=axis + center[0], y_axis=axis + center[1],
                 center_offset=center, inactive_polygon=vertices, cart_true=cart,
                 directions=directions.numpy(), neuron=neuron, analysis_indices=indices,
                 theta_norm=float(FL.flat_params(net).norm()),
                 original_network_vector_error_max=max(error))
        print(json.dumps(dict(number=number, center_offset=center.tolist(),
                              width=width, validation_error=max(error))), flush=True)


@torch.no_grad()
def batch(count=24):
    """Fresh planes across inactive neurons; all cells retained, no shape filtering."""
    torch.set_num_threads(4)
    run = runs.load_run('mnist', 'mlp', 0, tag='w64', epoch=305, device='cpu')
    net = run['net']
    assert net.activation == 'relu' and not net.batchnorm
    X, _, indices = run['data']['analysis']
    F0, Z, H = net(X, return_states=True)
    inactive = torch.where((Z[-1] <= 0).all(dim=0))[0].tolist()
    assert inactive
    features = torch.cat([H[-1], torch.ones(len(X), 1, dtype=X.dtype)], dim=1)
    masks = FL.base_masks(net, X)
    axis = np.linspace(-1.5, 1.5, 101)
    aa, bb = np.meshgrid(axis, axis, indexing='ij')
    coordinates = torch.tensor(np.stack([aa.ravel(), bb.ravel()], axis=1))
    directory = Path(runs.RESULTS_ROOT) / 'paper_slices/hero_fresh24'
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(run['dir']) / 'checkpoints/epoch_0305.pt'
    manifest = dict(checkpoint=str(checkpoint),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    analysis_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                    examples=len(X), inactive_neurons=inactive, grid=[101, 101],
                    coordinate_range=[-1.5, 1.5],
                    selection='cycle inactive neurons in index order; fixed independent seeds; retain all planes',
                    method='exact last-hidden-row ReLU/readout reduction; original-network vector checks per plane',
                    planes=[])
    parameter_norm = float(FL.flat_params(net).norm())
    for number in range(1, count + 1):
        neuron = inactive[(number - 1) % len(inactive)]
        seed = 2026112400 + number
        generator = torch.Generator().manual_seed(seed)
        directions, _ = torch.linalg.qr(torch.randn(features.shape[1], 2,
                                                   generator=generator, dtype=torch.float64))
        assert torch.linalg.matrix_norm(directions.T @ directions - torch.eye(2)) < 1e-12
        projected = features @ directions
        baseline = Z[-1][:, neuron]
        column = net.readout.weight[:, neuron].clone()
        response, flips = [], []
        for chunk in coordinates.split(128):
            preactivation = baseline[:, None] + projected @ chunk.T
            response.append(preactivation.relu().square().mean(dim=0).sqrt() * column.norm())
            flips.append((preactivation > 0).sum(dim=0))
        cart = torch.cat(response).reshape(101, 101).numpy()
        changed = torch.cat(flips).reshape(101, 101).numpy()
        row = net.layers[-1].weight[neuron].clone()
        bias = net.layers[-1].bias[neuron].clone()
        errors, frozen_errors = [], []
        # Include the largest measured response, the origin and three corners.
        largest = tuple(np.unravel_index(np.argmax(cart), cart.shape))
        for i, j in [(0, 0), (0, 100), (100, 0), (50, 50), largest]:
            ab = torch.tensor([axis[i], axis[j]])
            edit = directions @ ab
            net.layers[-1].weight[neuron].copy_(row + edit[:-1])
            net.layers[-1].bias[neuron].copy_(bias + edit[-1])
            predicted = (baseline + projected @ ab).relu()[:, None] * column[None, :]
            actual = net(X) - F0
            errors.append(float((actual - predicted).square().sum(1).mean().sqrt()))
            frozen = FL.frozen_forward(net, masks, X, torch.arange(len(X))) - F0
            frozen_errors.append(float(frozen.square().sum(1).mean().sqrt()))
            assert abs(float(actual.square().sum(1).mean().sqrt()) - cart[i, j]) < 1e-12
        net.layers[-1].weight[neuron].copy_(row)
        net.layers[-1].bias[neuron].copy_(bias)
        assert max(errors) < 1e-12 and max(frozen_errors) < 1e-12
        np.savez(directory / f'ocean_{number:02d}.npz', axis=axis, cart_true=cart,
                 cart_frozen=np.zeros_like(cart), cart_flips=changed,
                 directions=directions.numpy(), analysis_indices=indices,
                 neuron=neuron, seed=seed, theta_norm=parameter_norm,
                 F0_norm=float(F0.square().sum(1).mean().sqrt()))
        manifest['planes'].append(dict(number=number, neuron=neuron, seed=seed,
                                       full_forward_vector_error_max=max(errors),
                                       frozen_response_max=max(frozen_errors)))
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'plane {number}/{count}, neuron {neuron}, vector error {max(errors):.2g}', flush=True)


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    seed = 20261123
    run = runs.load_run('mnist', 'mlp', 0, tag='w64', epoch=305, device='cpu')
    net = run['net']
    assert net.activation == 'relu' and not net.batchnorm
    X, _, indices = run['data']['analysis']
    F0, Z, H = net(X, return_states=True)
    neuron = 0
    assert (Z[-1][:, neuron] <= 0).all()
    features = torch.cat([H[-1], torch.ones(len(X), 1, dtype=X.dtype)], dim=1)
    generator = torch.Generator().manual_seed(seed)
    directions, _ = torch.linalg.qr(torch.randn(features.shape[1], 2,
                                               generator=generator, dtype=torch.float64))
    assert torch.linalg.matrix_norm(directions.T @ directions - torch.eye(2)) < 1e-12
    projected = features @ directions
    baseline = Z[-1][:, neuron]
    column = net.readout.weight[:, neuron].clone()
    axis = np.linspace(-1.5, 1.5, 101)
    aa, bb = np.meshgrid(axis, axis, indexing='ij')
    coordinates = torch.tensor(np.stack([aa.ravel(), bb.ravel()], axis=1))
    response, flips = [], []
    for chunk in coordinates.split(128):
        preactivation = baseline[:, None] + projected @ chunk.T
        response.append(preactivation.relu().square().mean(dim=0).sqrt() * column.norm())
        flips.append((preactivation > 0).sum(dim=0))
    cart = torch.cat(response).reshape(101, 101).numpy()
    changed = torch.cat(flips).reshape(101, 101).numpy()

    row = net.layers[-1].weight[neuron].clone()
    bias = net.layers[-1].bias[neuron].clone()
    masks = FL.base_masks(net, X)
    errors = []
    frozen_errors = []
    for i, j in [(0, 0), (0, 50), (0, 100), (25, 25), (25, 75), (50, 0),
                 (50, 50), (50, 100), (75, 25), (75, 75), (100, 0), (100, 50), (100, 100)]:
        ab = torch.tensor([axis[i], axis[j]])
        edit = directions @ ab
        net.layers[-1].weight[neuron].copy_(row + edit[:-1])
        net.layers[-1].bias[neuron].copy_(bias + edit[-1])
        predicted = (baseline + projected @ ab).relu()[:, None] * column[None, :]
        actual = net(X) - F0
        errors.append(float((actual - predicted).square().sum(1).mean().sqrt()))
        frozen = FL.frozen_forward(net, masks, X, torch.arange(len(X))) - F0
        frozen_errors.append(float(frozen.square().sum(1).mean().sqrt()))
        assert abs(float(actual.square().sum(1).mean().sqrt()) - cart[i, j]) < 1e-12
    net.layers[-1].weight[neuron].copy_(row)
    net.layers[-1].bias[neuron].copy_(bias)
    assert max(errors) < 1e-12 and max(frozen_errors) < 1e-12
    directory = Path(runs.RESULTS_ROOT) / 'paper_slices'
    stem = directory / 'gal_dead_unit_fresh_11'
    checkpoint = Path(run['dir']) / 'checkpoints/epoch_0305.pt'
    record = dict(seed=seed, neuron=neuron, layer='W3', checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  examples=len(X), analysis_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                  full_forward_vector_error_max=max(errors), frozen_response_max=max(frozen_errors),
                  method='exact single-row ReLU/readout reduction, checked at 13 grid points',
                  grid=[101, 101], coordinate_range=[-1.5, 1.5])
    np.savez(stem.with_suffix('.npz'), axis=axis, cart_true=cart,
             cart_frozen=np.zeros_like(cart), cart_flips=changed,
             directions=directions.numpy(), analysis_indices=indices,
             theta_norm=float(FL.flat_params(net).norm()),
             F0_norm=float(F0.square().sum(1).mean().sqrt()))
    stem.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int)
    parser.add_argument('--center', nargs='+', type=int)
    parser.add_argument('--shared-window', action='store_true')
    parser.add_argument('--ggn', action='store_true')
    args = parser.parse_args()
    if args.ggn:
        shared_window(ggn=True)
    elif args.shared_window:
        shared_window()
    elif args.center:
        centered(args.center)
    elif args.batch:
        batch(args.batch)
    else:
        main()
