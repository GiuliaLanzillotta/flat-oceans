"""Small forward-pass measurements used by the representation figures.

The downstream sampling sequence is the original seed-0 NumPy sequence, with a
fresh stratified draw at each checkpoint. CNN activity means activity anywhere
in the channel's spatial map, rather than the fraction of active pixels.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
import argparse
import sys
import numpy as np
import torch
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flatocean import runs, blocks
from flatocean.paths import RESULTS


def downstream(out):
    rng = np.random.default_rng(0)
    data = {}
    for stage, epoch in [('high_accuracy', 40), ('late_endpoint', 500)]:
        run = runs.load_run('mnist', 'mlp', 0, epoch=epoch, device='cpu')
        X, labels, _ = run['data']['analysis']
        labels = labels.numpy()
        idx = np.concatenate([rng.choice(np.where(labels == c)[0], 200, replace=False) for c in range(10)])
        with torch.no_grad():
            structure = blocks.mlp_structure(run['net'], X[idx])
        for layer in (2, 3):
            D = structure['D'][layer - 1].reshape(len(idx), -1).numpy()
            D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-30)
            data[f'd_{stage}_{layer}'] = pdist(D).astype(np.float32)
        data['y'] = labels[idx]
    np.savez_compressed(out / 'dcluster_cache.npz', **data)


def cnn(out):
    run = runs.load_run('mnist', 'cnn', 0, epoch=500, device='cpu')
    net = run['net']; X, labels, _ = run['data']['analysis']; labels = labels.numpy()
    active = np.zeros((len(labels), 64), dtype=bool)
    means = np.zeros((len(labels), 64))
    with torch.no_grad():
        for start in range(0, len(labels), 500):
            _, Z, _ = net(X[start:start + 500], return_states=True)
            active[start:start + 500] = (Z[4] > 0).flatten(2).any(-1).numpy()
            means[start:start + 500] = torch.relu(Z[4]).flatten(2).mean(-1).numpy()
    frac = np.stack([active[labels == c].mean(0) for c in range(10)], axis=1)
    np.savez(out / 'cnn_units_cache.npz', frac5=frac)
    p = frac / np.maximum(frac.sum(1, keepdims=True), 1e-12)
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
    selected = list(np.argsort(entropy)[:3]) + [int(np.argsort(entropy)[-1])]
    imgs = np.zeros((4, 4, 32, 32)); acts = np.zeros_like(imgs); labs = np.zeros((4, 4), int)
    for k, channel in enumerate(selected):
        top = np.argsort(means[:, channel])[::-1][:4]
        for j, i in enumerate(map(int, top)):
            xi = X[i:i + 1].clone().requires_grad_(True)
            _, Z, _ = net(xi, return_states=True)
            a = torch.relu(Z[4][0, channel])
            a.flatten()[int(a.argmax())].backward()
            sal = xi.grad[0, 0].abs().numpy()
            sal = .25 * sal + .75 * np.array([[sal[max(u-1, 0):u+2, max(v-1, 0):v+2].mean() for v in range(32)] for u in range(32)])
            imgs[k, j] = X[i, 0].numpy(); acts[k, j] = sal; labs[k, j] = labels[i]
    np.savez(out / 'cnn_units_cache2.npz', imgs=imgs, acts=acts, labs=labs)


def margins(out):
    data = {}
    for tag in ['w64', 'mgn_l0', 'mgn_l1']:
        values = []
        for seed in range(3):
            run = runs.load_run('mnist', 'mlp', seed, tag=tag, epoch=500, device='cpu')
            with torch.no_grad():
                _, Z, _ = run['net'](run['data']['analysis'][0], return_states=True)
            values.append(np.abs(torch.cat([z.flatten() for z in Z]).numpy()))
        data[tag] = np.concatenate(values)
    np.savez(out / 'margin_dists_cache.npz', **data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurement', choices=['downstream', 'cnn', 'margins', 'all'])
    parser.add_argument('--output', type=Path, default=RESULTS / 'visualization')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, fn in [('downstream', downstream), ('cnn', cnn), ('margins', margins)]:
        if args.measurement in (name, 'all'):
            fn(args.output)


if __name__ == '__main__':
    main()
