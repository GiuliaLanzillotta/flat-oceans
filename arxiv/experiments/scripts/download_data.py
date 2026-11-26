"""Download the public datasets used by the paper. RESISC45 requires --resisc."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from flatocean.paths import ROOT, dataset

RESISC_REV = 'fe12fc5f1b7606543b0355eda392f1ddc54625c6'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resisc', action='store_true')
    p.add_argument('--check', action='store_true', help='check existing files without downloading')
    a = p.parse_args()
    if not a.check:
        from torchvision.datasets import MNIST, SVHN
        # torchvision puts MNIST/raw beneath the supplied parent directory.
        mnist = dataset('MNIST'); mnist.parent.mkdir(parents=True, exist_ok=True)
        for train in [True, False]: MNIST(str(mnist.parent), train=train, download=True)
        svhn = dataset('SVHN'); svhn.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'test']: SVHN(str(svhn), split=split, download=True)
        if a.resisc:
            resisc = dataset('RESISC45'); resisc.mkdir(parents=True, exist_ok=True)
            for split in ['train', 'test']:
                path = resisc/f'{split}.parquet'
                if path.exists(): continue
                url = f'https://huggingface.co/datasets/timm/resisc45/resolve/{RESISC_REV}/data/{split}-00000-of-00001.parquet'
                temp = path.with_suffix('.part')
                with urllib.request.urlopen(url) as response, temp.open('wb') as f: shutil.copyfileobj(response, f)
                temp.replace(path)
    records = json.loads((ROOT/'configs/datasets.json').read_text())
    failures=[]
    for name, files in records.items():
        if name=='RESISC45' and not a.resisc: continue
        for relative, expected in files.items():
            path=dataset(name)/relative
            if not path.is_file(): failures.append(f'missing {path}'); continue
            if sha(path)!=expected: failures.append(f'checksum differs: {path}')
    if failures: raise SystemExit('\n'.join(failures))
    print('Dataset hashes match the recorded inputs.')


if __name__ == '__main__': main()
