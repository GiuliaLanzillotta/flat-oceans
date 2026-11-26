"""Compile the paper in arxiv/.build and package its declared LaTeX dependencies."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tectonic', default='tectonic', help='path to Tectonic; use --pdflatex for TeX Live')
    p.add_argument('--pdflatex', action='store_true')
    a=p.parse_args(argv)
    if Path(a.tectonic).is_file():
        a.tectonic=str(Path(a.tectonic).resolve())
    source=ROOT.parent; dest=source/'.build'
    files=sorted(set(json.loads((ROOT/'configs/paper_files.json').read_text()).values()))
    if dest.exists(): shutil.rmtree(dest)
    for name in files:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Invalid manuscript path: {name}')
        target=dest/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/path,target)
    if a.pdflatex:
        for _ in range(3): subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=dest,check=True)
    else: subprocess.run([a.tectonic,'-k','main.tex'],cwd=dest,check=True)
    output=source
    shutil.copy2(dest/'main.pdf',output/'flat_oceans.pdf')
    with zipfile.ZipFile(output/'arxiv.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(source/name,name)
    print(output/'arxiv.zip')


if __name__=='__main__': main()
