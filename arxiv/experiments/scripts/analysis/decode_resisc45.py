"""One-time decode of RESISC45 parquet -> cached uint8 HWC arrays (scratchpad). Row order = parquet (canonical timm split)."""
import io, os, sys, time, numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ftvit_paths import DATA_ROOT, CACHE_DIR

ROOT = DATA_ROOT
OUT = CACHE_DIR; os.makedirs(OUT, exist_ok=True)

def decode(split):
    t = pq.read_table(os.path.join(ROOT, f"{split}.parquet"))
    labels = np.asarray(t.column("label").to_pylist(), dtype=np.int64)
    col = t.column("image")
    n = t.num_rows
    arr = np.empty((n, 256, 256, 3), dtype=np.uint8)
    t0 = time.time()
    for i in range(n):
        d = col[i].as_py()
        im = Image.open(io.BytesIO(d["bytes"])).convert("RGB")
        a = np.asarray(im, dtype=np.uint8)
        assert a.shape == (256, 256, 3), (split, i, a.shape)
        arr[i] = a
    print(f"{split}: {n} imgs decoded in {time.time()-t0:.0f}s; labels {labels.min()}..{labels.max()}", flush=True)
    np.save(os.path.join(OUT, f"resisc45_{split}_imgs.npy"), arr)
    np.save(os.path.join(OUT, f"resisc45_{split}_labels.npy"), labels)
    return n

def main():
    for sp in ("train", "test"):
        decode(sp)


if __name__ == "__main__":
    main()
