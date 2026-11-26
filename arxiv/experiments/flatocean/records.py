"""Result files with a precise caching contract.

Sequence enforced by `StageRecord`:
    authenticate inputs -> construct the identity (checkpoint, preprocessing, ordered samples, configuration, hashes of every
    consumed input) -> check the completeness of the requested measurements -> reuse or compute -> save atomically.
Completeness is per configuration and per measurement (`required` keys), so a new partition kind, a new neighbourhood size or
a new layer is computed without erasing unrelated work under the same identity.  A record under another identity is replaced
(the old record is moved in full to the file's "superseded" list, so nothing is silently lost).  Measurements that are
independent of each other (layers, blocks) are not part of the identity: each is complete on its own, with its own inputs
recorded, so adding one never erases another.
"""
import json
import os
import time

import numpy as np

from flatocean import provenance as P
jsonable = P._jsonable
KEEP_SUPERSEDED = 3


class ResultStore:
    """One JSON file: load once, save atomically (temporary file and rename), one process per file."""

    def __init__(self, path, header=None):
        self.path = path; os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = json.load(open(path)) if os.path.exists(path) else dict(header or {}, results={})

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=1, default=jsonable)
        os.replace(tmp, self.path)

    def stage(self, key, identity):
        """The record of `key` (stage name) under `identity`.  A stored record with another digest is moved in full to the
        file's "superseded" list (last KEEP_SUPERSEDED per stage), never dropped, and a fresh record is started."""
        rec = self.data["results"].get(key)
        if rec is not None and rec.get("identity", {}).get("digest") == identity["digest"]:
            return rec
        fresh = {"identity": identity}
        if rec is not None:
            arch = self.data.setdefault("superseded", [])
            arch.append({"stage": key, "replaced": time.strftime("%Y-%m-%d %H:%M"), "digest": rec.get("identity", {}).get("digest"), "record": rec})
            same = [e for e in arch if e["stage"] == key]
            for e in same[:-KEEP_SUPERSEDED]:
                arch.remove(e)
            fresh["superseded_digest"] = rec.get("identity", {}).get("digest")
        self.data["results"][key] = fresh
        return fresh


def identity(run, rd, ep, config, inputs=None):
    """Analysis identity whose configuration also carries the SHA-256 of every consumed input file (`inputs`: name -> hash)."""
    cfg = dict(config)
    if inputs:
        cfg["inputs_sha256"] = dict(inputs)
    return P.analysis_identity(run, rd, ep, cfg)


def missing(record, required):
    """The requested measurement keys absent from a record ('a.b' addresses nested keys)."""
    out = []
    for k in required:
        node = record; ok = True
        for part in k.split("."):
            if not isinstance(node, dict) or part not in node:
                ok = False; break
            node = node[part]
        if not ok:
            out.append(k)
    return out


def artifacts_present(paths):
    """True when every artifact file of a measurement exists (a JSON entry alone is not completeness)."""
    return all(os.path.exists(p) for p in paths)


def dump_json(path, obj):
    """Atomic JSON write (temporary file in the same directory, then rename): a killed process never leaves a truncated file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=jsonable)
    os.replace(tmp, path)
