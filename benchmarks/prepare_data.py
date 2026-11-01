#!/usr/bin/env python3
"""Prepare benchmark inputs as raw little-endian float32 files.

Mirrors the upstream turbovec benchmark loader: normalize every row to unit
length, draw the 100K corpus from the training split with seed 42, and take
the first 1,000 queries from the test split.

Usage:
  python3 benchmarks/prepare_data.py glove-200 ~/data/go-turbovec/glove-200-angular.hdf5
  python3 benchmarks/prepare_data.py openai-1536 ~/data/go-turbovec/openai-1536.npy
"""

import os
import sys

import numpy as np

N, NQ, SEED = 100_000, 1_000, 42


def normalize(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms


def load_glove(path):
    import h5py

    with h5py.File(path, "r") as f:
        train = f["train"][:]
        test = f["test"][:]
    return train, test


def load_npy(path):
    # The DBpedia export is one flat array with no train/test split, so the
    # permutation below draws both corpus and queries from it.
    return np.load(path), None


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    name, src = sys.argv[1], sys.argv[2]
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    if src.endswith((".hdf5", ".h5")):
        train, test = load_glove(src)
    else:
        train, test = load_npy(src)

    rng = np.random.RandomState(SEED)
    if test is None:
        # One array, no split: draw corpus and queries from it in one pass.
        idx = rng.choice(len(train), N + NQ, replace=False)
        database, queries = train[idx[:N]], train[idx[N:]]
    else:
        # Same draw as the upstream scripts: 100K corpus rows from train, and
        # every test row as a recall query.
        database = train[rng.choice(len(train), N, replace=False)]
        queries = test
    recall_queries = queries
    database, queries = normalize(database), normalize(queries[:NQ])

    dim = database.shape[1]
    database.astype("<f4").tofile(os.path.join(out_dir, f"{name}-corpus.f32"))
    queries.astype("<f4").tofile(os.path.join(out_dir, f"{name}-queries.f32"))
    normalize(recall_queries).astype("<f4").tofile(
        os.path.join(out_dir, f"{name}-recall-queries.f32")
    )
    print(
        f"{name}: dim={dim} corpus={database.shape} "
        f"queries={queries.shape} recall_queries={recall_queries.shape}"
    )


if __name__ == "__main__":
    main()
