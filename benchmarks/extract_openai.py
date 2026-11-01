#!/usr/bin/env python3
"""Extract the OpenAI DBpedia embedding column from parquet shards.

The Hugging Face dataset ships as 26 parquet shards; three of them hold
enough rows for the 100K corpus plus 1K queries. Output is a single raw
float32 file in the same layout prepare_data.py expects.

Usage:
  python3 benchmarks/extract_openai.py 1536 ~/data/go-turbovec/openai ~/data/go-turbovec/openai-1536
"""

import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    dim, shard_dir, out_base = sys.argv[1], sys.argv[2], sys.argv[3]

    shards = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    if not shards:
        sys.exit(f"no parquet shards in {shard_dir}")

    chunks, rows = [], 0
    for shard in shards:
        col = pq.read_table(shard, columns=[f"text-embedding-3-large-{dim}-embedding"])
        arr = col.column(0).combine_chunks().values.to_numpy(zero_copy_only=False)
        arr = arr.reshape(-1, int(dim)).astype(np.float32)
        chunks.append(arr)
        rows += len(arr)
        print(f"{os.path.basename(shard)}: {len(arr)} rows (total {rows})")
        if rows >= 101_000:
            break

    all_vecs = np.concatenate(chunks)[:101_000]
    np.save(out_base + ".npy", all_vecs)
    print(f"wrote {out_base}.npy {all_vecs.shape} {all_vecs.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
