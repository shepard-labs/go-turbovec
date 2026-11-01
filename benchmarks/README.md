# Benchmarks

The harness in this directory reproduces the shapes the upstream
[`turbovec` benchmark suite](https://github.com/RyanCodrai/turbovec/blob/main/docs/benchmarks.md)
uses: a 100,000-vector corpus, 1,000 queries, `k=64`, and the median of five
timed runs. Recall is scored against an exact float32 scan of the same corpus,
computed here rather than copied from upstream.

Two things differ from upstream on purpose:

- **Baseline.** Upstream compares against FAISS. This harness compares the Go
  binding's compressed index against an exact float32 top-k scan in Go, so the
  numbers answer the question a Go caller has ("what does quantization buy me
  over brute force?") without a Python/FAISS dependency.
- **Calibration.** Upstream publishes both TQ and TQ+ (`calibrate()` on a
  1,024-row sample). The Go binding does not expose `calibrate`, so every
  recall number here is uncalibrated TQ.

## Datasets

| Name | Source | Shape |
|---|---|---|
| `glove-200-angular` | [ann-benchmarks](http://ann-benchmarks.com/glove-200-angular.hdf5) (`train` split) | 100K corpus, 10K recall queries, d=200 |
| `openai-1536` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) | 100K corpus, 1K recall queries, d=1536 |
| `openai-3072` | [Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) | 100K corpus, 1K recall queries, d=3072 |

Every row is normalized to unit length before use, matching upstream.

The OpenAI datasets are shards of a 1M-row corpus. Upstream draws its 100K
rows from the whole thing; the recipe below stops after enough shards
(3 at d=1536, 7 at d=3072), which is why the OpenAI recall figures sit within
about a point and a half of upstream's rather than matching exactly.

## Running it

```bash
# 1. Fetch the datasets.
curl -L -o ~/data/go-turbovec/glove-200-angular.hdf5 \
  http://ann-benchmarks.com/glove-200-angular.hdf5

# OpenAI DBpedia ships as parquet shards. Three hold enough rows at d=1536;
# d=3072 packs half as many rows per shard, so it needs seven.
for i in 0 1 2 3 4 5 6; do
  curl -sL -o ~/data/go-turbovec/openai3072/train-0000$i.parquet \
    "https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M/resolve/main/data/train-0000$i-of-00063.parquet"
done

# 2. Convert to raw float32. Needs numpy, h5py and pyarrow.
python3 benchmarks/extract_openai.py 3072 ~/data/go-turbovec/openai3072 ~/data/go-turbovec/openai-3072
python3 benchmarks/prepare_data.py openai-3072 ~/data/go-turbovec/openai-3072.npy
python3 benchmarks/prepare_data.py glove-200 ~/data/go-turbovec/glove-200-angular.hdf5

# 3. Build and run. RAYON_NUM_THREADS=1 pins the Rust core to one thread.
make build
go build -o /tmp/turbovec-bench ./benchmarks/bench

RAYON_NUM_THREADS=1 /tmp/turbovec-bench \
  -corpus benchmarks/data/openai-3072-corpus.f32 \
  -queries benchmarks/data/openai-3072-queries.f32 \
  -recall-queries benchmarks/data/openai-3072-recall-queries.f32 \
  -dim 3072 -k 64 -runs 5 -exact-threads 1 \
  -machine "Apple M4 Max, 16 cores, macOS arm64" \
  -out benchmarks/results/openai_3072_st.json

/tmp/turbovec-bench ... -out benchmarks/results/openai_3072_mt.json   # omit RAYON_NUM_THREADS
```

Results land in `benchmarks/results/` as JSON, and the README table is derived
from those files.

## What each measurement means

| Field | Meaning |
|---|---|
| `exact.search_ms_per_query` | Exact float32 top-64 scan of the corpus, in Go, one goroutine per available core unless `-exact-threads` pins it. |
| `recall_at_1_in_top_k` | Fraction of recall queries whose true top-1 appears in the first `k'` results, `k'` in 1, 2, 4, 8, 16, 32, 64. |
| `search_ms_per_query` | Full 1,000-query batch divided by the batch size, after a warm-up query. |
| `build_ms` | `Add` of all 100K vectors into a fresh index, one call. |
| `insert_single_us` / `insert_batch100_us_per_vector` | `Add` of one vector, and of a 100-vector batch, into the warm, populated 100K index. |
| `write_ms` / `load_first_search_ms` | Whole-file `Write`, then `LoadTurboQuantIndex` followed by one query. |
| `file_mb` vs `fp32_mb` | Serialized index size against the same vectors as raw float32. |
| `remove.single_us_per_op` / `remove.batch100_us_per_op` | `IdMapIndex.Remove` per op, over 1,000 removes and over the first 100 removes. |

Single-threaded runs set `RAYON_NUM_THREADS=1`, which covers the Rust core's
search and encode parallelism; `-exact-threads 1` pins the Go baseline to
match, so the speedup ratio compares like with like.
