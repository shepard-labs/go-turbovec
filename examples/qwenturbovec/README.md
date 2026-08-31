# qwenturbovec — go-turbovec × LM Studio demo

A small Go example that

1. downloads a parquet dataset from Hugging Face,
2. embeds every row by calling **LM Studio's local server** running **text-embedding-qwen3-embedding-4b**,
3. indexes the vectors with **go-turbovec**,
4. exposes a minimal web UI that does semantic search.

```
            ┌──────────────────────────────┐
text  ──►   │  LM Studio                   │   ──►  float32[2560]
            │  (text-embedding-qwen3-4b)   │
query ──►   │  localhost:8234/v1/embeddings│   ──►  query vector
            └──────────────────────────────┘
                              │
                              ▼
                  go-turbovec IdMapIndex
                              │
                              ▼
                         top-k results
```

The Go binary has **no cgo, no ONNX, no tokenizer library** — it's a thin HTTP client around the embedding server. All the heavy lifting happens in LM Studio.

## Prerequisites

1. **Go 1.22+**.
2. **Rust + C toolchain** for the `turbovec-ffi` static library. Build it once from the repo root:
   ```bash
   cd /Users/divy/Projects/go-turbovec    # or wherever you cloned it
   make build
   ```
   The build produces `turbovec-ffi/target/release/libturbovec_ffi.a` which the cgo layer in `go-turbovec/turbovec` links **statically** into the example binary. If this step is skipped, `go build` will fail with "library not found" errors.
3. **LM Studio** with:
   - the `text-embedding-qwen3-embedding-4b` model loaded,
   - the **local server** started (Developer tab → Start Server).

   By default LM Studio listens on `http://127.0.0.1:8234`. The URL is overridable with `--embed-url`. The model identifier is overridable with `--embed-model`.

### Sanity-check the prerequisites

Run these two commands **before** going further. If either fails, the example won't work either.

```bash
# 1. The static FFI library exists (built by `make build` in the parent repo)
ls /Users/divy/Projects/go-turbovec/turbovec-ffi/target/release/libturbovec_ffi.a

# 2. LM Studio's server is up and the model is loaded
curl -s http://127.0.0.1:8234/v1/models | grep text-embedding-qwen3-embedding-4b
```

## Run

From this directory (`examples/qwenturbovec/`):

```bash
# Build the binary once into ./bin/ so the example root stays clean
go build -o bin/qwenturbovec ./cmd/qwenturbovec

# First run: downloads the parquet (~30MB) and embeds the rows through LM Studio.
./bin/qwenturbovec --max-rows 5000 --port 8080

# Subsequent runs: skip embedding, just load the cached index.
./bin/qwenturbovec --max-rows 5000 --port 8080
```

> **About `go run`:** it works, but it recompiles on every invocation (~1-3s). The `go build` form is faster for repeated runs while iterating.

Open <http://localhost:8080> in a browser. Type a query — try "space launch" or "stock market crash" — and the page shows the 10 most semantically similar rows from `ag_news`.

## CLI

| Flag            | Default                                  | Description                                                  |
|-----------------|------------------------------------------|--------------------------------------------------------------|
| `--data-dir`    | `./data`                                 | Cache directory for datasets and the index.                  |
| `--dataset`     | `fancyzhx/ag_news`                       | HF dataset repo id. Override with e.g. `dbpedia_14`.         |
| `--max-rows`    | `5000`                                   | Cap on rows embedded (0 = no cap; 120k for ag_news).         |
| `--port`        | `8080`                                   | HTTP port.                                                   |
| `--embed-url`   | `http://127.0.0.1:8234/v1`               | OpenAI-compatible embedding endpoint.                         |
| `--embed-model` | `text-embedding-qwen3-embedding-4b`      | Model identifier passed to the embed server.                 |
| `--rebuild`     | `false`                                  | Force re-embedding (ignores cached index).                   |

## On-disk layout

```
examples/qwenturbovec/
├── bin/
│   └── qwenturbovec        (the built binary)
├── cmd/qwenturbovec/       (source)
├── data/
│   └── datasets/fancyzhx_ag_news/
│       ├── train.parquet   (raw rows from HF, ~30MB)
│       ├── index.tvim      (go-turbovec IdMapIndex, fast to load)
│       └── rows.bin        (id → text sidecar for search results)
├── go.mod
└── README.md
```

Restart the binary with the same `--data-dir` and it loads `index.tvim` directly. The `index.tvim` is a go-turbovec `IdMapIndex` — only the float32 vectors are stored there. The original row text lives in `rows.bin` (id → text) so search results can be rendered. If you delete `index.tvim` and `rows.bin`, the next run re-embeds from `train.parquet`.

### Changing `--max-rows` or `--embed-model`

The cache is keyed by **dataset only** — the on-disk files are `index.tvim` and `rows.bin`, both written with no metadata about how they were built. If you change `--max-rows` to a bigger value, the next run will silently reuse the smaller cached index. To force a rebuild:

```bash
./bin/qwenturbovec --max-rows 50000 --rebuild
```

The `index.tvim` and `rows.bin` files are deleted and rebuilt from scratch.

## Using your own dataset

The example reads a Hugging Face dataset whose parquet shard has at least one of these column layouts:

| Columns                | What gets embedded                              |
|------------------------|-------------------------------------------------|
| `title` + `text`       | `title — text` (concatenated with em-dash)      |
| `text` only            | `text`                                          |
| `title` only           | `title`                                         |

Pick any HF dataset whose parquet shards match — `dbpedia_14`, `SetFit/20_newsgroups`, `cc_news`, etc. all work out of the box:

```bash
./bin/qwenturbovec --dataset SetFit/20_newsgroups --max-rows 30000 --rebuild
```

The example does **not** read JSON, markdown, or arbitrary text formats. To index a JSON/CSV/markdown corpus, convert it to a parquet file with one row per document and a `text` column, push it to a HF dataset repo, then point `--dataset` at that repo.

For local-only datasets, the simplest path is:

1. Build a parquet file with `pandas.DataFrame({"text": [...]}).to_parquet("data.parquet")` and push it to a private HF dataset repo.
2. Run the example with `--dataset your-username/your-dataset`.

## Code map

| File                              | What it does                                                |
|-----------------------------------|-------------------------------------------------------------|
| `cmd/qwenturbovec/main.go`        | CLI flag parsing, orchestration, log lines.                 |
| `cmd/qwenturbovec/hf.go`          | HF tree API listing + `downloadIfMissing` cache.            |
| `cmd/qwenturbovec/parquet.go`     | Stream rows from a HF parquet file with `parquet-go`.       |
| `cmd/qwenturbovec/embed.go`       | HTTP client → `/v1/embeddings`, L2-normalise.               |
| `cmd/qwenturbovec/index.go`       | `IdMapIndex` load-or-build, `.tvim` cache, row sidecar.     |
| `cmd/qwenturbovec/serve.go`       | `net/http` server + embedded `index.html`.                  |
| `cmd/qwenturbovec/web/index.html` | The UI.                                                     |

## Notes

- **Dimension.** Qwen3-Embedding-4B's output dimension is probed from the first embedding response and used to size the `IdMapIndex`. The current default is 2560.
- **Why L2-normalise?** go-turbovec uses **inner-product distance** for ranking (cosine similarity for unit vectors). Server-side embeddings are usually not normalised, so we normalise in the client. This also makes the index score directly interpretable as a cosine similarity in `[0, 1]`.
- **Bit width.** 4 bits is the go-turbovec default and offers a good accuracy/memory tradeoff. Drop to 2 or 3 if you're indexing 1M+ rows on memory-constrained hardware.
- **Why an `IdMapIndex`?** It keys by external `uint64` ids so we can store the row text in a parallel slice and look it up on a hit. `TurboQuantIndex` would also work if you don't need external ids.
- **Why a `replace` directive in go.mod?** This example is a sibling module to the parent `go-turbovec` repo. The `replace` line makes `go.mod` resolve `github.com/divy/go-turbovec` to the local `../..` checkout, so any change to the library is picked up on the next `go build`.
- **LM Studio log noise.** You may see warnings like *"At least one last token in strings embedded is not SEP. 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header"* in the LM Studio log. This is harmless for embedding — it just means the GGUF model file didn't mark the end-of-sequence token as auto-appended, and LM Studio fell back to its default behaviour. The returned vectors are still correct.

## Troubleshooting

| Symptom                                                                                | Likely cause                                                              | Fix                                                                                  |
|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `could not reach embedding server at …`                                                 | LM Studio's local server isn't running                                   | Open LM Studio → Developer tab → "Start Server"                                     |
| `HTTP 400: model not found`                                                            | The model identifier doesn't match what's loaded in LM Studio            | Run `curl -s http://127.0.0.1:8234/v1/models` and use one of the returned `id`s     |
| `cgo: C compiler cannot create executables` / `library not found -lturbovec_ffi`       | `make build` was not run in the parent repo                               | `cd /Users/divy/Projects/go-turbovec && make build`, then verify `turbovec-ffi/target/release/libturbovec_ffi.a` exists |
| `listen tcp :8080: bind: address already in use`                                        | A previous run is still occupying the port                                | `lsof -ti :8080 | xargs kill`                                                        |
| Search returns poor results                                                            | The embedding model isn't fully loaded yet, or it's the wrong model       | Wait for the model to finish loading in LM Studio's UI; verify with `--embed-model`  |
