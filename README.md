<p align="center">
  <strong>go-turbovec</strong><br>
  Go bindings for Google's TurboQuant vector index
</p>

<p align="center">
  <a href="https://github.com/RyanCodrai/turbovec"><img src="https://img.shields.io/badge/core-RyanCodrai%2Fturbovec-blue" alt="Upstream turbovec"></a>
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="TurboQuant paper"></a>
  <img src="https://img.shields.io/badge/go-1.22%2B-00ADD8" alt="Go 1.22+">
</p>

---

**A 10 million document corpus takes 31 GB of RAM as float32. TurboQuant fits it in a fraction of that memory while keeping fast local search.**

`go-turbovec` exposes [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) to Go through idiomatic index types backed by the same Rust core and C FFI. It replaces the upstream Python binding with Go APIs for building local vector search into Go services.

- **Online ingest.** Add vectors and search them immediately. No separate train step, no parameter tuning, and no rebuild as the corpus grows.
- **Compressed vector storage.** Store vectors with TurboQuant's 2-, 3-, or 4-bit quantization instead of raw float32 payloads.
- **Stable external ids.** Use `IdMapIndex` when application ids must survive deletes.
- **Filtered search.** Restrict results at search time with a slot mask or id allowlist, useful for hybrid retrieval, tenancy, ACLs, and time windows.
- **Pure local.** Keep embeddings inside your process, machine, or VPC. No managed vector service required.

Building RAG or semantic search in Go where memory, privacy, or latency matters? This package gives you the TurboQuant core without leaving Go.

## Install

```bash
go get github.com/shepard-labs/go-turbovec
```

This package uses cgo and links against the Rust FFI library built from the vendored upstream core. See [Building](#building) when working from source.

## Quick Start

```go
package main

import (
	"fmt"
	"log"

	"github.com/shepard-labs/go-turbovec/turbovec"
)

func main() {
	idx, err := turbovec.NewTurboQuantIndex(128, 4)
	if err != nil {
		log.Fatal(err)
	}
	defer idx.Close()

	vectors := make([]float32, 100*128) // 100 vectors, dim 128
	if err := idx.Add(vectors); err != nil {
		log.Fatal(err)
	}

	idx.Prepare() // optional: pay one-time setup cost before the first search

	results, err := idx.Search(vectors[:128], 10)
	if err != nil {
		log.Fatal(err)
	}

	for rank, slot := range results.IndicesForQuery(0) {
		fmt.Printf("rank %d: slot %d score %.4f\n", rank, slot, results.ScoresForQuery(0)[rank])
	}
}
```

Need stable ids that survive deletes? Use `IdMapIndex`:

```go
idx, err := turbovec.NewIdMapIndex(128, 4)
if err != nil {
	log.Fatal(err)
}
defer idx.Close()

vectors := make([]float32, 3*128)
ids := []uint64{1001, 1002, 1003}

if err := idx.AddWithIDs(vectors, ids); err != nil {
	log.Fatal(err)
}

results, err := idx.Search(vectors[:128], 2)
if err != nil {
	log.Fatal(err)
}

fmt.Println(results.IDsForQuery(0)) // external uint64 ids
idx.Remove(1002)                    // O(1) by id
```

## Hybrid Retrieval

Restrict dense search to a candidate set produced by another system such as SQL, BM25, ACL checks, or a time-window filter.

```go
allowed := []uint64{1001, 1003, 1042}
results, err := idx.SearchWithAllowlist(query, 10, allowed)
if err != nil {
	log.Fatal(err)
}

// results.K == min(10, len(allowed))
fmt.Println(results.IDsForQuery(0))
```

For positional indexes, pass a boolean slot mask:

```go
mask := make([]bool, tq.Len())
mask[0] = true
mask[3] = true

results, err := tq.SearchWithMask(query, 10, mask)
```

Filtering happens inside the search path rather than by over-fetching and dropping results afterward. The output length is `min(k, allowedCount)`.

See [`docs/api.md`](docs/api.md) for the full Go API reference.

## How It Works

Each vector is treated as a direction on a high-dimensional hypersphere. TurboQuant compresses these directions using a data-oblivious quantizer: after applying a random rotation, every coordinate follows a known distribution regardless of the input data.

1. **Normalize.** Store the vector length separately and quantize the unit direction.
2. **Random rotation.** Rotate vectors into a domain where coordinate distributions are predictable.
3. **Scalar quantization.** Quantize each coordinate using precomputed Lloyd-Max buckets for the target bit width.
4. **Bit-pack.** Store each coordinate in 2, 3, or 4 bits instead of 32-bit floats.
5. **Direct scoring.** Rotate the query once, then score against compressed codes without fully decompressing the database vectors.

The Go package calls into the upstream Rust implementation for indexing, serialization, and search.

## Building

```bash
make submodules
make build
```

Requirements:

- Go 1.22+
- Rust stable toolchain
- C toolchain with cgo enabled
- OpenBLAS on Linux

`make submodules` fetches the upstream `vendor/turbovec` submodule and applies the sparse checkout used by this binding.

## Testing

```bash
go test ./...
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) - the paper behind TurboQuant.
- [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) - upstream Rust vector index and Python bindings.
- [FAISS Fast accumulation of PQ and AQ codes](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)) - background on FastScan-style product quantization kernels.

## License

MIT, same as the upstream project.
