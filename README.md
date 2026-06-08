# go-turbovec

A Go package that re-exposes the [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) Rust vector index under idiomatic Go types. It replaces the upstream's Python piece (`turbovec-python`, a thin PyO3 wrapper) with the equivalent Go binding backed by the same Rust core.

## Quick start

```go
package main

import (
	"fmt"
	"github.com/divy/go-turbovec/turbovec"
)

func main() {
	idx, _ := turbovec.NewTurboQuantIndex(128, 4)
	defer idx.Close()

	vectors := make([]float32, 100*128) // 100 vectors, dim 128
	idx.Add(vectors)
	idx.Prepare()

	r, _ := idx.Search(vectors[:128], 10)
	for i, idx := range r.IndicesForQuery(0) {
		fmt.Printf("rank %d: id %d score %.4f\n", i, idx, r.ScoresForQuery(0)[i])
	}
}
```

## Build

```bash
git submodule update --init
make build
```

Requires Go 1.22+, a C toolchain, and a Rust toolchain (stable).

## API

- `turbovec.TurboQuantIndex` — basic top-k index
- `turbovec.IdMapIndex` — top-k index keyed by external `uint64` ids
- `turbovec.Results` and `turbovec.IdResults` — search output

See `docs/superpowers/specs/2026-06-07-go-turbovec-design.md` for the full design.

## License

MIT, same as the upstream.
