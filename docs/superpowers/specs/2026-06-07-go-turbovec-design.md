# go-turbovec: a Go wrapper around the turbovec Rust core

**Status:** approved (brainstorming)
**Date:** 2026-06-07
**Upstream:** https://github.com/RyanCodrai/turbovec (pinned via git submodule)

## Summary

`go-turbovec` is a Go package that re-exposes the `RyanCodrai/turbovec` vector index (Rust core) under idiomatic Go types. It replaces the Python piece of the upstream — `turbovec-python`, a thin PyO3 wrapper — with the equivalent Go binding. The Rust core is unchanged and is consumed through a small C-ABI shim crate built and statically linked in this repository. The end result is a single `import "github.com/divy/go-turbovec"` that gives a Go user the same two index types, the same method semantics, and the same file formats as a Python user gets from `pip install turbovec`.

## Goals

- Go users can build and query a `TurboQuantIndex` and an `IdMapIndex` with the same surface as the Python package, but with Go-idiomatic types and errors.
- The Rust core remains the source of truth for quantization, codebook lookups, and SIMD scoring — no re-derivation of the math.
- The Go binary embeds the Rust static library; no runtime dependency on a `.so`/`.dylib` at deploy time.
- The `.tq` and `.tvim` file formats produced and consumed by upstream are byte-compatible with the upstream's output.

## Non-goals (v1)

- HTTP / gRPC / network surface. The upstream is explicitly an embeddable library; the Go package is the same.
- Framework integrations (LangChain, LlamaIndex, Haystack, Agno equivalents). v1 ships the two index types only.
- GPU support. The Rust core does not provide one; neither does this wrapper.
- Cross-compilation. cgo + a Rust toolchain on the build host are required; we will not chase a `GOOS=windows` build in v1.

## Repository layout

```
go-turbovec/
├── turbovec/                       # upstream as a git submodule (pinned to a known commit)
│   ├── turbovec/                   # the Rust core crate (consumed via path = "../turbovec")
│   ├── turbovec-python/            # ignored; we don't need the Python piece
│   └── ...
├── turbovec-ffi/                   # NEW: C-ABI shim crate (cdylib + staticlib)
│   ├── Cargo.toml
│   ├── include/
│   │   └── cabi.h                  # hand-written; consumed by Go's cgo
│   └── src/
│       └── lib.rs                  # extern "C" surface
├── turbovec/                       # NEW: the Go package (top-level)
│   ├── doc.go
│   ├── turbovec.go                 # cgo bridge, error mapping
│   ├── turboquant_index.go         # TurboQuantIndex wrapper
│   ├── id_map_index.go             # IdMapIndex wrapper
│   ├── search_results.go           # Results / IdResults structs
│   ├── errors.go                   # typed errors (sentinels + Error types)
│   └── *_test.go
├── turbovec_test/                  # integration tests + recall benchmark
├── go.mod
├── go.sum
├── Makefile                        # `make build` = submodule + cargo + go
├── gen.go                          # //go:generate directive
└── docs/superpowers/specs/         # this file
```

The submodule is referenced from `turbovec-ffi/Cargo.toml` as `path = "../turbovec/turbovec"` (i.e. the `turbovec` crate inside the upstream submodule). We pin the submodule to a specific commit, recorded in `.gitmodules` and `git submodule status`, and bump it explicitly when we want to follow upstream.

## Architecture

```
┌─────────────────────────┐
│  Go user code           │
│  (imports "turbovec")   │
└────────────┬────────────┘
             │ method calls
┌────────────▼────────────┐
│  Go package             │     runtime.SetFinalizer
│  (turbovec/*.go)        │     for leaked handles
│  cgo bridge             │
└────────────┬────────────┘
             │ extern "C" calls (cgo)
┌────────────▼────────────┐
│  turbovec-ffi           │
│  (cdylib + staticlib)   │     extern "C" surface
│  thin wrapper           │
└────────────┬────────────┘
             │ Rust calls
┌────────────▼────────────┐
│  turbovec (Rust core)   │     unchanged upstream
│  SIMD encode + search   │
└─────────────────────────┘
```

All quantization, codebook, rotation, packing, and SIMD search logic lives in the Rust core. The shim translates `extern "C"` to safe Rust. The Go package translates Go slices to C-ABI and back, and owns nothing on the C side beyond an opaque handle.

## The C-ABI shim (`turbovec-ffi`)

A new Rust crate with `crate-type = ["staticlib", "cdylib"]`. The staticlib is what cgo links against in the default `make build` flow. The cdylib is for users who want a `libturbovec_ffi.so` they can `dlopen`.

### Public functions

All function names are `lower_snake_case` with a type prefix (`tq_` for `TurboQuantIndex`, `id_` for `IdMapIndex`).

```c
// TurboQuantIndex
TqHandle *tq_new(size_t dim, uint32_t bit_width, char **err);
TqHandle *tq_new_lazy(uint32_t bit_width, char **err);
void      tq_drop(TqHandle *h);
void      tq_add(TqHandle *h, const float *vectors, size_t n, size_t dim, char **err);
TqSearchResults tq_search(const TqHandle *h, const float *queries, size_t nq,
                         size_t dim, size_t k, const uint8_t *mask /* nullable */);
size_t    tq_swap_remove(TqHandle *h, size_t idx, char **err);
void      tq_prepare(const TqHandle *h);
int32_t   tq_write(const TqHandle *h, const char *path);                 // 0 = ok
TqHandle *tq_load(const char *path, char **err);
size_t    tq_len(const TqHandle *h);
size_t    tq_dim(const TqHandle *h);            // 0 = uncommitted lazy
uint32_t  tq_bit_width(const TqHandle *h);
void      tq_results_free(TqSearchResults r);

// IdMapIndex
IdHandle  *id_new(size_t dim, uint32_t bit_width, char **err);
IdHandle  *id_new_lazy(uint32_t bit_width, char **err);
void       id_drop(IdHandle *h);
void       id_add_with_ids(IdHandle *h, const float *vectors, size_t n,
                           size_t dim, const uint64_t *ids, char **err);
bool       id_remove(IdHandle *h, uint64_t id);
bool       id_contains(const IdHandle *h, uint64_t id);
IdSearchResults id_search(const IdHandle *h, const float *queries, size_t nq,
                          size_t dim, size_t k,
                          const uint64_t *allowlist, size_t allowlist_len);
void       id_prepare(const IdHandle *h);
int32_t    id_write(const IdHandle *h, const char *path);
IdHandle  *id_load(const char *path, char **err);
size_t     id_len(const IdHandle *h);
size_t     id_dim(const IdHandle *h);
uint32_t   id_bit_width(const IdHandle *h);
void       id_results_free(IdSearchResults r);

// Cross-cutting
void      err_free(char *err);
```

### Result structs

```c
typedef struct {
    float  *scores;   // row-major (nq, k); caller frees with tq_results_free
    int64_t *indices;
    size_t   nq;
    size_t   k;
} TqSearchResults;

typedef struct {
    float   *scores;
    uint64_t *ids;    // row-major (nq, k)
    size_t    nq;
    size_t    k;
} IdSearchResults;
```

`scores` and `indices`/`ids` are `Vec`-allocated by the shim, copied out by the Go side, then freed via `tq_results_free` / `id_results_free`. The buffer is laid out as `nq * k` flat; the row stride is `k`.

### Error policy

Every fallible function takes a trailing `char **err` argument. On success, `*err` is set to `NULL`. On failure:
- The function returns a sentinel (`NULL` for handle-returning functions, `UINT64_MAX` for `tq_swap_remove`'s `usize`, `0` for length functions when the length is unknown, non-zero for the `*_write` int32 return).
- `*err` is set to a heap-allocated, NUL-terminated string. The Go side copies it, then calls `err_free` to release it.

The error string is prefixed with a stable token (`"invalid_dim:"`, `"duplicate_id:"`, `"unknown_id:"`, `"empty_allowlist:"`, `"add:"`, `"construct:"`, etc.) so the Go side can map it to a typed error sentinel. Unprefixed errors wrap into a generic `*Error` with the upstream message preserved.

### Memory ownership

- Input slices (`vectors`, `queries`, `ids`, `mask`, `allowlist`) are borrowed; the Go side keeps them alive across the call.
- Output buffers in `*_SearchResults` are owned by Rust; the Go side copies them into Go-owned slices and then calls the matching `*_results_free`.
- The string returned via `err` is owned by Rust; the Go side copies and `err_free`s.
- Handles (`*mut TqHandle`, `*mut IdHandle`) are opaque pointers into Rust-owned `Box<TurboQuantIndex>` / `Box<IdMapIndex>`. The Go side stores them and passes them back; the C side drops them via `tq_drop` / `id_drop`.

## Go API

A single top-level package: `github.com/divy/go-turbovec/turbovec`. No init, no globals, no init-order surprises.

### `TurboQuantIndex`

```go
type TurboQuantIndex struct { /* unexported, holds *C.TqHandle */ }

// Constructors
func NewTurboQuantIndex(dim int, bitWidth int) (*TurboQuantIndex, error)
func NewLazyTurboQuantIndex(bitWidth int) (*TurboQuantIndex, error)

// Mutators
func (i *TurboQuantIndex) Add(vectors []float32) error
func (i *TurboQuantIndex) AddWithDim(vectors []float32, dim int) error
func (i *TurboQuantIndex) SwapRemove(idx int) (int, error)

// Search
func (i *TurboQuantIndex) Search(queries []float32, k int) (Results, error)
func (i *TurboQuantIndex) SearchWithMask(queries []float32, k int, mask []bool) (Results, error)

// Persistence
func (i *TurboQuantIndex) Write(path string) error
func LoadTurboQuantIndex(path string) (*TurboQuantIndex, error)
func (i *TurboQuantIndex) Prepare()

// Accessors
func (i *TurboQuantIndex) Len() int
func (i *TurboQuantIndex) Dim() (int, bool)         // bool = uncommitted
func (i *TurboQuantIndex) BitWidth() int

// Lifecycle (required; backed by a finalizer safety net)
func (i *TurboQuantIndex) Close() error
```

### `IdMapIndex`

```go
type IdMapIndex struct { /* unexported, holds *C.IdHandle */ }

func NewIdMapIndex(dim int, bitWidth int) (*IdMapIndex, error)
func NewLazyIdMapIndex(bitWidth int) (*IdMapIndex, error)

func (i *IdMapIndex) AddWithIDs(vectors []float32, ids []uint64) error
func (i *IdMapIndex) AddWithIDsWithDim(vectors []float32, ids []uint64, dim int) error
func (i *IdMapIndex) Remove(id uint64) bool
func (i *IdMapIndex) Contains(id uint64) bool

func (i *IdMapIndex) Search(queries []float32, k int) (IdResults, error)
func (i *IdMapIndex) SearchWithAllowlist(queries []float32, k int, allowlist []uint64) (IdResults, error)

func (i *IdMapIndex) Write(path string) error
func LoadIdMapIndex(path string) (*IdMapIndex, error)
func (i *IdMapIndex) Prepare()

func (i *IdMapIndex) Len() int
func (i *IdMapIndex) Dim() (int, bool)
func (i *IdMapIndex) BitWidth() int
func (i *IdMapIndex) Close() error
```

### Result structs

```go
type Results struct {
    Scores  []float32  // row-major (nq, k)
    Indices []int64
    Nq, K   int
}

type IdResults struct {
    Scores []float32
    IDs    []uint64
    Nq, K  int
}

func (r Results) ScoresForQuery(qi int) []float32  // r.Scores[qi*r.K : (qi+1)*r.K]
func (r Results) IndicesForQuery(qi int) []int64
// (IdResults analog)
```

### Errors

```go
// Sentinels — match the upstream's error taxonomy
var (
    ErrInvalidDim      = errors.New("turbovec: invalid dim")
    ErrEmptyAllowlist  = errors.New("turbovec: allowlist is empty")
    ErrUnknownID       = errors.New("turbovec: allowlist contains id(s) not present in index")
    ErrDuplicateID     = errors.New("turbovec: id already present in index")
    ErrIndexOutOfRange = errors.New("turbovec: index out of range")
    ErrIO              = errors.New("turbovec: I/O error")
)

// Generic wrapper carrying the upstream message
type Error struct {
    Op   string  // "construct", "add", "search", "write", "load", "swap_remove"
    Err  error   // sentinel, may be nil
    Msg  string  // full message from the Rust side
}

func (e *Error) Error() string  // "[op]: [Msg]"
func (e *Error) Unwrap() error  // returns Err
```

`errors.Is(err, ErrInvalidDim)` works for the wrapped sentinels.

### Slice layout

`Add` and `Search` take a single flat `[]float32`. The Go package treats it as row-major, `len = rows * dim`. The cgo bridge checks the length is a multiple of the index's dim (or, for a lazy index, of the dim committed on this call), and panics-or-returns-`ErrInvalidDim` otherwise. We return an error rather than panic, matching the Python `ValueError` discipline.

`Add` is total only on an eager index: it requires `len(vectors) % dim == 0` with `dim` taken from the index. `AddWithDim` is the explicit form for lazy indices — it commits the dim on the first call and enforces `len(vectors) % dim == 0` for the given `dim` thereafter. The split keeps both signatures total, since the upstream's `Add(vectors)` API is ambiguous on a lazy index (the dim is not committed until the first add). `IdMapIndex` mirrors this with `AddWithIDs` (eager) and `AddWithIDsWithDim` (lazy / explicit dim).

## Build, test, CI

### Local build

`Makefile`:

```make
.PHONY: build test
build:
	git submodule update --init
	cd turbovec-ffi && cargo build --release
	go build ./...

test:
	git submodule update --init
	cd turbovec-ffi && cargo build --release
	go test ./...
```

`gen.go`:

```go
//go:build ignore
package main

//go:generate go run gen.go
//
// Build the C-ABI shim so cgo has a lib to link against.
```

The cgo directives on the import line:

```go
// #cgo CFLAGS: -I${SRCDIR}/../turbovec-ffi/include
// #cgo linux  LDFLAGS: -L${SRCDIR}/../turbovec-ffi/target/release -lturbovec_ffi -lopenblas -lpthread -ldl -lm
// #cgo darwin LDFLAGS: -L${SRCDIR}/../turbovec-ffi/target/release -lturbovec_ffi -framework Accelerate
import "C"
```

The shim depends on the upstream `turbovec` core's BLAS-backed matmul, so the cgo link step must resolve a BLAS provider. Linux uses OpenBLAS (install `libopenblas-dev`); macOS uses Apple's Accelerate framework. The first `Add` / `Search` is what actually pulls in the BLAS symbols, so LDFLAGS are required even if the test only exercises constructors.

### Unit tests (`turbovec/*_test.go`)

Mirror the Python `tests/` cases:

- `TestLazyDimCommitsOnFirstAdd` — lazy `NewLazyTurboQuantIndex`, first `Add` commits the dim; `Dim()` returns `(dim, true)` after, `(0, false)` before.
- `TestWrongDimQuery` — build an index with `dim=128`, query with `dim=64` → `errors.Is(err, ErrInvalidDim)`.
- `TestMaskLengthMismatch` — `SearchWithMask` with a `mask` whose length differs from `Len()` → typed error.
- `TestSwapRemove` — remove `idx=2`; the vector previously at `len-1` is now at `2`; `SwapRemove` returns the previous last index.
- `TestSwapRemoveOutOfRange` — `idx >= Len()` → `errors.Is(err, ErrIndexOutOfRange)`.
- `TestIdMapDuplicateID` — `AddWithIDs` with a repeated id → `errors.Is(err, ErrDuplicateID)`.
- `TestIdMapUnknownAllowlistID` — `SearchWithAllowlist` with a non-present id → `errors.Is(err, ErrUnknownID)`; error message preview shows up to 5 unknown ids.
- `TestIdMapEmptyAllowlist` — `SearchWithAllowlist([]queries, k, []uint64{})` → `errors.Is(err, ErrEmptyAllowlist)`.
- `TestWriteLoadRoundTrip` — write a populated index, load it, search, results match.
- `TestCloseIsIdempotent` — second `Close()` returns nil (or a typed "already closed" — design choice documented in code).
- `TestFinalizerReleases` — drop an index without `Close()`, the finalizer runs without crashing in a smoke test (best-effort; GC timing is not deterministic).

### Integration test (`turbovec_test/`)

`TestRecallAt10` — builds a random corpus (50K unit vectors, dim=128), ingests it, queries 1K held-out queries, and computes recall@10 against a brute-force Go baseline. Threshold: 0.90 recall@10 at `bit_width=4` on random unit vectors. Mirrors the spirit of `benchmarks/suite/recall_*.py` without depending on the Python harness.

`TestIdMapRoundTrip` — similar but with synthetic `uint64` ids and an allowlist-restricted search.

### CI

GitHub Actions matrix: `{os: ubuntu-latest, macos-latest}` × `{arch: amd64}` plus `{os: macos-latest, arch: arm64}`. Each job:
1. Check out with `submodules: recursive`.
2. Install Rust toolchain (stable).
3. `make test`.
4. On `main`, upload `turbovec_test` binary as a release artifact.

We do **not** commit pre-built `.a` files in v1. `cargo build --release` is fast enough for CI. A future v2 may add a release workflow that publishes pre-built artifacts for `go install`-style consumption.

### Runtime dependencies

- Go 1.22+ (uses `runtime.SetFinalizer` and the modern `errors.Join` if we ever need it).
- CGO_ENABLED=1 (default on most systems).
- A C toolchain and a Rust toolchain on the build host.

No third-party Go dependencies for the runtime package. Only stdlib: `errors`, `runtime`, `unsafe`, `os`, plus cgo.

## Lifecycle and memory safety

- `*TurboQuantIndex` and `*IdMapIndex` own a C-side heap allocation. `Close()` releases it. We also install a `runtime.SetFinalizer` that calls `Close` if the user has not, as a safety net. The finalizer warns via `os.Stderr` if the user leaked the handle; it does not panic. This is best-effort: a finalizer may never run before program exit, but it ensures the C free happens at some point in tests and long-lived programs.
- `*Results` and `*IdResults` own only Go-allocated slices. No finalizer needed. The C-side buffers are copied out and freed in the call that produced them.
- Errors are returned as `*Error` values; no C-side lifetime for them.

## Open questions / future work

- A future v2 could vendor pre-built `.a` artifacts per `(os, arch)` tuple to remove the Rust toolchain requirement for end users. CI is the natural place to produce them.
- Framework integrations (e.g., a `vectorstores` interface analogous to LangChain's `VectorStore`) are out of scope for v1.
- A small `cmd/tvbench` CLI tool wrapping the integration test for smoke-testing the static build is a likely v1.1.

## Acceptance criteria

A user can:

1. `git clone` the repo, run `make build`, and produce a Go binary that statically links the Rust core.
2. From a Go program, do:
    ```go
    idx, _ := turbovec.NewTurboQuantIndex(128, 4)
    idx.Add(vectors)                                  // []float32
    r, _ := idx.Search(queries, 10)
    defer idx.Close()
    ```
   and get the same recall / latency as the Python package on identical inputs.
3. Round-trip a `.tq` and a `.tvim` file written by the Python package — load and query it from Go.
4. `go test ./...` passes on Linux amd64, Linux arm64 (if available in CI), and macOS arm64.
5. `TestRecallAt10` ≥ 0.90 at `bit_width=4` on a deterministic 50K random unit-vector corpus.
