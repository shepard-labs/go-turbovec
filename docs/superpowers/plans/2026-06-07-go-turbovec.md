# go-turbovec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Go package (`go-turbovec`) that re-exposes the `RyanCodrai/turbovec` Rust vector index under idiomatic Go types, replacing the upstream's Python piece with a C-ABI shim + cgo binding.

**Architecture:** Add a `turbovec-ffi` Rust crate in this repo (sibling of the Go package) that wraps the upstream `turbovec` core behind an `extern "C"` surface. Build it as a `staticlib`. The Go package uses cgo to link the static lib and exposes two index types (`TurboQuantIndex`, `IdMapIndex`) with method names and semantics that mirror the upstream's Python wrapper 1:1. The upstream is vendored as a git submodule pinned to a known commit. All quantization, SIMD, and codebook logic stays in the Rust core — the shim is a thin translator.

**Tech Stack:** Rust (cdylib + staticlib, no extra deps beyond `turbovec` itself), cgo, Go 1.22+ stdlib only, GNU Make, GitHub Actions. BLAS resolution is needed at the cgo link step (Accelerate on macOS, OpenBLAS on Linux).

---

## File Structure

| Path | Created in task | Responsibility |
|---|---|---|
| `turbovec-ffi/Cargo.toml` | T1 | Shim crate manifest, depends on the upstream submodule |
| `turbovec-ffi/include/cabi.h` | T1, T2 | Hand-written C header for the FFI surface (consumed by cgo) |
| `turbovec-ffi/src/lib.rs` | T2, T3, T4, T5, T6 | The `extern "C"` surface: handles, mutators, search, persistence, error helpers |
| `turbovec-ffi/src/error.rs` | T2 | Stable error-prefix catalog; maps `AddError`/`ConstructError` to prefixed C strings |
| `.gitmodules` | T1 | Pins the upstream submodule |
| `go.mod` | T1 | Go module declaration |
| `gen.go` | T7 | `//go:generate` directive that builds the shim |
| `Makefile` | T7 | `make build` and `make test` |
| `turbovec/errors.go` | T8 | Sentinel errors, `*Error` type, `errors.Is` support |
| `turbovec/cgo.go` | T8 | cgo bridge, error-string helper, finalizer-safe handle release |
| `turbovec/turboquant_index.go` | T9, T10, T11, T12 | `TurboQuantIndex` constructor, accessors, mutator, search, persistence |
| `turbovec/id_map_index.go` | T13, T14, T15, T16 | `IdMapIndex` constructor, accessors, mutator, search, persistence |
| `turbovec/search_results.go` | T9 | `Results` and `IdResults` structs, row accessors |
| `turbovec/turboquant_index_test.go` | T9, T10, T11, T12 | Unit tests for `TurboQuantIndex` |
| `turbovec/id_map_index_test.go` | T13, T14, T15, T16 | Unit tests for `IdMapIndex` |
| `turbovec_test/integration_test.go` | T17 | Recall@10 test against a Go brute-force baseline |
| `.github/workflows/ci.yml` | T18 | CI: rust + go on linux/macos |

Each file is focused on one concern. The shim's `lib.rs` is the largest file by far (~400 lines), and even then it's mostly a 1:1 translation of the upstream's `TurboQuantIndex`/`IdMapIndex` methods.

---

## Task 1: Scaffold the repo and submodule

**Files:**
- Create: `.gitmodules`
- Create: `go.mod`
- Create: `.gitignore`
- Create: `turbovec-ffi/Cargo.toml`
- Create: `turbovec-ffi/include/.gitkeep`
- Create: `turbovec-ffi/src/.gitkeep`

- [ ] **Step 1: Pick and pin a known commit of the upstream**

Run:

```bash
git submodule add https://github.com/RyanCodrai/turbovec.git turbovec
cd turbovec
git checkout <PINNED_COMMIT>
cd ..
git add .gitmodules turbovec
git commit -m "chore: vendor upstream turbovec as submodule at <PINNED_COMMIT>"
```

Use the commit that corresponds to `turbovec` crate version `0.8.0` (the version present in the `Cargo.toml` we read during exploration). Capture the full SHA and put it in the commit message.

- [ ] **Step 2: Add the Go module**

Create `go.mod` with:

```go
module github.com/divy/go-turbovec

go 1.22
```

(`github.com/divy/go-turbovec` is a placeholder; replace with the actual module path the user wants. The plan assumes this path throughout.)

- [ ] **Step 3: Add a minimal `.gitignore`**

```gitignore
# Rust build output
turbovec-ffi/target/

# Go build output
*.test
*.out
bin/
```

- [ ] **Step 4: Create the shim crate skeleton**

Create `turbovec-ffi/Cargo.toml` with:

```toml
[package]
name = "turbovec-ffi"
version = "0.1.0"
edition = "2021"
rust-version = "1.70"
license = "MIT"
description = "C-ABI shim around the turbovec Rust core for go-turbovec"

[lib]
crate-type = ["staticlib", "cdylib"]

[dependencies]
turbovec = { path = "../turbovec/turbovec" }
```

- [ ] **Step 5: Create empty source dirs**

```bash
mkdir -p turbovec-ffi/include turbovec-ffi/src
touch turbovec-ffi/include/.gitkeep turbovec-ffi/src/.gitkeep
```

- [ ] **Step 6: Verify the crate builds (sanity)**

Run:

```bash
cd turbovec-ffi && cargo build --release
```

Expected: `Finished release [optimized] target(s)` with no errors. The output `target/release/libturbovec_ffi.a` (and `.dylib` on macOS) should exist.

- [ ] **Step 7: Commit**

```bash
git add go.mod .gitignore turbovec-ffi/Cargo.toml turbovec-ffi/include/.gitkeep turbovec-ffi/src/.gitkeep
git commit -m "chore: scaffold go-turbovec with shim crate skeleton"
```

---

## Task 2: Define the C-ABI surface — header and error mapping

**Files:**
- Create: `turbovec-ffi/include/cabi.h`
- Create: `turbovec-ffi/src/error.rs`
- Create: `turbovec-ffi/src/lib.rs` (skeleton: re-exports, no FFI yet beyond `err_free`)

This task defines the contract. The actual FFI functions are filled in by tasks 3–6.

- [ ] **Step 1: Write `cabi.h`**

```c
// turbovec-ffi/include/cabi.h
// C-ABI surface for go-turbovec. Consumed by cgo via #cgo CFLAGS.
// Hand-written; keep in sync with turbovec-ffi/src/lib.rs.
//
// Opaque handle types. The shim owns a Box<...> behind each one.
typedef struct TqHandle TqHandle;
typedef struct IdHandle IdHandle;

// Search result structs. Buffers are flat, row-major (nq, k).
// Caller frees with tq_results_free / id_results_free.
typedef struct {
    float  *scores;   // length nq * k
    int64_t *indices; // length nq * k
    size_t   nq;
    size_t   k;
} TqSearchResults;

typedef struct {
    float   *scores;  // length nq * k
    uint64_t *ids;     // length nq * k
    size_t    nq;
    size_t    k;
} IdSearchResults;

// TurboQuantIndex
TqHandle *tq_new(size_t dim, uint32_t bit_width, char **err);
TqHandle *tq_new_lazy(uint32_t bit_width, char **err);
void      tq_drop(TqHandle *h);
void      tq_add(TqHandle *h, const float *vectors, size_t n, size_t dim, char **err);
TqSearchResults tq_search(const TqHandle *h, const float *queries, size_t nq,
                          size_t dim, size_t k, const uint8_t *mask /* nullable */);
size_t    tq_swap_remove(TqHandle *h, size_t idx, char **err);
void      tq_prepare(const TqHandle *h);
int32_t   tq_write(const TqHandle *h, const char *path);
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
void err_free(char *err);
```

- [ ] **Step 2: Create `error.rs` with the prefix catalog**

```rust
// turbovec-ffi/src/error.rs

//! Stable error-prefix catalog used by every fallible FFI call.
//!
//! Each fallible FFI function takes a trailing `char **err`. On failure it
//! writes a heap-allocated, NUL-terminated C string of the form
//! `"<prefix>: <message>"`. The Go side reads the prefix and maps it to a
//! typed sentinel. The upstream `Display` message follows the colon and is
//! preserved verbatim in `Error.Msg` for diagnostics.

pub const PREFIX_INVALID_DIM: &str = "invalid_dim";
pub const PREFIX_BIT_WIDTH: &str = "bit_width";
pub const PREFIX_DIM_NOT_MULTIPLE_OF_8: &str = "dim_not_multiple_of_8";
pub const PREFIX_BUFFER_NOT_MULTIPLE_OF_DIM: &str = "buffer_not_multiple_of_dim";
pub const PREFIX_IDS_COUNT_MISMATCH: &str = "ids_count_mismatch";
pub const PREFIX_DUPLICATE_ID: &str = "duplicate_id";
pub const PREFIX_INVALID_INPUT_VALUE: &str = "invalid_input_value";
pub const PREFIX_EMPTY_ALLOWLIST: &str = "empty_allowlist";
pub const PREFIX_UNKNOWN_ID: &str = "unknown_id";
pub const PREFIX_INDEX_OUT_OF_RANGE: &str = "index_out_of_range";
pub const PREFIX_IO: &str = "io";
```

- [ ] **Step 3: Skeleton `lib.rs` with the `err_free` helper and re-export**

```rust
// turbovec-ffi/src/lib.rs
//
// C-ABI shim around the upstream `turbovec` Rust core. Consumed by the
// go-turbovec Go package via cgo. See include/cabi.h for the surface.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

pub mod error;

/// Free a C string previously written into a `char **err` out-parameter.
#[no_mangle]
pub extern "C" fn err_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// Read a borrowed C string into an owned Rust String. Returns Err on null
/// or invalid UTF-8.
pub(crate) fn cstr_to_string(p: *const c_char) -> Result<String, std::ffi::NulError> {
    if p.is_null() {
        return Err(std::ffi::NulError {
            // The inner field is private; construct via from_ptr below.
            // In practice we never hit this branch in the shim.
        });
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map(str::to_string)
        .map_err(|_| std::ffi::NulError)
}

/// Convert a `Result<T, E>` whose error implements `Display` into either
/// `Ok(T)` or a heap-allocated C string written into `*err` and a sentinel
/// return value. Use this from every fallible FFI function.
pub(crate) fn map_err<T, E: std::fmt::Display, F>(
    r: Result<T, E>,
    err_out: *mut *mut c_char,
    sentinel: F,
) -> Option<T>
where
    F: FnOnce() -> T,
{
    match r {
        Ok(v) => {
            if !err_out.is_null() {
                unsafe { *err_out = std::ptr::null_mut() };
            }
            Some(v)
        }
        Err(e) => {
            if !err_out.is_null() {
                let s = CString::new(format!("{}", e)).unwrap_or_else(|_| CString::new("error").unwrap());
                unsafe { *err_out = s.into_raw() };
            }
            None
        }
    }
    .map(|_| ())
    .and(Some(r.unwrap_or_else(|_| sentinel())))
}
```

The `map_err` helper is intentionally awkward in v1; the shim is small enough that hand-rolling the `match` per function is also fine. Replace this helper with a small macro once we have ≥3 callers in tasks 3–6.

For now, replace the body of `map_err` with this simpler version and let each FFI function build its own `*err` string with a prefix:

```rust
// simpler: each FFI function matches on its own Result and writes its own
// prefixed string. Keep this helper for the no-prefix case only.

/// Build a prefixed C string for the err out-parameter. Returns a raw
/// pointer suitable for `*err_out`; the caller is responsible for handing
/// it back to `err_free`.
pub(crate) fn prefixed_err(prefix: &str, msg: impl std::fmt::Display) -> *mut c_char {
    let s = format!("{}: {}", prefix, msg);
    CString::new(s).unwrap_or_else(|_| CString::new(prefix).unwrap()).into_raw()
}
```

Replace the original `map_err` definition with just `prefixed_err` and a 2-line note. This is cleaner.

- [ ] **Step 4: Verify the shim still builds with the helper**

Run:

```bash
cd turbovec-ffi && cargo build --release
```

Expected: builds cleanly, no warnings beyond the `unused` warning for the helper (we'll suppress it with `#[allow(dead_code)]` for now).

- [ ] **Step 5: Commit**

```bash
git add turbovec-ffi/include/cabi.h turbovec-ffi/src/error.rs turbovec-ffi/src/lib.rs
git commit -m "feat(ffi): add C-ABI header and error-prefix helper"
```

---

## Task 3: FFI for `TurboQuantIndex` — constructors, accessors, drop

**Files:**
- Modify: `turbovec-ffi/src/lib.rs` (add the `TqHandle` opaque type and the constructor/accessor/drop functions)
- Create: `turbovec/cgo.go` (minimal: cgo import, error-mapper, but no Go types yet)
- Create: `turbovec/errors.go`
- Create: `turbovec/turboquant_index.go` (struct + constructor + accessors + Close)
- Create: `turbovec/turboquant_index_test.go`

- [ ] **Step 1: Write the failing Go test first (TDD)**

Create `turbovec/turboquant_index_test.go`:

```go
package turbovec

import (
    "path/filepath"
    "testing"
)

func TestNewTurboQuantIndex_Eager(t *testing.T) {
    idx, err := NewTurboQuantIndex(128, 4)
    if err != nil {
        t.Fatalf("NewTurboQuantIndex: %v", err)
    }
    defer idx.Close()
    if got, _ := idx.Dim(); got != 128 {
        t.Errorf("Dim() = %d, want 128", got)
    }
    if got := idx.BitWidth(); got != 4 {
        t.Errorf("BitWidth() = %d, want 4", got)
    }
    if got := idx.Len(); got != 0 {
        t.Errorf("Len() = %d, want 0", got)
    }
}

func TestNewTurboQuantIndex_InvalidBitWidth(t *testing.T) {
    _, err := NewTurboQuantIndex(128, 7)
    if err == nil {
        t.Fatal("want error for bit_width=7, got nil")
    }
    if !errorsIs(err, ErrInvalidBitWidth) {
        t.Errorf("err = %v, want errors.Is(ErrInvalidBitWidth)", err)
    }
}

func TestNewTurboQuantIndex_InvalidDim(t *testing.T) {
    _, err := NewTurboQuantIndex(7, 4) // not a multiple of 8
    if err == nil {
        t.Fatal("want error for dim=7, got nil")
    }
    if !errorsIs(err, ErrInvalidDim) {
        t.Errorf("err = %v, want errors.Is(ErrInvalidDim)", err)
    }
}

func TestNewLazyTurboQuantIndex(t *testing.T) {
    idx, err := NewLazyTurboQuantIndex(4)
    if err != nil {
        t.Fatalf("NewLazyTurboQuantIndex: %v", err)
    }
    defer idx.Close()
    if got, ok := idx.Dim(); ok {
        t.Errorf("Dim() committed before add: got (%d, %v), want (0, false)", got, ok)
    }
}

func TestTurboQuantIndex_Close_Idempotent(t *testing.T) {
    idx, err := NewTurboQuantIndex(64, 4)
    if err != nil {
        t.Fatalf("NewTurboQuantIndex: %v", err)
    }
    if err := idx.Close(); err != nil {
        t.Fatalf("first Close: %v", err)
    }
    if err := idx.Close(); err != nil {
        t.Errorf("second Close: %v, want nil", err)
    }
}

// --- write/load round trip will be added in Task 12; we just need the
// helper to create a tmp dir for later tests. ---

func tmpDir(t *testing.T) string {
    t.Helper()
    d := t.TempDir()
    return filepath.Clean(d)
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run:

```bash
make test 2>&1 | head -40
```

Expected: the `go test ./...` step fails to compile because `turbovec/turboquant_index.go` does not exist yet. The error should mention `undefined: NewTurboQuantIndex`, `undefined: NewLazyTurboQuantIndex`, `undefined: ErrInvalidBitWidth`, `undefined: ErrInvalidDim`, `undefined: errorsIs`.

- [ ] **Step 3: Implement the FFI side — `TqHandle` and constructor/accessor/drop**

Append to `turbovec-ffi/src/lib.rs`:

```rust
use turbovec::TurboQuantIndex as CoreTq;
use std::sync::Arc;

/// Opaque handle to a heap-allocated `TurboQuantIndex`.
#[repr(C)]
pub struct TqHandle {
    inner: Box<CoreTq>,
}

#[no_mangle]
pub extern "C" fn tq_new(dim: usize, bit_width: u32, err: *mut *mut c_char) -> *mut TqHandle {
    let res = CoreTq::new(dim, bit_width as usize).map(Box::new).map(|b| Box::into_raw(Box::new(TqHandle { inner: b })));
    match res {
        Ok(h) => {
            if !err.is_null() { unsafe { *err = std::ptr::null_mut() } }
            h
        }
        Err(e) => {
            let prefix = match &e {
                turbovec::ConstructError::BitWidthOutOfRange(_) => error::PREFIX_BIT_WIDTH,
                turbovec::ConstructError::DimNotPositiveMultipleOf8(_) => error::PREFIX_DIM_NOT_MULTIPLE_OF_8,
            };
            if !err.is_null() {
                unsafe { *err = prefixed_err(prefix, e) }
            }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tq_new_lazy(bit_width: u32, err: *mut *mut c_char) -> *mut TqHandle {
    let res = CoreTq::new_lazy(bit_width as usize)
        .map(|b| Box::into_raw(Box::new(TqHandle { inner: Box::new(b) })));
    match res {
        Ok(h) => {
            if !err.is_null() { unsafe { *err = std::ptr::null_mut() } }
            h
        }
        Err(e) => {
            if !err.is_null() { unsafe { *err = prefixed_err(error::PREFIX_BIT_WIDTH, e) } }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tq_drop(h: *mut TqHandle) {
    if !h.is_null() {
        unsafe { drop(Box::from_raw(h)) }
    }
}

#[no_mangle]
pub extern "C" fn tq_len(h: *const TqHandle) -> usize {
    unsafe { (*h).inner.len() }
}

#[no_mangle]
pub extern "C" fn tq_dim(h: *const TqHandle) -> usize {
    unsafe { (*h).inner.dim() } // returns 0 for uncommitted lazy
}

#[no_mangle]
pub extern "C" fn tq_bit_width(h: *const TqHandle) -> u32 {
    unsafe { (*h).inner.bit_width() as u32 }
}
```

(The `Arc` import is intentional noise from a draft — drop it. Only `Box` is needed.)

- [ ] **Step 4: Implement the Go side — `errors.go`, `cgo.go`, `turboquant_index.go`**

Create `turbovec/errors.go`:

```go
package turbovec

import (
    "errors"
    "fmt"
    "strings"
)

// Sentinels — match the upstream's error taxonomy. Wrap with *Error via
// errors.Is / errors.As.
var (
    ErrInvalidDim          = errors.New("turbovec: invalid dim")
    ErrInvalidBitWidth     = errors.New("turbovec: invalid bit width (must be 2, 3, or 4)")
    ErrBufferNotMultipleOfDim = errors.New("turbovec: vector buffer length is not a multiple of dim")
    ErrIDsCountMismatch    = errors.New("turbovec: ids count does not match vectors length")
    ErrDuplicateID         = errors.New("turbovec: id already present in index")
    ErrInvalidInputValue   = errors.New("turbovec: input contains non-finite or out-of-range value")
    ErrEmptyAllowlist      = errors.New("turbovec: allowlist is empty")
    ErrUnknownID           = errors.New("turbovec: allowlist contains id(s) not present in index")
    ErrIndexOutOfRange     = errors.New("turbovec: index out of range")
    ErrIO                  = errors.New("turbovec: I/O error")
)

// Error wraps an upstream error with its operation context.
type Error struct {
    Op  string
    Err error
    Msg string
}

func (e *Error) Error() string {
    if e.Err != nil {
        return fmt.Sprintf("turbovec: %s: %s", e.Op, e.Err.Error())
    }
    return fmt.Sprintf("turbovec: %s: %s", e.Op, e.Msg)
}

func (e *Error) Unwrap() error { return e.Err }

// mapErr converts a "<prefix>: <message>" string returned by the FFI shim
// into a typed Go error. Unknown prefixes become a generic *Error with
// Err=nil and Msg=full-string.
func mapErr(op string, raw string) error {
    if raw == "" {
        return nil
    }
    var sentinel error
    switch {
    case strings.HasPrefix(raw, errorPrefixInvalidDim):
        sentinel = ErrInvalidDim
    case strings.HasPrefix(raw, errorPrefixBitWidth):
        sentinel = ErrInvalidBitWidth
    case strings.HasPrefix(raw, errorPrefixBufferNotMultipleOfDim):
        sentinel = ErrBufferNotMultipleOfDim
    case strings.HasPrefix(raw, errorPrefixIDsCountMismatch):
        sentinel = ErrIDsCountMismatch
    case strings.HasPrefix(raw, errorPrefixDuplicateID):
        sentinel = ErrDuplicateID
    case strings.HasPrefix(raw, errorPrefixInvalidInputValue):
        sentinel = ErrInvalidInputValue
    case strings.HasPrefix(raw, errorPrefixEmptyAllowlist):
        sentinel = ErrEmptyAllowlist
    case strings.HasPrefix(raw, errorPrefixUnknownID):
        sentinel = ErrUnknownID
    case strings.HasPrefix(raw, errorPrefixIndexOutOfRange):
        sentinel = ErrIndexOutOfRange
    case strings.HasPrefix(raw, errorPrefixIO):
        sentinel = ErrIO
    }
    msg := strings.TrimSpace(strings.SplitN(raw, ":", 2)[1])
    return &Error{Op: op, Err: sentinel, Msg: msg}
}
```

Create `turbovec/cgo.go` with the cgo bridge and prefix constants:

```go
package turbovec

/*
#include <stdlib.h>
#include "cabi.h"
*/
import "C"

import (
    "unsafe"
)

// Error-prefix constants — must match turbovec-ffi/src/error.rs.
const (
    errorPrefixInvalidDim               = "invalid_dim"
    errorPrefixBitWidth                 = "bit_width"
    errorPrefixDimNotMultipleOf8        = "dim_not_multiple_of_8"
    errorPrefixBufferNotMultipleOfDim   = "buffer_not_multiple_of_dim"
    errorPrefixIDsCountMismatch         = "ids_count_mismatch"
    errorPrefixDuplicateID              = "duplicate_id"
    errorPrefixInvalidInputValue        = "invalid_input_value"
    errorPrefixEmptyAllowlist           = "empty_allowlist"
    errorPrefixUnknownID                = "unknown_id"
    errorPrefixIndexOutOfRange          = "index_out_of_range"
    errorPrefixIO                       = "io"
)

// takeErr reads a `char **err` out-parameter, copies the string into Go,
// frees the C string, and returns a typed Go error (or nil).
func takeErr(cerr *C.char) error {
    if cerr == nil {
        return nil
    }
    s := C.GoString(cerr)
    C.err_free(cerr)
    return mapErr("", s)
}

// stringToC returns a heap-allocated C string. Caller is responsible for
// calling C.free on the returned pointer.
func stringToC(s string) *C.char {
    return C.CString(s)
}

// boolSliceToCBytes copies a Go []bool into a C-allocated u8 buffer.
// The caller must call C.free on the returned pointer.
func boolSliceToCBytes(b []bool) (*C.uint8_t, error) {
    if len(b) == 0 {
        return nil, nil
    }
    out := C.calloc(C.size_t(len(b)), 1)
    if out == nil {
        return nil, errAlloc
    }
    for i, v := range b {
        if v {
            *(*C.uint8_t)(unsafe.Pointer(uintptr(out) + uintptr(i))) = 1
        }
    }
    return (*C.uint8_t)(out), nil
}

var errAlloc = &Error{Op: "cgo", Msg: "calloc returned null"}
```

Create `turbovec/turboquant_index.go`:

```go
package turbovec

/*
#include <stdlib.h>
#include "cabi.h"
*/
import "C"

import (
    "runtime"
    "unsafe"
)

// TurboQuantIndex is the basic top-k index. See upstream docs.
type TurboQuantIndex struct {
    h *C.TqHandle
}

// NewTurboQuantIndex constructs an index with the given dim and bit width.
// bit_width must be 2, 3, or 4. dim must be a positive multiple of 8.
func NewTurboQuantIndex(dim, bitWidth int) (*TurboQuantIndex, error) {
    var cerr *C.char
    h := C.tq_new(C.size_t(dim), C.uint32_t(bitWidth), &cerr)
    if err := takeErr(cerr); err != nil {
        return nil, &Error{Op: "NewTurboQuantIndex", Err: err, Msg: err.Error()}
    }
    if h == nil {
        return nil, &Error{Op: "NewTurboQuantIndex", Msg: "tq_new returned null without error"}
    }
    return newTurboQuantIndex(h), nil
}

// NewLazyTurboQuantIndex constructs an index whose dim is determined by
// the first Add. bit_width must be 2, 3, or 4.
func NewLazyTurboQuantIndex(bitWidth int) (*TurboQuantIndex, error) {
    var cerr *C.char
    h := C.tq_new_lazy(C.uint32_t(bitWidth), &cerr)
    if err := takeErr(cerr); err != nil {
        return nil, &Error{Op: "NewLazyTurboQuantIndex", Err: err, Msg: err.Error()}
    }
    if h == nil {
        return nil, &Error{Op: "NewLazyTurboQuantIndex", Msg: "tq_new_lazy returned null without error"}
    }
    return newTurboQuantIndex(h), nil
}

func newTurboQuantIndex(h *C.TqHandle) *TurboQuantIndex {
    idx := &TurboQuantIndex{h: h}
    runtime.SetFinalizer(idx, func(i *TurboQuantIndex) {
        // Safety net: warn to stderr and free the C handle. A finalizer
        // may never run before program exit; Close is still required.
        if i.h != nil {
            C.tq_drop(i.h)
            i.h = nil
        }
    })
    return idx
}

// Len returns the number of vectors currently in the index.
func (i *TurboQuantIndex) Len() int { return int(C.tq_len(i.h)) }

// Dim returns the index's vector dimensionality. The second return value
// is false when the index is lazy and has not yet seen an Add.
func (i *TurboQuantIndex) Dim() (int, bool) {
    d := C.tq_dim(i.h)
    if d == 0 {
        return 0, false
    }
    return int(d), true
}

// BitWidth returns the bit width used for quantization (2, 3, or 4).
func (i *TurboQuantIndex) BitWidth() int { return int(C.tq_bit_width(i.h)) }

// Close releases the C-side handle. Calling Close more than once is a
// no-op after the first.
func (i *TurboQuantIndex) Close() error {
    if i.h == nil {
        return nil
    }
    C.tq_drop(i.h)
    i.h = nil
    runtime.SetFinalizer(i, nil)
    return nil
}

// helpers used in later tasks
func cstringsToC(s []string) **C.char {
    if len(s) == 0 {
        return nil
    }
    out := C.calloc(C.size_t(len(s)), C.size_t(unsafe.Sizeof((*C.char)(nil))))
    return (**C.char)(out)
}
```

(The `cstringsToC` helper is included preemptively for Task 12's `Write`/`Load` methods. It's unused for now; suppress the `unused` warning by adding `//nolint:unused` or just leaving it — Go's compiler doesn't warn on unused functions, only unused variables and imports.)

- [ ] **Step 5: Wire up the build (Makefile + gen.go)**

This is also task 7's content, but we need it for tests to run, so a minimal version goes here. Create `Makefile`:

```make
.PHONY: build test generate

generate:
	cd turbovec-ffi && cargo build --release

build: generate
	go build ./...

test: generate
	go test ./...
```

Create `gen.go`:

```go
//go:build ignore
// +build ignore

package main

// Build the C-ABI shim so cgo has a lib to link against.
//go:generate go run gen.go
func main() {
    // Intentionally empty: `make generate` (cargo build) is what actually
    // builds the lib. This file exists so `go generate ./...` is a no-op
    // and developers are pointed at the Makefile.
}
```

- [ ] **Step 6: Run tests and verify they pass**

Run:

```bash
make test
```

Expected: all four tests pass. `go vet` clean.

- [ ] **Step 7: Commit**

```bash
git add turbovec-ffi/src/lib.rs turbovec/errors.go turbovec/cgo.go turbovec/turboquant_index.go turbovec/turboquant_index_test.go Makefile gen.go
git commit -m "feat(tq): expose TurboQuantIndex constructor, accessors, Close"
```

---

## Task 4: FFI for `TurboQuantIndex` — `Add`

**Files:**
- Modify: `turbovec-ffi/src/lib.rs` (add `tq_add`)
- Modify: `turbovec/turboquant_index.go` (add `Add` method)
- Modify: `turbovec/turboquant_index_test.go` (add tests for add + lazy dim commit)

- [ ] **Step 1: Write the failing Go tests**

Append to `turbovec/turboquant_index_test.go`:

```go
func TestTurboQuantIndex_Add_Eager(t *testing.T) {
    idx, err := NewTurboQuantIndex(8, 4)
    if err != nil {
        t.Fatalf("NewTurboQuantIndex: %v", err)
    }
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0}
    if err := idx.Add(vecs); err != nil {
        t.Fatalf("Add: %v", err)
    }
    if got := idx.Len(); got != 2 {
        t.Errorf("Len() = %d, want 2", got)
    }
}

func TestTurboQuantIndex_Add_LazyCommitsDim(t *testing.T) {
    idx, err := NewLazyTurboQuantIndex(4)
    if err != nil {
        t.Fatalf("NewLazyTurboQuantIndex: %v", err)
    }
    defer idx.Close()
    if d, ok := idx.Dim(); ok {
        t.Fatalf("Dim() committed before add: (%d, %v)", d, ok)
    }
    vecs := make([]float32, 16) // 2 vectors of dim 8
    if err := idx.Add(vecs); err != nil {
        t.Fatalf("Add: %v", err)
    }
    if d, ok := idx.Dim(); !ok || d != 8 {
        t.Errorf("Dim() after first add = (%d, %v), want (8, true)", d, ok)
    }
}

func TestTurboQuantIndex_Add_DimMismatch(t *testing.T) {
    idx, _ := NewTurboQuantIndex(8, 4)
    defer idx.Close()
    // 4 vectors of dim 4 — wrong dim
    vecs := make([]float32, 16)
    if err := idx.Add(vecs); !errorsIs(err, ErrInvalidDim) {
        t.Errorf("err = %v, want errors.Is(ErrInvalidDim)", err)
    }
}

func TestTurboQuantIndex_Add_BufferNotMultipleOfDim(t *testing.T) {
    idx, _ := NewTurboQuantIndex(8, 4)
    defer idx.Close()
    vecs := make([]float32, 7) // not a multiple of 8
    if err := idx.Add(vecs); !errorsIs(err, ErrBufferNotMultipleOfDim) {
        t.Errorf("err = %v, want errors.Is(ErrBufferNotMultipleOfDim)", err)
    }
}

func TestTurboQuantIndex_Add_InvalidInputValue(t *testing.T) {
    idx, _ := NewTurboQuantIndex(8, 4)
    defer idx.Close()
    vecs := []float32{float32.NaN(), 0, 0, 0, 0, 0, 0, 0}
    if err := idx.Add(vecs); !errorsIs(err, ErrInvalidInputValue) {
        t.Errorf("err = %v, want errors.Is(ErrInvalidInputValue)", err)
    }
}
```

Add the `errorsIs` helper at the bottom of the same file:

```go
// errorsIs is a tiny shim that lets test files avoid an extra import
// collision with the `errors` package. It is `errors.Is` verbatim.
func errorsIs(err, target error) bool {
    return errors.Is(err, target)
}
```

Add `"errors"` to the import block of `turboquant_index_test.go`.

- [ ] **Step 2: Run tests to verify they fail to compile**

Run:

```bash
make test 2>&1 | head -40
```

Expected: compile error `i.Add undefined` and `errorsIs undefined`.

- [ ] **Step 3: Implement `tq_add` in the shim**

Append to `turbovec-ffi/src/lib.rs`:

```rust
#[no_mangle]
pub extern "C" fn tq_add(
    h: *mut TqHandle,
    vectors: *const f32,
    n: usize,
    dim: usize,
    err: *mut *mut c_char,
) {
    if h.is_null() || vectors.is_null() {
        if !err.is_null() {
            unsafe { *err = prefixed_err(error::PREFIX_INVALID_DIM, "null handle or buffer") }
        }
        return;
    }
    if n == 0 {
        // Nothing to add; match the upstream's empty-add behavior (commits
        // TQ+ shift/scale to identity). The Go side treats this as a
        // success.
        if !err.is_null() { unsafe { *err = std::ptr::null_mut() } }
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts(vectors, n * dim) };
    let res = unsafe { (*h).inner.add_2d(slice, dim) };
    if !err.is_null() {
        match res {
            Ok(()) => unsafe { *err = std::ptr::null_mut() },
            Err(e) => {
                let prefix = match &e {
                    turbovec::AddError::DimMismatch { .. } => error::PREFIX_INVALID_DIM,
                    turbovec::AddError::DimNotMultipleOf8(_) => error::PREFIX_DIM_NOT_MULTIPLE_OF_8,
                    turbovec::AddError::VectorBufferNotMultipleOfDim { .. } => error::PREFIX_BUFFER_NOT_MULTIPLE_OF_DIM,
                    turbovec::AddError::IdsCountMismatch { .. } => error::PREFIX_IDS_COUNT_MISMATCH,
                    turbovec::AddError::IdAlreadyPresent(_) => error::PREFIX_DUPLICATE_ID,
                    turbovec::AddError::InvalidInputValue { .. } => error::PREFIX_INVALID_INPUT_VALUE,
                };
                unsafe { *err = prefixed_err(prefix, e) }
            }
        }
    }
}
```

- [ ] **Step 4: Implement `Add` in Go**

Append to `turbovec/turboquant_index.go`:

```go
// Add inserts n = len(vectors)/dim vectors into the index. On a lazy
// index, the first Add commits the dim.
//
// The returned error wraps a typed sentinel: ErrInvalidDim,
// ErrBufferNotMultipleOfDim, ErrInvalidInputValue.
func (i *TurboQuantIndex) Add(vectors []float32) error {
    if i.h == nil {
        return &Error{Op: "Add", Msg: "index is closed"}
    }
    if len(vectors) == 0 {
        return nil
    }
    dim, _ := i.Dim()
    if dim == 0 {
        // Lazy: caller must use AddWithDim instead. We reject here to
        // avoid an ambiguous contract; AddWithDim is added in step 5.
        return &Error{Op: "Add", Msg: "dim not committed; use AddWithDim on a lazy index"}
    }
    if len(vectors)%dim != 0 {
        return &Error{Op: "Add", Err: ErrBufferNotMultipleOfDim}
    }
    var cerr *C.char
    C.tq_add(i.h, (*C.float)(unsafe.Pointer(&vectors[0])), C.size_t(len(vectors)/dim), C.size_t(dim), &cerr)
    if err := takeErr(cerr); err != nil {
        return &Error{Op: "Add", Err: err, Msg: err.Error()}
    }
    return nil
}
```

We need an `AddWithDim` for the lazy case. Add it now:

```go
// AddWithDim inserts n = len(vectors)/dim vectors and commits the
// dim on a lazy index. On an eager index, dim must match.
func (i *TurboQuantIndex) AddWithDim(vectors []float32, dim int) error {
    if i.h == nil {
        return &Error{Op: "AddWithDim", Msg: "index is closed"}
    }
    if len(vectors)%dim != 0 {
        return &Error{Op: "AddWithDim", Err: ErrBufferNotMultipleOfDim}
    }
    var cerr *C.char
    C.tq_add(i.h, (*C.float)(unsafe.Pointer(&vectors[0])), C.size_t(len(vectors)/dim), C.size_t(dim), &cerr)
    if err := takeErr(cerr); err != nil {
        return &Error{Op: "AddWithDim", Err: err, Msg: err.Error()}
    }
    return nil
}
```

Update the lazy test to use `AddWithDim` (it already passes `dim=8` implicitly by using 16 floats; the test's `Add` call needs to be `AddWithDim(vecs, 8)`):

In `turboquant_index_test.go`, change:

```go
if err := idx.Add(vecs); err != nil {
```

to:

```go
if err := idx.AddWithDim(vecs, 8); err != nil {
```

inside `TestTurboQuantIndex_Add_LazyCommitsDim`. The other test functions (`TestTurboQuantIndex_Add_Eager`, `TestTurboQuantIndex_Add_DimMismatch`, `TestTurboQuantIndex_Add_BufferNotMultipleOfDim`, `TestTurboQuantIndex_Add_InvalidInputValue`) keep using `Add`.

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add turbovec-ffi/src/lib.rs turbovec/turboquant_index.go turbovec/turboquant_index_test.go
git commit -m "feat(tq): expose Add and AddWithDim"
```

---

## Task 5: FFI for `TurboQuantIndex` — `Search` and `SearchWithMask`

**Files:**
- Modify: `turbovec-ffi/src/lib.rs` (add `tq_search` and `tq_results_free`)
- Modify: `turbovec/turboquant_index.go` (add `Search`, `SearchWithMask`)
- Create: `turbovec/search_results.go` (add `Results` and row accessors)
- Modify: `turbovec/turboquant_index_test.go` (add search tests)

- [ ] **Step 1: Write the failing Go tests**

Append to `turbovec/turboquant_index_test.go`:

```go
func TestTurboQuantIndex_Search_Basic(t *testing.T) {
    idx, _ := NewTurboQuantIndex(4, 4)
    defer idx.Close()
    // 4 unit vectors along the 4 axes.
    vecs := []float32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }
    if err := idx.Add(vecs); err != nil {
        t.Fatalf("Add: %v", err)
    }
    idx.Prepare()
    // Query close to v0.
    r, err := idx.Search([]float32{1, 0, 0, 0}, 2)
    if err != nil {
        t.Fatalf("Search: %v", err)
    }
    if r.Nq != 1 || r.K != 2 {
        t.Fatalf("Nq=%d K=%d, want 1 2", r.Nq, r.K)
    }
    if len(r.Scores) != 2 || len(r.Indices) != 2 {
        t.Fatalf("len(Scores)=%d len(Indices)=%d, want 2 2", len(r.Scores), len(r.Indices))
    }
    // The first result should be index 0 with the highest score.
    if r.Indices[0] != 0 {
        t.Errorf("Indices[0] = %d, want 0", r.Indices[0])
    }
}

func TestTurboQuantIndex_SearchWithMask(t *testing.T) {
    idx, _ := NewTurboQuantIndex(4, 4)
    defer idx.Close()
    vecs := []float32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }
    idx.Add(vecs)
    idx.Prepare()
    // Allow only index 2.
    mask := []bool{false, false, true, false}
    r, err := idx.SearchWithMask([]float32{1, 0, 0, 0}, 2, mask)
    if err != nil {
        t.Fatalf("SearchWithMask: %v", err)
    }
    if r.K != 1 {
        t.Errorf("K = %d, want 1", r.K)
    }
    if r.Indices[0] != 2 {
        t.Errorf("Indices[0] = %d, want 2", r.Indices[0])
    }
}

func TestTurboQuantIndex_Search_WrongDim(t *testing.T) {
    idx, _ := NewTurboQuantIndex(8, 4)
    defer idx.Close()
    _, err := idx.Search([]float32{1, 0, 0, 0}, 1)
    if !errorsIs(err, ErrInvalidDim) {
        t.Errorf("err = %v, want errors.Is(ErrInvalidDim)", err)
    }
}

func TestTurboQuantIndex_SearchWithMask_LengthMismatch(t *testing.T) {
    idx, _ := NewTurboQuantIndex(4, 4)
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 1, 0, 0}
    idx.Add(vecs)
    idx.Prepare()
    mask := []bool{true} // length 1, but Len()=2
    _, err := idx.SearchWithMask([]float32{1, 0, 0, 0}, 1, mask)
    if err == nil {
        t.Error("want error for mask length mismatch, got nil")
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run:

```bash
make test 2>&1 | head -40
```

Expected: compile errors `i.Search undefined`, `i.SearchWithMask undefined`, `i.Prepare undefined`, `Results undefined`, etc.

- [ ] **Step 3: Implement the FFI side — `tq_search` and `tq_results_free`**

Append to `turbovec-ffi/src/lib.rs`:

```rust
#[no_mangle]
pub extern "C" fn tq_search(
    h: *const TqHandle,
    queries: *const f32,
    nq: usize,
    dim: usize,
    k: usize,
    mask: *const u8, // Option<&[bool]>, 1 byte per bool
) -> TqSearchResults {
    let handle = unsafe { &*h };
    let q_slice = unsafe { std::slice::from_raw_parts(queries, nq * dim) };
    let mask_slice: Option<Vec<bool>> = if mask.is_null() {
        None
    } else {
        let raw = unsafe { std::slice::from_raw_parts(mask, handle.inner.len()) };
        Some(raw.iter().map(|&b| b != 0).collect())
    };
    let res = handle.inner.search_with_mask(q_slice, k, mask_slice.as_deref());
    let n = res.scores.len();
    let mut scores = res.scores.into_boxed_slice();
    let mut indices = res.indices.into_boxed_slice();
    let scores_ptr = scores.as_mut_ptr();
    let indices_ptr = indices.as_mut_ptr();
    std::mem::forget(scores);
    std::mem::forget(indices);
    TqSearchResults {
        scores: scores_ptr,
        indices: indices_ptr,
        nq: res.nq,
        k: res.k,
    }
}

#[no_mangle]
pub extern "C" fn tq_results_free(r: TqSearchResults) {
    if !r.scores.is_null() {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(r.scores, r.nq * r.k));
        }
    }
    if !r.indices.is_null() {
        unsafe {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(r.indices, r.nq * r.k));
        }
    }
}
```

- [ ] **Step 4: Add `Prepare` to the FFI (T5 prerequisite for the tests above)**

Append:

```rust
#[no_mangle]
pub extern "C" fn tq_prepare(h: *const TqHandle) {
    unsafe { (*h).inner.prepare() }
}
```

- [ ] **Step 5: Implement the Go side — `search_results.go`**

```go
package turbovec

// Results holds the output of a TurboQuantIndex search. Slices are
// row-major (nq, k); use ScoresForQuery / IndicesForQuery for per-row
// access.
type Results struct {
    Scores  []float32 // length Nq * K
    Indices []int64   // length Nq * K
    Nq, K   int
}

// ScoresForQuery returns the scores for query qi.
func (r Results) ScoresForQuery(qi int) []float32 {
    return r.Scores[qi*r.K : (qi+1)*r.K]
}

// IndicesForQuery returns the indices for query qi.
func (r Results) IndicesForQuery(qi int) []int64 {
    return r.Indices[qi*r.K : (qi+1)*r.K]
}

// IdResults holds the output of an IdMapIndex search. Same layout as
// Results, but indices are external uint64 ids.
type IdResults struct {
    Scores []float32
    IDs    []uint64
    Nq, K  int
}

// ScoresForQuery returns the scores for query qi.
func (r IdResults) ScoresForQuery(qi int) []float32 {
    return r.Scores[qi*r.K : (qi+1)*r.K]
}

// IDsForQuery returns the external ids for query qi.
func (r IdResults) IDsForQuery(qi int) []uint64 {
    return r.IDs[qi*r.K : (qi+1)*r.K]
}
```

- [ ] **Step 6: Implement `Search`, `SearchWithMask`, `Prepare` in Go**

Append to `turbovec/turboquant_index.go`:

```go
// Prepare eagerly materialises the lazy caches (rotation matrix, Lloyd-Max
// centroids, SIMD-blocked code layout) so the first Search does not pay
// the one-time initialisation cost.
func (i *TurboQuantIndex) Prepare() {
    C.tq_prepare(i.h)
}

// Search returns the top-k nearest neighbors of each query vector.
// queries is row-major (nq, dim). The returned Results has K=k for queries
// that are not on a lazy index, and min(k, n_vectors) when k > n.
func (i *TurboQuantIndex) Search(queries []float32, k int) (Results, error) {
    return i.SearchWithMask(queries, k, nil)
}

// SearchWithMask restricts the search to slots where mask[i] == true. The
// returned k is min(k, mask.sum()).
func (i *TurboQuantIndex) SearchWithMask(queries []float32, k int, mask []bool) (Results, error) {
    if i.h == nil {
        return Results{}, &Error{Op: "Search", Msg: "index is closed"}
    }
    dim, _ := i.Dim()
    if dim == 0 {
        return Results{}, &Error{Op: "Search", Err: ErrInvalidDim, Msg: "index dim is uncommitted"}
    }
    if len(queries)%dim != 0 {
        return Results{}, &Error{Op: "Search", Err: ErrInvalidDim}
    }
    nq := len(queries) / dim
    var cmask *C.uint8_t
    if len(mask) > 0 {
        if len(mask) != i.Len() {
            return Results{}, &Error{Op: "Search", Err: ErrBufferNotMultipleOfDim, Msg: "mask length does not match index size"}
        }
        var err error
        cmask, err = boolSliceToCBytes(mask)
        if err != nil {
            return Results{}, &Error{Op: "Search", Err: err, Msg: "calloc failed"}
        }
        defer C.free(unsafe.Pointer(cmask))
    }
    var qptr *C.float
    if len(queries) > 0 {
        qptr = (*C.float)(unsafe.Pointer(&queries[0]))
    }
    cr := C.tq_search(i.h, qptr, C.size_t(nq), C.size_t(dim), C.size_t(k), cmask)
    defer C.tq_results_free(cr)
    return copyTqResults(cr)
}

func copyTqResults(cr C.TqSearchResults) (Results, error) {
    n := int(cr.nq) * int(cr.k)
    if n == 0 {
        return Results{Nq: int(cr.nq), K: int(cr.k)}, nil
    }
    scores := unsafe.Slice((*float32)(unsafe.Pointer(cr.scores)), n)
    indices := unsafe.Slice((*int64)(unsafe.Pointer(cr.indices)), n)
    sCopy := make([]float32, n)
    copy(sCopy, scores)
    iCopy := make([]int64, n)
    copy(iCopy, indices)
    return Results{Scores: sCopy, Indices: iCopy, Nq: int(cr.nq), K: int(cr.k)}, nil
}
```

- [ ] **Step 7: Run tests and verify they pass**

Run:

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add turbovec-ffi/src/lib.rs turbovec/turboquant_index.go turbovec/search_results.go turbovec/turboquant_index_test.go
git commit -m "feat(tq): expose Search, SearchWithMask, Prepare"
```

---

## Task 6: FFI for `TurboQuantIndex` — `SwapRemove`, `Write`, `Load`

**Files:**
- Modify: `turbovec-ffi/src/lib.rs` (add `tq_swap_remove`, `tq_write`, `tq_load`)
- Modify: `turbovec/turboquant_index.go` (add `SwapRemove`, `Write`, `LoadTurboQuantIndex`)

- [ ] **Step 1: Write the failing Go tests**

Append to `turbovec/turboquant_index_test.go`:

```go
func TestTurboQuantIndex_SwapRemove(t *testing.T) {
    idx, _ := NewTurboQuantIndex(4, 4)
    defer idx.Close()
    vecs := []float32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    }
    idx.Add(vecs)
    if got := idx.Len(); got != 3 {
        t.Fatalf("Len()=%d, want 3", got)
    }
    movedFrom, err := idx.SwapRemove(0)
    if err != nil {
        t.Fatalf("SwapRemove: %v", err)
    }
    if got := idx.Len(); got != 2 {
        t.Errorf("Len()=%d, want 2", got)
    }
    if movedFrom != 2 {
        t.Errorf("movedFrom=%d, want 2", movedFrom)
    }
}

func TestTurboQuantIndex_SwapRemove_OutOfRange(t *testing.T) {
    idx, _ := NewTurboQuantIndex(4, 4)
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 1, 0, 0}
    idx.Add(vecs)
    _, err := idx.SwapRemove(5)
    if !errorsIs(err, ErrIndexOutOfRange) {
        t.Errorf("err = %v, want errors.Is(ErrIndexOutOfRange)", err)
    }
}

func TestTurboQuantIndex_WriteLoad(t *testing.T) {
    idx, _ := NewTurboQuantIndex(4, 4)
    defer idx.Close()
    vecs := []float32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }
    idx.Add(vecs)
    idx.Prepare()
    path := filepath.Join(tmpDir(t), "idx.tq")
    if err := idx.Write(path); err != nil {
        t.Fatalf("Write: %v", err)
    }
    loaded, err := LoadTurboQuantIndex(path)
    if err != nil {
        t.Fatalf("LoadTurboQuantIndex: %v", err)
    }
    defer loaded.Close()
    if got := loaded.Len(); got != 4 {
        t.Errorf("loaded Len()=%d, want 4", got)
    }
    r, err := loaded.Search([]float32{1, 0, 0, 0}, 1)
    if err != nil {
        t.Fatalf("loaded Search: %v", err)
    }
    if r.Indices[0] != 0 {
        t.Errorf("loaded Indices[0]=%d, want 0", r.Indices[0])
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run:

```bash
make test 2>&1 | head -40
```

Expected: compile errors `i.SwapRemove undefined`, `i.Write undefined`, `LoadTurboQuantIndex undefined`.

- [ ] **Step 3: Implement the FFI side**

Append to `turbovec-ffi/src/lib.rs`:

```rust
#[no_mangle]
pub extern "C" fn tq_swap_remove(h: *mut TqHandle, idx: usize, err: *mut *mut c_char) -> usize {
    let handle = unsafe { &mut *h };
    let len = handle.inner.len();
    if idx >= len {
        if !err.is_null() {
            unsafe {
                *err = prefixed_err(
                    error::PREFIX_INDEX_OUT_OF_RANGE,
                    format!("index {} out of range for length {}", idx, len),
                )
            }
        }
        return usize::MAX;
    }
    if !err.is_null() {
        unsafe { *err = std::ptr::null_mut() }
    }
    handle.inner.swap_remove(idx)
}

#[no_mangle]
pub extern "C" fn tq_write(h: *const TqHandle, path: *const c_char) -> i32 {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path) };
    match unsafe { (*h).inner.write(path_str) } {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn tq_load(path: *const c_char, err: *mut *mut c_char) -> *mut TqHandle {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path) };
    match CoreTq::load(path_str) {
        Ok(core) => {
            if !err.is_null() { unsafe { *err = std::ptr::null_mut() } }
            Box::into_raw(Box::new(TqHandle { inner: Box::new(core) }))
        }
        Err(e) => {
            if !err.is_null() {
                unsafe { *err = prefixed_err(error::PREFIX_IO, e) }
            }
            std::ptr::null_mut()
        }
    }
}
```

- [ ] **Step 4: Implement the Go side**

Append to `turbovec/turboquant_index.go`:

```go
// SwapRemove removes the vector at idx in O(1) by swapping with the last
// vector. The last vector moves into the deleted slot. Returns the
// previous index of the moved vector; equals idx when idx was already
// the last element.
func (i *TurboQuantIndex) SwapRemove(idx int) (int, error) {
    if i.h == nil {
        return 0, &Error{Op: "SwapRemove", Msg: "index is closed"}
    }
    var cerr *C.char
    moved := C.tq_swap_remove(i.h, C.size_t(idx), &cerr)
    if err := takeErr(cerr); err != nil {
        return 0, &Error{Op: "SwapRemove", Err: err, Msg: err.Error()}
    }
    if moved == C.size_t(^uint(0)>>1) { // SIZE_MAX
        return 0, &Error{Op: "SwapRemove", Err: ErrIndexOutOfRange, Msg: "upstream returned SIZE_MAX"}
    }
    return int(moved), nil
}

// Write serializes the index to a .tq file.
func (i *TurboQuantIndex) Write(path string) error {
    if i.h == nil {
        return &Error{Op: "Write", Msg: "index is closed"}
    }
    cpath := C.CString(path)
    defer C.free(unsafe.Pointer(cpath))
    if rc := C.tq_write(i.h, cpath); rc != 0 {
        return &Error{Op: "Write", Err: ErrIO, Msg: path}
    }
    return nil
}

// LoadTurboQuantIndex loads an index previously written by Write.
func LoadTurboQuantIndex(path string) (*TurboQuantIndex, error) {
    cpath := C.CString(path)
    defer C.free(unsafe.Pointer(cpath))
    var cerr *C.char
    h := C.tq_load(cpath, &cerr)
    if err := takeErr(cerr); err != nil {
        return nil, &Error{Op: "LoadTurboQuantIndex", Err: err, Msg: path}
    }
    if h == nil {
        return nil, &Error{Op: "LoadTurboQuantIndex", Msg: "tq_load returned null without error"}
    }
    return newTurboQuantIndex(h), nil
}
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add turbovec-ffi/src/lib.rs turbovec/turboquant_index.go turbovec/turboquant_index_test.go
git commit -m "feat(tq): expose SwapRemove, Write, LoadTurboQuantIndex"
```

---

## Task 7: Wire up the build cleanly (Makefile, gen.go, BLAS LDFLAGS)

This is the task that fixes the cgo LDFLAGS to include BLAS. Without this, the binary will fail to link on first build with undefined references to `cblas_sgemm`.

**Files:**
- Modify: `Makefile` (finalize; add `clean` target)
- Create: `scripts/check-tools.sh` (verify `cargo` and `go` are present)
- Modify: `turbovec/cgo.go` (add `// #cgo` LDFLAGS directives)

- [ ] **Step 1: Finalize the Makefile**

Replace the Makefile with:

```make
.PHONY: all build test clean generate

# Detect host OS for cgo LDFLAGS.
UNAME_S := $(shell uname -s)

all: build

generate:
	cd turbovec-ffi && cargo build --release

build: generate
	go build ./...

test: generate
	go test ./...

clean:
	rm -rf turbovec-ffi/target
	go clean -testcache
```

- [ ] **Step 2: Add a `// #cgo` LDFLAGS directive to `turbovec/cgo.go`**

The upstream's `build.rs` links OpenBLAS on Linux and Accelerate on macOS. The staticlib from the shim embeds the upstream's link directive into itself, so the *final* Go binary also needs those symbols. Add to the top of `turbovec/cgo.go`, inside the file-level comment block above the `#include`s:

```go
// #cgo CFLAGS: -I${SRCDIR}/../turbovec-ffi/include
// #cgo linux LDFLAGS: -L${SRCDIR}/../turbovec-ffi/target/release -lturbovec_ffi -lopenblas -lpthread -ldl -lm
// #cgo darwin LDFLAGS: -L${SRCDIR}/../turbovec-ffi/target/release -lturbovec_ffi -framework Accelerate
import "C"
```

(Move the `import "C"` line to the bottom of the cgo block, as Go requires. The LDFLAGS lines must come before the import line in the file body — above the doc comment that includes `#include` lines.)

Final shape of the top of `turbovec/cgo.go`:

```go
package turbovec

/*
#cgo CFLAGS: -I${SRCDIR}/../turbovec-ffi/include
#cgo linux LDFLAGS: -L${SRCDIR}/../turbovec-ffi/target/release -lturbovec_ffi -lopenblas -lpthread -ldl -lm
#cgo darwin LDFLAGS: -L${SRCDIR}/../turbovec-ffi/target/release -lturbovec_ffi -framework Accelerate

#include <stdlib.h>
#include "cabi.h"
*/
import "C"
```

- [ ] **Step 3: Verify a clean build works end-to-end**

Run:

```bash
make clean && make test
```

Expected: builds the Rust shim, links the Go binary, runs all tests. On Linux, expect a long initial build as Cargo pulls `turbovec`'s dependencies (faer, ndarray, etc.). On macOS, similar but with Accelerate already on the system.

If you see `undefined reference to cblas_sgemm`, the LDFLAGS are wrong — re-check the LDFLAGS line for your OS.

- [ ] **Step 4: Commit**

```bash
git add Makefile turbovec/cgo.go
git commit -m "build: finalize Makefile and add BLAS cgo LDFLAGS"
```

---

## Task 8: FFI for `IdMapIndex` — constructors, accessors, drop

**Files:**
- Modify: `turbovec-ffi/src/lib.rs` (add `IdHandle` and `id_*` constructors/accessors/drop)
- Create: `turbovec/id_map_index.go` (struct + constructors + accessors + Close)
- Create: `turbovec/id_map_index_test.go`

- [ ] **Step 1: Write the failing Go tests**

Create `turbovec/id_map_index_test.go`:

```go
package turbovec

import (
    "testing"
)

func TestNewIdMapIndex_Eager(t *testing.T) {
    idx, err := NewIdMapIndex(128, 4)
    if err != nil {
        t.Fatalf("NewIdMapIndex: %v", err)
    }
    defer idx.Close()
    if got, _ := idx.Dim(); got != 128 {
        t.Errorf("Dim() = %d, want 128", got)
    }
    if got := idx.BitWidth(); got != 4 {
        t.Errorf("BitWidth() = %d, want 4", got)
    }
    if got := idx.Len(); got != 0 {
        t.Errorf("Len() = %d, want 0", got)
    }
}

func TestNewIdMapIndex_InvalidBitWidth(t *testing.T) {
    _, err := NewIdMapIndex(128, 7)
    if !errorsIs(err, ErrInvalidBitWidth) {
        t.Errorf("err = %v, want errors.Is(ErrInvalidBitWidth)", err)
    }
}

func TestNewLazyIdMapIndex(t *testing.T) {
    idx, err := NewLazyIdMapIndex(4)
    if err != nil {
        t.Fatalf("NewLazyIdMapIndex: %v", err)
    }
    defer idx.Close()
    if _, ok := idx.Dim(); ok {
        t.Error("Dim() committed before add")
    }
}

func TestIdMapIndex_Close_Idempotent(t *testing.T) {
    idx, _ := NewIdMapIndex(64, 4)
    if err := idx.Close(); err != nil {
        t.Fatalf("first Close: %v", err)
    }
    if err := idx.Close(); err != nil {
        t.Errorf("second Close: %v, want nil", err)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run:

```bash
make test 2>&1 | head -40
```

Expected: compile errors `NewIdMapIndex undefined`, `NewLazyIdMapIndex undefined`.

- [ ] **Step 3: Implement the FFI side**

Append to `turbovec-ffi/src/lib.rs`:

```rust
use turbovec::IdMapIndex as CoreIdMap;

#[repr(C)]
pub struct IdHandle {
    inner: Box<CoreIdMap>,
}

#[no_mangle]
pub extern "C" fn id_new(dim: usize, bit_width: u32, err: *mut *mut c_char) -> *mut IdHandle {
    match CoreIdMap::new(dim, bit_width as usize) {
        Ok(core) => {
            if !err.is_null() { unsafe { *err = std::ptr::null_mut() } }
            Box::into_raw(Box::new(IdHandle { inner: Box::new(core) }))
        }
        Err(e) => {
            if !err.is_null() {
                let prefix = match &e {
                    turbovec::ConstructError::BitWidthOutOfRange(_) => error::PREFIX_BIT_WIDTH,
                    turbovec::ConstructError::DimNotPositiveMultipleOf8(_) => error::PREFIX_DIM_NOT_MULTIPLE_OF_8,
                };
                unsafe { *err = prefixed_err(prefix, e) }
            }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn id_new_lazy(bit_width: u32, err: *mut *mut c_char) -> *mut IdHandle {
    match CoreIdMap::new_lazy(bit_width as usize) {
        Ok(core) => {
            if !err.is_null() { unsafe { *err = std::ptr::null_mut() } }
            Box::into_raw(Box::new(IdHandle { inner: Box::new(core) }))
        }
        Err(e) => {
            if !err.is_null() { unsafe { *err = prefixed_err(error::PREFIX_BIT_WIDTH, e) } }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn id_drop(h: *mut IdHandle) {
    if !h.is_null() {
        unsafe { drop(Box::from_raw(h)) }
    }
}

#[no_mangle]
pub extern "C" fn id_len(h: *const IdHandle) -> usize {
    unsafe { (*h).inner.len() }
}

#[no_mangle]
pub extern "C" fn id_dim(h: *const IdHandle) -> usize {
    unsafe { (*h).inner.dim() }
}

#[no_mangle]
pub extern "C" fn id_bit_width(h: *const IdHandle) -> u32 {
    unsafe { (*h).inner.bit_width() as u32 }
}
```

- [ ] **Step 4: Implement the Go side**

Create `turbovec/id_map_index.go`:

```go
package turbovec

/*
#include <stdlib.h>
#include "cabi.h"
*/
import "C"

import (
    "runtime"
)

// IdMapIndex is a top-k index keyed by external uint64 ids. Adds, removes,
// and allowlist-restricted searches all key on the external id.
type IdMapIndex struct {
    h *C.IdHandle
}

// NewIdMapIndex constructs an id-mapped index. bit_width must be 2, 3, or 4.
// dim must be a positive multiple of 8.
func NewIdMapIndex(dim, bitWidth int) (*IdMapIndex, error) {
    var cerr *C.char
    h := C.id_new(C.size_t(dim), C.uint32_t(bitWidth), &cerr)
    if err := takeErr(cerr); err != nil {
        return nil, &Error{Op: "NewIdMapIndex", Err: err, Msg: err.Error()}
    }
    if h == nil {
        return nil, &Error{Op: "NewIdMapIndex", Msg: "id_new returned null without error"}
    }
    return newIdMapIndex(h), nil
}

// NewLazyIdMapIndex constructs an id-mapped index whose dim is committed
// on the first AddWithIDs / AddWithIDsWithDim.
func NewLazyIdMapIndex(bitWidth int) (*IdMapIndex, error) {
    var cerr *C.char
    h := C.id_new_lazy(C.uint32_t(bitWidth), &cerr)
    if err := takeErr(cerr); err != nil {
        return nil, &Error{Op: "NewLazyIdMapIndex", Err: err, Msg: err.Error()}
    }
    if h == nil {
        return nil, &Error{Op: "NewLazyIdMapIndex", Msg: "id_new_lazy returned null without error"}
    }
    return newIdMapIndex(h), nil
}

func newIdMapIndex(h *C.IdHandle) *IdMapIndex {
    idx := &IdMapIndex{h: h}
    runtime.SetFinalizer(idx, func(i *IdMapIndex) {
        if i.h != nil {
            C.id_drop(i.h)
            i.h = nil
        }
    })
    return idx
}

// Len returns the number of vectors currently in the index.
func (i *IdMapIndex) Len() int { return int(C.id_len(i.h)) }

// Dim returns the index's vector dimensionality. The second return value
// is false when the index is lazy and has not yet seen an AddWithIDs.
func (i *IdMapIndex) Dim() (int, bool) {
    d := C.id_dim(i.h)
    if d == 0 {
        return 0, false
    }
    return int(d), true
}

// BitWidth returns the bit width used for quantization (2, 3, or 4).
func (i *IdMapIndex) BitWidth() int { return int(C.id_bit_width(i.h)) }

// Close releases the C-side handle. Idempotent.
func (i *IdMapIndex) Close() error {
    if i.h == nil {
        return nil
    }
    C.id_drop(i.h)
    i.h = nil
    runtime.SetFinalizer(i, nil)
    return nil
}
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add turbovec-ffi/src/lib.rs turbovec/id_map_index.go turbovec/id_map_index_test.go
git commit -m "feat(id): expose IdMapIndex constructor, accessors, Close"
```

---

## Task 9: FFI for `IdMapIndex` — `AddWithIDs`, `Remove`, `Contains`

**Files:**
- Modify: `turbovec-ffi/src/lib.rs` (add `id_add_with_ids`, `id_remove`, `id_contains`)
- Modify: `turbovec/id_map_index.go` (add the methods)
- Modify: `turbovec/id_map_index_test.go` (add tests)

- [ ] **Step 1: Write the failing Go tests**

Append to `turbovec/id_map_index_test.go`:

```go
func TestIdMapIndex_AddWithIDs(t *testing.T) {
    idx, _ := NewIdMapIndex(8, 4)
    defer idx.Close()
    vecs := []float32{
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
    }
    ids := []uint64{100, 200}
    if err := idx.AddWithIDs(vecs, ids); err != nil {
        t.Fatalf("AddWithIDs: %v", err)
    }
    if got := idx.Len(); got != 2 {
        t.Errorf("Len()=%d, want 2", got)
    }
    if !idx.Contains(100) || !idx.Contains(200) {
        t.Error("Contains(100/200) = false, want true")
    }
    if idx.Contains(999) {
        t.Error("Contains(999) = true, want false")
    }
}

func TestIdMapIndex_AddWithIDsWithDim(t *testing.T) {
    idx, _ := NewLazyIdMapIndex(4)
    defer idx.Close()
    vecs := make([]float32, 16)
    ids := []uint64{1, 2}
    if err := idx.AddWithIDsWithDim(vecs, ids, 8); err != nil {
        t.Fatalf("AddWithIDsWithDim: %v", err)
    }
    if d, ok := idx.Dim(); !ok || d != 8 {
        t.Errorf("Dim() = (%d, %v), want (8, true)", d, ok)
    }
}

func TestIdMapIndex_AddWithIDs_DuplicateID(t *testing.T) {
    idx, _ := NewIdMapIndex(4, 4)
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 1, 0, 0}
    ids := []uint64{1, 1}
    if err := idx.AddWithIDs(vecs, ids); !errorsIs(err, ErrDuplicateID) {
        t.Errorf("err = %v, want errors.Is(ErrDuplicateID)", err)
    }
}

func TestIdMapIndex_AddWithIDs_CountMismatch(t *testing.T) {
    idx, _ := NewIdMapIndex(4, 4)
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 1, 0, 0}
    ids := []uint64{1} // 1 id, 2 vectors
    if err := idx.AddWithIDs(vecs, ids); !errorsIs(err, ErrIDsCountMismatch) {
        t.Errorf("err = %v, want errors.Is(ErrIDsCountMismatch)", err)
    }
}

func TestIdMapIndex_Remove(t *testing.T) {
    idx, _ := NewIdMapIndex(4, 4)
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 1, 0, 0}
    ids := []uint64{1, 2}
    idx.AddWithIDs(vecs, ids)
    if !idx.Remove(1) {
        t.Error("Remove(1) = false, want true")
    }
    if idx.Remove(1) {
        t.Error("Remove(1) again = true, want false")
    }
    if idx.Contains(1) {
        t.Error("Contains(1) after Remove = true, want false")
    }
    if got := idx.Len(); got != 1 {
        t.Errorf("Len()=%d, want 1", got)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail to compile**

Run:

```bash
make test 2>&1 | head -40
```

Expected: compile errors `AddWithIDs undefined`, `AddWithIDsWithDim undefined`, `Contains undefined`, `Remove undefined`.

- [ ] **Step 3: Implement the FFI side**

Append to `turbovec-ffi/src/lib.rs`:

```rust
#[no_mangle]
pub extern "C" fn id_add_with_ids(
    h: *mut IdHandle,
    vectors: *const f32,
    n: usize,
    dim: usize,
    ids: *const u64,
    err: *mut *mut c_char,
) {
    if h.is_null() || vectors.is_null() || ids.is_null() {
        if !err.is_null() {
            unsafe { *err = prefixed_err(error::PREFIX_INVALID_DIM, "null handle or buffer") }
        }
        return;
    }
    let v_slice = unsafe { std::slice::from_raw_parts(vectors, n * dim) };
    let i_slice = unsafe { std::slice::from_raw_parts(ids, n) };
    let res = unsafe { (*h).inner.add_with_ids_2d(v_slice, dim, i_slice) };
    if !err.is_null() {
        match res {
            Ok(()) => unsafe { *err = std::ptr::null_mut() },
            Err(e) => {
                let prefix = match &e {
                    turbovec::AddError::DimMismatch { .. } => error::PREFIX_INVALID_DIM,
                    turbovec::AddError::DimNotMultipleOf8(_) => error::PREFIX_DIM_NOT_MULTIPLE_OF_8,
                    turbovec::AddError::VectorBufferNotMultipleOfDim { .. } => error::PREFIX_BUFFER_NOT_MULTIPLE_OF_DIM,
                    turbovec::AddError::IdsCountMismatch { .. } => error::PREFIX_IDS_COUNT_MISMATCH,
                    turbovec::AddError::IdAlreadyPresent(_) => error::PREFIX_DUPLICATE_ID,
                    turbovec::AddError::InvalidInputValue { .. } => error::PREFIX_INVALID_INPUT_VALUE,
                };
                unsafe { *err = prefixed_err(prefix, e) }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn id_remove(h: *mut IdHandle, id: u64) -> bool {
    unsafe { (*h).inner.remove(id) }
}

#[no_mangle]
pub extern "C" fn id_contains(h: *const IdHandle, id: u64) -> bool {
    unsafe { (*h).inner.contains(id) }
}
```

- [ ] **Step 4: Implement the Go side**

Append to `turbovec/id_map_index.go`:

```go
// AddWithIDs inserts n = len(vectors)/dim vectors keyed by ids. On a lazy
// index, the dim is taken from the eager index or rejected; use
// AddWithIDsWithDim for lazy.
func (i *IdMapIndex) AddWithIDs(vectors []float32, ids []uint64) error {
    if i.h == nil {
        return &Error{Op: "AddWithIDs", Msg: "index is closed"}
    }
    dim, _ := i.Dim()
    if dim == 0 {
        return &Error{Op: "AddWithIDs", Msg: "dim not committed; use AddWithIDsWithDim on a lazy index"}
    }
    return i.addWithIDsInner(vectors, ids, dim)
}

// AddWithIDsWithDim inserts n vectors with the given dim, committing it
// on a lazy index.
func (i *IdMapIndex) AddWithIDsWithDim(vectors []float32, ids []uint64, dim int) error {
    if i.h == nil {
        return &Error{Op: "AddWithIDsWithDim", Msg: "index is closed"}
    }
    return i.addWithIDsInner(vectors, ids, dim)
}

func (i *IdMapIndex) addWithIDsInner(vectors []float32, ids []uint64, dim int) error {
    if len(vectors)%dim != 0 {
        return &Error{Op: "AddWithIDs", Err: ErrBufferNotMultipleOfDim}
    }
    n := len(vectors) / dim
    if n != len(ids) {
        return &Error{Op: "AddWithIDs", Err: ErrIDsCountMismatch}
    }
    if n == 0 {
        return nil
    }
    var cerr *C.char
    C.id_add_with_ids(
        i.h,
        (*C.float)(unsafe.Pointer(&vectors[0])),
        C.size_t(n),
        C.size_t(dim),
        (*C.uint64_t)(unsafe.Pointer(&ids[0])),
        &cerr,
    )
    if err := takeErr(cerr); err != nil {
        return &Error{Op: "AddWithIDs", Err: err, Msg: err.Error()}
    }
    return nil
}

// Remove deletes the vector with the given external id. Returns true if
// it was present, false otherwise.
func (i *IdMapIndex) Remove(id uint64) bool {
    if i.h == nil {
        return false
    }
    return bool(C.id_remove(i.h, C.uint64_t(id)))
}

// Contains reports whether the given id is present in the index.
func (i *IdMapIndex) Contains(id uint64) bool {
    if i.h == nil {
        return false
    }
    return bool(C.id_contains(i.h, C.uint64_t(id)))
}
```

(`unsafe` needs to be imported in `id_map_index.go`. Add `"unsafe"` to the import block.)

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add turbovec-ffi/src/lib.rs turbovec/id_map_index.go turbovec/id_map_index_test.go
git commit -m "feat(id): expose AddWithIDs, AddWithIDsWithDim, Remove, Contains"
```

---

## Task 10: FFI for `IdMapIndex` — `Search`, `SearchWithAllowlist`, persistence, prepare

**Files:**
- Modify: `turbovec-ffi/src/lib.rs` (add `id_search`, `id_results_free`, `id_prepare`, `id_write`, `id_load`)
- Modify: `turbovec/id_map_index.go` (add the methods)
- Modify: `turbovec/id_map_index_test.go` (add tests)

- [ ] **Step 1: Write the failing Go tests**

Append to `turbovec/id_map_index_test.go`:

```go
func TestIdMapIndex_Search_Basic(t *testing.T) {
    idx, _ := NewIdMapIndex(4, 4)
    defer idx.Close()
    vecs := []float32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }
    ids := []uint64{10, 20, 30, 40}
    idx.AddWithIDs(vecs, ids)
    idx.Prepare()
    r, err := idx.Search([]float32{1, 0, 0, 0}, 2)
    if err != nil {
        t.Fatalf("Search: %v", err)
    }
    if r.Nq != 1 || r.K != 2 {
        t.Fatalf("Nq=%d K=%d, want 1 2", r.Nq, r.K)
    }
    if r.IDs[0] != 10 {
        t.Errorf("IDs[0] = %d, want 10", r.IDs[0])
    }
}

func TestIdMapIndex_SearchWithAllowlist(t *testing.T) {
    idx, _ := NewIdMapIndex(4, 4)
    defer idx.Close()
    vecs := []float32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }
    ids := []uint64{10, 20, 30, 40}
    idx.AddWithIDs(vecs, ids)
    idx.Prepare()
    r, err := idx.SearchWithAllowlist([]float32{1, 0, 0, 0}, 2, []uint64{30, 40})
    if err != nil {
        t.Fatalf("SearchWithAllowlist: %v", err)
    }
    if r.K != 2 {
        t.Errorf("K = %d, want 2", r.K)
    }
    // Both ids should be in the result, order is not guaranteed.
    found30 := false
    found40 := false
    for _, id := range r.IDs {
        if id == 30 {
            found30 = true
        }
        if id == 40 {
            found40 = true
        }
    }
    if !found30 || !found40 {
        t.Errorf("IDs = %v, want 30 and 40", r.IDs)
    }
}

func TestIdMapIndex_SearchWithAllowlist_EmptyAllowlist(t *testing.T) {
    idx, _ := NewIdMapIndex(4, 4)
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 1, 0, 0}
    idx.AddWithIDs(vecs, []uint64{1, 2})
    idx.Prepare()
    _, err := idx.SearchWithAllowlist([]float32{1, 0, 0, 0}, 2, []uint64{})
    if !errorsIs(err, ErrEmptyAllowlist) {
        t.Errorf("err = %v, want errors.Is(ErrEmptyAllowlist)", err)
    }
}

func TestIdMapIndex_SearchWithAllowlist_UnknownID(t *testing.T) {
    idx, _ := NewIdMapIndex(4, 4)
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 1, 0, 0}
    idx.AddWithIDs(vecs, []uint64{1, 2})
    idx.Prepare()
    _, err := idx.SearchWithAllowlist([]float32{1, 0, 0, 0}, 2, []uint64{1, 9999})
    if !errorsIs(err, ErrUnknownID) {
        t.Errorf("err = %v, want errors.Is(ErrUnknownID)", err)
    }
}

func TestIdMapIndex_WriteLoad(t *testing.T) {
    idx, _ := NewIdMapIndex(4, 4)
    defer idx.Close()
    vecs := []float32{1, 0, 0, 0, 0, 1, 0, 0}
    idx.AddWithIDs(vecs, []uint64{1, 2})
    idx.Prepare()
    path := filepath.Join(tmpDir(t), "idx.tvim")
    if err := idx.Write(path); err != nil {
        t.Fatalf("Write: %v", err)
    }
    loaded, err := LoadIdMapIndex(path)
    if err != nil {
        t.Fatalf("LoadIdMapIndex: %v", err)
    }
    defer loaded.Close()
    if got := loaded.Len(); got != 2 {
        t.Errorf("loaded Len() = %d, want 2", got)
    }
    if !loaded.Contains(1) || !loaded.Contains(2) {
        t.Error("loaded missing ids")
    }
}
```

Add `"path/filepath"` to the imports of `id_map_index_test.go`.

- [ ] **Step 2: Run tests to verify they fail to compile**

Run:

```bash
make test 2>&1 | head -40
```

Expected: compile errors `Search undefined`, `SearchWithAllowlist undefined`, `Prepare undefined`, `Write undefined`, `LoadIdMapIndex undefined`.

- [ ] **Step 3: Implement the FFI side**

Append to `turbovec-ffi/src/lib.rs`:

```rust
#[no_mangle]
pub extern "C" fn id_search(
    h: *const IdHandle,
    queries: *const f32,
    nq: usize,
    dim: usize,
    k: usize,
    allowlist: *const u64,
    allowlist_len: usize,
) -> IdSearchResults {
    let handle = unsafe { &*h };
    let q_slice = unsafe { std::slice::from_raw_parts(queries, nq * dim) };
    let allow_opt: Option<&[u64]> = if allowlist.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(allowlist, allowlist_len) })
    };
    let (scores, ids) = handle.inner.search_with_allowlist(q_slice, k, allow_opt);
    let n = scores.len();
    let mut scores_box = scores.into_boxed_slice();
    let mut ids_box = ids.into_boxed_slice();
    let sp = scores_box.as_mut_ptr();
    let ip = ids_box.as_mut_ptr();
    std::mem::forget(scores_box);
    std::mem::forget(ids_box);
    IdSearchResults {
        scores: sp,
        ids: ip,
        nq,
        k: n / nq.max(1),
    }
}

#[no_mangle]
pub extern "C" fn id_results_free(r: IdSearchResults) {
    let n = r.nq * r.k;
    if !r.scores.is_null() {
        unsafe { let _ = Box::from_raw(std::slice::from_raw_parts_mut(r.scores, n)); }
    }
    if !r.ids.is_null() {
        unsafe { let _ = Box::from_raw(std::slice::from_raw_parts_mut(r.ids, n)); }
    }
}

#[no_mangle]
pub extern "C" fn id_prepare(h: *const IdHandle) {
    unsafe { (*h).inner.prepare() }
}

#[no_mangle]
pub extern "C" fn id_write(h: *const IdHandle, path: *const c_char) -> i32 {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path) };
    match unsafe { (*h).inner.write(path_str) } {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn id_load(path: *const c_char, err: *mut *mut c_char) -> *mut IdHandle {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path) };
    match CoreIdMap::load(path_str) {
        Ok(core) => {
            if !err.is_null() { unsafe { *err = std::ptr::null_mut() } }
            Box::into_raw(Box::new(IdHandle { inner: Box::new(core) }))
        }
        Err(e) => {
            if !err.is_null() { unsafe { *err = prefixed_err(error::PREFIX_IO, e) } }
            std::ptr::null_mut()
        }
    }
}
```

- [ ] **Step 4: Implement the Go side**

Append to `turbovec/id_map_index.go`:

```go
// Prepare eagerly materialises the lazy caches.
func (i *IdMapIndex) Prepare() {
    C.id_prepare(i.h)
}

// Search returns the top-k nearest external ids for each query. The
// returned IdResults has K=k for queries on an index with at least k
// vectors.
func (i *IdMapIndex) Search(queries []float32, k int) (IdResults, error) {
    return i.SearchWithAllowlist(queries, k, nil)
}

// SearchWithAllowlist restricts the search to the given external ids.
// Returns ErrEmptyAllowlist for an empty allowlist, ErrUnknownID if any
// allowlist id is not present.
func (i *IdMapIndex) SearchWithAllowlist(queries []float32, k int, allowlist []uint64) (IdResults, error) {
    if i.h == nil {
        return IdResults{}, &Error{Op: "Search", Msg: "index is closed"}
    }
    dim, _ := i.Dim()
    if dim == 0 {
        return IdResults{}, &Error{Op: "Search", Err: ErrInvalidDim, Msg: "index dim is uncommitted"}
    }
    if len(queries)%dim != 0 {
        return IdResults{}, &Error{Op: "Search", Err: ErrInvalidDim}
    }
    nq := len(queries) / dim
    var allowptr *C.uint64_t
    var allowlen C.size_t
    if len(allowlist) > 0 {
        allowptr = (*C.uint64_t)(unsafe.Pointer(&allowlist[0]))
        allowlen = C.size_t(len(allowlist))
    }
    var qptr *C.float
    if len(queries) > 0 {
        qptr = (*C.float)(unsafe.Pointer(&queries[0]))
    }
    cr := C.id_search(i.h, qptr, C.size_t(nq), C.size_t(dim), C.size_t(k), allowptr, allowlen)
    defer C.id_results_free(cr)
    return copyIdResults(cr)
}

func copyIdResults(cr C.IdSearchResults) (IdResults, error) {
    n := int(cr.nq) * int(cr.k)
    if n == 0 {
        return IdResults{Nq: int(cr.nq), K: int(cr.k)}, nil
    }
    scores := unsafe.Slice((*float32)(unsafe.Pointer(cr.scores)), n)
    ids := unsafe.Slice((*uint64)(unsafe.Pointer(cr.ids)), n)
    sCopy := make([]float32, n)
    copy(sCopy, scores)
    iCopy := make([]uint64, n)
    copy(iCopy, ids)
    return IdResults{Scores: sCopy, IDs: iCopy, Nq: int(cr.nq), K: int(cr.k)}, nil
}

// Write serializes the index to a .tvim file.
func (i *IdMapIndex) Write(path string) error {
    if i.h == nil {
        return &Error{Op: "Write", Msg: "index is closed"}
    }
    cpath := C.CString(path)
    defer C.free(unsafe.Pointer(cpath))
    if rc := C.id_write(i.h, cpath); rc != 0 {
        return &Error{Op: "Write", Err: ErrIO, Msg: path}
    }
    return nil
}

// LoadIdMapIndex loads an index previously written by Write.
func LoadIdMapIndex(path string) (*IdMapIndex, error) {
    cpath := C.CString(path)
    defer C.free(unsafe.Pointer(cpath))
    var cerr *C.char
    h := C.id_load(cpath, &cerr)
    if err := takeErr(cerr); err != nil {
        return nil, &Error{Op: "LoadIdMapIndex", Err: err, Msg: path}
    }
    if h == nil {
        return nil, &Error{Op: "LoadIdMapIndex", Msg: "id_load returned null without error"}
    }
    return newIdMapIndex(h), nil
}
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
make test
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add turbovec-ffi/src/lib.rs turbovec/id_map_index.go turbovec/id_map_index_test.go
git commit -m "feat(id): expose Search, SearchWithAllowlist, Prepare, Write, Load"
```

---

## Task 11: Integration test — recall@10 against a Go brute-force baseline

**Files:**
- Create: `turbovec_test/integration_test.go`
- Create: `turbovec_test/brute.go` (small helper: brute-force top-k on []float32)

- [ ] **Step 1: Write the brute-force helper**

Create `turbovec_test/brute.go`:

```go
package turbovec_test

import (
    "container/heap"
)

// bruteTopK returns the top-k highest-scoring (score, index) pairs from
// corpus, scored against query. Both query and corpus rows have the
// given dim. Uses inner product as the score (matches turbovec's
// default inner-product search).
func bruteTopK(query []float32, corpus []float32, dim int, k int) []bruteHit {
    type scored struct {
        score float32
        idx   int
    }
    pq := make(scoredPQ, 0, k)
    n := len(corpus) / dim
    for i := 0; i < n; i++ {
        row := corpus[i*dim : (i+1)*dim]
        var s float32
        for j := 0; j < dim; j++ {
            s += query[j] * row[j]
        }
        if pq.Len() < k {
            heap.Push(&pq, scored{score: s, idx: i})
        } else if pq.Len() > 0 && s > pq[0].score {
            heap.Pop(&pq)
            heap.Push(&pq, scored{score: s, idx: i})
        }
    }
    out := make([]bruteHit, pq.Len())
    for j := pq.Len() - 1; j >= 0; j-- {
        out[j] = bruteHit{Score: pq[0].score, Index: pq[0].idx}
        heap.Pop(&pq)
    }
    return out
}

type bruteHit struct {
    Score float32
    Index int
}

type scoredPQ []bruteHit

func (p scoredPQ) Len() int            { return len(p) }
func (p scoredPQ) Less(i, j int) bool  { return p[i].Score < p[j].Score }
func (p scoredPQ) Swap(i, j int)       { p[i], p[j] = p[j], p[i] }
func (p *scoredPQ) Push(x interface{}) { *p = append(*p, x.(bruteHit)) }
func (p *scoredPQ) Pop() interface{} {
    old := *p
    n := len(old)
    x := old[n-1]
    *p = old[:n-1]
    return x
}
```

- [ ] **Step 2: Write the recall test**

Create `turbovec_test/integration_test.go`:

```go
package turbovec_test

import (
    "math/rand"
    "testing"

    "github.com/divy/go-turbovec/turbovec"
)

const (
    corpusN   = 5000
    queryN    = 200
    testDim   = 64
    testK     = 10
    testBitW  = 4
    recallThreshold = 0.90
)

func TestTurboQuantIndex_RecallAt10(t *testing.T) {
    rand.Seed(42)
    corpus := randomUnitVectors(corpusN, testDim, rand.New(rand.NewSource(1)))
    queries := randomUnitVectors(queryN, testDim, rand.New(rand.NewSource(2)))

    idx, err := turbovec.NewTurboQuantIndex(testDim, testBitW)
    if err != nil {
        t.Fatalf("NewTurboQuantIndex: %v", err)
    }
    defer idx.Close()
    if err := idx.Add(flatten(corpus)); err != nil {
        t.Fatalf("Add: %v", err)
    }
    idx.Prepare()

    flatCorpus := flatten(corpus)
    total := 0
    hits := 0
    for _, q := range queries {
        qflat := flatten([][]float32{q})
        r, err := idx.Search(qflat, testK)
        if err != nil {
            t.Fatalf("Search: %v", err)
        }
        truth := bruteTopK(qflat, flatCorpus, testDim, testK)
        truthSet := map[int]bool{}
        for _, h := range truth {
            truthSet[h.Index] = true
        }
        for _, idx64 := range r.Indices {
            if truthSet[int(idx64)] {
                hits++
            }
        }
        total += testK
    }
    recall := float64(hits) / float64(total)
    if recall < recallThreshold {
        t.Errorf("recall@%d = %.3f, want >= %.3f", testK, recall, recallThreshold)
    }
}

func TestIdMapIndex_RoundTrip(t *testing.T) {
    rand.Seed(43)
    corpus := randomUnitVectors(1000, 32, rand.New(rand.NewSource(1)))
    queries := randomUnitVectors(50, 32, rand.New(rand.NewSource(2)))

    idx, err := turbovec.NewIdMapIndex(32, 4)
    if err != nil {
        t.Fatalf("NewIdMapIndex: %v", err)
    }
    defer idx.Close()

    flat := flatten(corpus)
    ids := make([]uint64, len(corpus))
    for i := range ids {
        ids[i] = uint64(i + 1)
    }
    if err := idx.AddWithIDs(flat, ids); err != nil {
        t.Fatalf("AddWithIDs: %v", err)
    }
    idx.Prepare()

    for _, q := range queries {
        qflat := flatten([][]float32{q})
        r, err := idx.SearchWithAllowlist(qflat, 5, ids)
        if err != nil {
            t.Fatalf("SearchWithAllowlist: %v", err)
        }
        if r.K != 5 {
            t.Errorf("K = %d, want 5", r.K)
        }
    }
}

func randomUnitVectors(n, dim int, rng *rand.Rand) [][]float32 {
    out := make([][]float32, n)
    for i := range out {
        row := make([]float32, dim)
        var sumSq float64
        for j := range row {
            row[j] = rng.NormFloat64()
            sumSq += float64(row[j]) * float64(row[j])
        }
        inv := float32(1.0 / (sumSq + 1e-300))
        for j := range row {
            row[j] *= float32(inv)
        }
        // Use a re-cast to avoid the unused-variable lint on inv.
        _ = inv
        out[i] = row
    }
    return out
}

func flatten(rows [][]float32) []float32 {
    if len(rows) == 0 {
        return nil
    }
    n := len(rows)
    dim := len(rows[0])
    out := make([]float32, n*dim)
    for i, row := range rows {
        copy(out[i*dim:(i+1)*dim], row)
    }
    return out
}
```

- [ ] **Step 3: Run the integration test**

Run:

```bash
make test
```

Expected: integration tests pass. `TestTurboQuantIndex_RecallAt10` should produce recall ≥ 0.90 at bit_width=4 on 5K random unit vectors. If recall comes in lower (e.g., 0.85), this is a known property of 4-bit quantization on synthetic data; bump `corpusN` to 50K (matching the spec) or relax the threshold to 0.80 with a TODO. Do not silently lower the threshold — investigate first by checking the upstream's published recall numbers for similar corpus sizes.

- [ ] **Step 4: Commit**

```bash
git add turbovec_test/
git commit -m "test: add recall@10 integration test against Go brute-force baseline"
```

---

## Task 12: Documentation and CI

**Files:**
- Create: `README.md`
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the README**

```markdown
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
```

- [ ] **Step 2: Write the CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: test (${{ matrix.os }}, ${{ matrix.arch }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            arch: amd64
          - os: macos-latest
            arch: arm64
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
          cache: true
      - name: Install OpenBLAS (Linux)
        if: runner.os == 'Linux'
        run: sudo apt-get update && sudo apt-get install -y libopenblas-dev
      - name: Build
        run: make build
      - name: Test
        run: make test
```

- [ ] **Step 3: Verify CI locally (best effort)**

Run:

```bash
make clean && make test
```

Expected: full build + test cycle. CI will run on push; we cannot trigger it locally, but the workflow is the canonical definition of "works on ubuntu-amd64 and macos-arm64."

- [ ] **Step 4: Commit**

```bash
git add README.md .github/workflows/ci.yml
git commit -m "docs+ci: add README and GitHub Actions workflow"
```

---

## Self-review

**1. Spec coverage:**

| Spec section | Task(s) |
|---|---|
| Repository layout | T1 |
| C-ABI shim surface | T2 (header) + T3–T6, T8–T10 (functions) |
| Memory ownership (handle/buffer rules) | T3, T5, T8, T10 (every FFI function follows the convention) |
| Error policy (prefixes, sentinel mapping) | T2, T3, T5, T9 (each fallible function emits the right prefix) |
| Go API: `TurboQuantIndex` constructors/accessors | T3 |
| Go API: `TurboQuantIndex` `Add` | T4 |
| Go API: `TurboQuantIndex` `Search`, `SearchWithMask`, `Prepare` | T5 |
| Go API: `TurboQuantIndex` `SwapRemove`, `Write`, `Load` | T6 |
| Go API: `IdMapIndex` constructors/accessors | T8 |
| Go API: `IdMapIndex` `AddWithIDs`, `Remove`, `Contains` | T9 |
| Go API: `IdMapIndex` `Search`, `SearchWithAllowlist`, `Prepare`, `Write`, `Load` | T10 |
| `Results` / `IdResults` structs with row accessors | T5 |
| Sentinel errors and `*Error` | T3 |
| Build (Makefile, gen.go, cgo LDFLAGS) | T1 (skeleton), T7 (finalize + BLAS) |
| Unit tests (mirroring Python `tests/`) | T3, T4, T5, T6, T8, T9, T10 |
| Integration test (recall@10 + IdMap round-trip) | T11 |
| CI (GitHub Actions matrix) | T12 |
| README | T12 |

All spec sections are covered. Two minor deviations from the spec, both intentional:

- **Spec said `Add` for both eager and lazy; the plan splits it into `Add` (eager only) and `AddWithDim` (lazy or explicit dim).** The spec's `Add(vectors)` API is ambiguous on lazy indices (the dim is not committed until the first add). Splitting the methods keeps each signature total. This is a small ergonomic improvement; will note in the README and update the spec at the end.
- **Spec didn't call out BLAS LDFLAGS.** T7 captures this as the build "just works" requirement. Will add a sentence to the spec at the end.

**2. Placeholder scan:** No "TBD", "TODO" (in the plan-step sense), "fill in details", or "similar to Task N" patterns. Every test, every FFI function, every Go method has the full code shown.

**3. Type / signature consistency:**

- `C.TqSearchResults` is used identically in T5 (FFI) and T5 (Go copy). ✓
- `C.IdSearchResults` is used identically in T10. ✓
- `TqHandle` and `IdHandle` are `#[repr(C)]` opaque structs; the `cabi.h` declares them as `typedef struct TqHandle TqHandle;` matching. ✓
- `boolSliceToCBytes` defined in T3 (`cgo.go`) and used in T5 (`SearchWithMask`) — consistent.
- `takeErr` defined in T3 (`cgo.go`) and used in T3, T4, T5, T6, T8, T9, T10 — consistent.
- `prefixed_err` defined in T2 (`lib.rs`) and used in T3, T4, T5, T6, T8, T9, T10 — consistent. Returns `*mut c_char` (caller hands to `err_free`).
- `*Error` defined in T3 (`errors.go`) and used in every method — consistent.
- `newTurboQuantIndex` / `newIdMapIndex` factory functions in T3 / T8 install the finalizer; `Close` clears the finalizer. ✓

**4. One concern I want to flag for the executor:**

In T5 the `*C.TqSearchResults` is passed by value to `tq_results_free`. This is the standard C pattern for "consume a struct full of pointers" and the cgo bridge handles it fine, but it's a small gotcha — if the executor refactors the FFI to return a pointer to the struct, they need to update both the shim and the Go `copyTqResults`. The plan keeps the value-return path. If a task fails on this, the executor should preserve the value-return, not convert to a pointer.

---

## Plan complete

The plan has 12 tasks. Each task is self-contained and ends with a passing test suite + commit. Tasks 1–7 ship the `TurboQuantIndex` end-to-end. Tasks 8–10 ship the `IdMapIndex`. Task 11 adds the integration test. Task 12 closes out with docs and CI.
