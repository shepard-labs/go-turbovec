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
		// Safety net: free the C handle. A finalizer may never run
		// before program exit; Close is still required.
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
		// avoid an ambiguous contract.
		return &Error{Op: "Add", Msg: "dim not committed; use AddWithDim on a lazy index"}
	}
	if len(vectors)%dim != 0 {
		return &Error{Op: "Add", Err: ErrBufferNotMultipleOfDim}
	}
	var cerr *C.char
	C.tq_add(
		i.h,
		(*C.float)(unsafe.Pointer(&vectors[0])),
		C.size_t(len(vectors)/dim),
		C.size_t(dim),
		&cerr,
	)
	if err := takeErr(cerr); err != nil {
		return &Error{Op: "Add", Err: err, Msg: err.Error()}
	}
	return nil
}

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
	C.tq_add(
		i.h,
		(*C.float)(unsafe.Pointer(&vectors[0])),
		C.size_t(len(vectors)/dim),
		C.size_t(dim),
		&cerr,
	)
	if err := takeErr(cerr); err != nil {
		return &Error{Op: "AddWithDim", Err: err, Msg: err.Error()}
	}
	return nil
}

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
	var cerr *C.char
	cr := C.tq_search(i.h, qptr, C.size_t(nq), C.size_t(dim), C.size_t(k), cmask, &cerr)
	defer C.tq_results_free(cr)
	if err := takeErr(cerr); err != nil {
		return Results{}, &Error{Op: "Search", Err: err, Msg: err.Error()}
	}
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
