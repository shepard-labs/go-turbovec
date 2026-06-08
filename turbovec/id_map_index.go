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
