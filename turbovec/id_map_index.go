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
