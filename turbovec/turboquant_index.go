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
