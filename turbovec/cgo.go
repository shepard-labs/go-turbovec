package turbovec

/*
#cgo CFLAGS: -I${SRCDIR}/../turbovec-ffi/include
#cgo LDFLAGS: -L${SRCDIR}/../turbovec-ffi/target/release -lturbovec_ffi -lpthread -ldl -lm

#include <stdlib.h>
#include "cabi.h"
*/
import "C"

import (
	"unsafe"
)

// Error-prefix constants — must match turbovec-ffi/src/error.rs.
const (
	errorPrefixInvalidDim             = "invalid_dim"
	errorPrefixBitWidth               = "bit_width"
	errorPrefixDimNotMultipleOf8      = "dim_not_multiple_of_8"
	errorPrefixBufferNotMultipleOfDim = "buffer_not_multiple_of_dim"
	errorPrefixIDsCountMismatch       = "ids_count_mismatch"
	errorPrefixDuplicateID            = "duplicate_id"
	errorPrefixInvalidInputValue      = "invalid_input_value"
	errorPrefixEmptyAllowlist         = "empty_allowlist"
	errorPrefixUnknownID              = "unknown_id"
	errorPrefixIndexOutOfRange        = "index_out_of_range"
	errorPrefixIO                     = "io"
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
