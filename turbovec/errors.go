package turbovec

import (
	"errors"
	"fmt"
	"strings"
)

// Sentinels — match the upstream's error taxonomy. Wrap with *Error via
// errors.Is / errors.As.
var (
	ErrInvalidDim             = errors.New("turbovec: invalid dim")
	ErrInvalidBitWidth        = errors.New("turbovec: invalid bit width (must be 2, 3, or 4)")
	ErrBufferNotMultipleOfDim = errors.New("turbovec: vector buffer length is not a multiple of dim")
	ErrIDsCountMismatch       = errors.New("turbovec: ids count does not match vectors length")
	ErrDuplicateID            = errors.New("turbovec: id already present in index")
	ErrInvalidInputValue      = errors.New("turbovec: input contains non-finite or out-of-range value")
	ErrEmptyAllowlist         = errors.New("turbovec: allowlist is empty")
	ErrUnknownID              = errors.New("turbovec: allowlist contains id(s) not present in index")
	ErrIndexOutOfRange        = errors.New("turbovec: index out of range")
	ErrIO                     = errors.New("turbovec: I/O error")
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
	case strings.HasPrefix(raw, errorPrefixDimNotMultipleOf8):
		// Upstream's `DimNotPositiveMultipleOf8` is conceptually a
		// dim-shape error; surface it through ErrInvalidDim so callers
		// can use a single sentinel for "the dim you gave me is bad."
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
