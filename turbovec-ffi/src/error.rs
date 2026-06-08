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
