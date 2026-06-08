// turbovec-ffi/src/lib.rs
//
// C-ABI shim around the upstream `turbovec` Rust core. Consumed by the
// go-turbovec Go package via cgo. See include/cabi.h for the surface.

#![allow(clippy::missing_safety_doc)]

use std::ffi::CString;
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

/// Build a prefixed C string for the err out-parameter. Returns a raw
/// pointer suitable for `*err_out`; the caller is responsible for handing
/// it back to `err_free`.
#[allow(dead_code)]
pub(crate) fn prefixed_err(prefix: &str, msg: impl std::fmt::Display) -> *mut c_char {
    let s = format!("{}: {}", prefix, msg);
    CString::new(s)
        .unwrap_or_else(|_| CString::new(prefix).unwrap())
        .into_raw()
}
