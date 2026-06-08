// turbovec-ffi/src/lib.rs
//
// C-ABI shim around the upstream `turbovec` Rust core. Consumed by the
// go-turbovec Go package via cgo. See include/cabi.h for the surface.

#![allow(clippy::missing_safety_doc)]

use std::ffi::CString;
use std::os::raw::c_char;

use turbovec::TurboQuantIndex as CoreTq;

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

/// Opaque handle to a heap-allocated `TurboQuantIndex`.
#[repr(C)]
pub struct TqHandle {
    inner: Box<CoreTq>,
}

#[no_mangle]
pub extern "C" fn tq_new(
    dim: usize,
    bit_width: u32,
    err: *mut *mut c_char,
) -> *mut TqHandle {
    match CoreTq::new(dim, bit_width as usize) {
        Ok(core) => {
            if !err.is_null() {
                unsafe { *err = std::ptr::null_mut() };
            }
            Box::into_raw(Box::new(TqHandle {
                inner: Box::new(core),
            }))
        }
        Err(e) => {
            if !err.is_null() {
                let prefix = match &e {
                    turbovec::ConstructError::BitWidthOutOfRange(_) => {
                        error::PREFIX_BIT_WIDTH
                    }
                    turbovec::ConstructError::DimNotPositiveMultipleOf8(_) => {
                        error::PREFIX_DIM_NOT_MULTIPLE_OF_8
                    }
                };
                unsafe { *err = prefixed_err(prefix, e) };
            }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn tq_new_lazy(bit_width: u32, err: *mut *mut c_char) -> *mut TqHandle {
    match CoreTq::new_lazy(bit_width as usize) {
        Ok(core) => {
            if !err.is_null() {
                unsafe { *err = std::ptr::null_mut() };
            }
            Box::into_raw(Box::new(TqHandle {
                inner: Box::new(core),
            }))
        }
        Err(e) => {
            if !err.is_null() {
                unsafe { *err = prefixed_err(error::PREFIX_BIT_WIDTH, e) };
            }
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
