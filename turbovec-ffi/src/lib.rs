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

/// Search result struct. Mirrors cabi.h's TqSearchResults.
#[repr(C)]
pub struct TqSearchResults {
    pub scores: *mut f32,
    pub indices: *mut i64,
    pub nq: usize,
    pub k: usize,
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
            unsafe {
                *err = prefixed_err(error::PREFIX_INVALID_DIM, "null handle or buffer")
            }
        }
        return;
    }
    if n == 0 {
        // Nothing to add; match the upstream's empty-add behavior.
        if !err.is_null() {
            unsafe { *err = std::ptr::null_mut() }
        }
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
                    turbovec::AddError::DimNotMultipleOf8(_) => {
                        error::PREFIX_DIM_NOT_MULTIPLE_OF_8
                    }
                    turbovec::AddError::VectorBufferNotMultipleOfDim { .. } => {
                        error::PREFIX_BUFFER_NOT_MULTIPLE_OF_DIM
                    }
                    turbovec::AddError::IdsCountMismatch { .. } => {
                        error::PREFIX_IDS_COUNT_MISMATCH
                    }
                    turbovec::AddError::IdAlreadyPresent(_) => error::PREFIX_DUPLICATE_ID,
                    turbovec::AddError::InvalidInputValue { .. } => {
                        error::PREFIX_INVALID_INPUT_VALUE
                    }
                };
                unsafe { *err = prefixed_err(prefix, e) }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn tq_prepare(h: *const TqHandle) {
    unsafe { (*h).inner.prepare() }
}

#[no_mangle]
pub extern "C" fn tq_search(
    h: *const TqHandle,
    queries: *const f32,
    nq: usize,
    dim: usize,
    k: usize,
    mask: *const u8, // Option<&[u8]>, 1 byte per bool
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
    let path_os: &std::path::Path = match path_str.to_str() {
        Ok(s) => std::path::Path::new(s),
        Err(_) => return -1,
    };
    match unsafe { (*h).inner.write(path_os) } {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn tq_load(path: *const c_char, err: *mut *mut c_char) -> *mut TqHandle {
    let path_str = unsafe { std::ffi::CStr::from_ptr(path) };
    let path_os: &std::path::Path = match path_str.to_str() {
        Ok(s) => std::path::Path::new(s),
        Err(_) => {
            if !err.is_null() {
                unsafe { *err = prefixed_err(error::PREFIX_IO, "path is not valid UTF-8") }
            }
            return std::ptr::null_mut();
        }
    };
    match CoreTq::load(path_os) {
        Ok(core) => {
            if !err.is_null() {
                unsafe { *err = std::ptr::null_mut() }
            }
            Box::into_raw(Box::new(TqHandle {
                inner: Box::new(core),
            }))
        }
        Err(e) => {
            if !err.is_null() {
                unsafe { *err = prefixed_err(error::PREFIX_IO, e) }
            }
            std::ptr::null_mut()
        }
    }
}
