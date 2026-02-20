//! Safe wrappers around the cuGraph C API (RAPIDS 24.6).
//!
//! Loaded dynamically via `libloading` so the crate compiles without RAPIDS
//! at build time.  At runtime, `libcugraph.so` must be on `LD_LIBRARY_PATH`.
//!
//! cuGraph C API reference:
//!   https://docs.rapids.ai/api/cugraph/nightly/api_docs/cugraph_c/
//!
//! All unsafe code is contained within this module. Public functions are safe.

use std::ffi::c_void;
use libloading::{Library, Symbol};

use crate::CuGraphError;

// ── Opaque C handles (pointer-sized) ─────────────────────────────────────────

#[repr(transparent)]
struct ResourceHandle(*mut c_void);
#[repr(transparent)]
struct Graph(*mut c_void);
#[repr(transparent)]
struct DevArray(*mut c_void);
#[repr(transparent)]
struct DevArrayView(*mut c_void);
#[repr(transparent)]
struct PathsResult(*mut c_void);
#[repr(transparent)]
struct CentralityResult(*mut c_void);
#[repr(transparent)]
struct CugraphError(*mut c_void);

// ── Error code ───────────────────────────────────────────────────────────────

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum ErrorCode {
    Success = 0,
    Unknown = 1,
    InvalidInput = 2,
    OutOfMemory = 3,
    #[allow(dead_code)]
    NotImplemented = 4,
}

// ── Data type IDs (cuGraph C API) ─────────────────────────────────────────────

#[repr(i32)]
#[allow(dead_code)]
enum DataTypeId {
    Int8    = 0,
    Int16   = 1,
    Int32   = 2,
    Int64   = 3,
    Uint8   = 4,
    Uint16  = 5,
    Uint32  = 6,
    Uint64  = 7,
    Float32 = 8,
    Float64 = 9,
}

// ── Graph properties ──────────────────────────────────────────────────────────

#[repr(C)]
struct GraphProperties {
    is_symmetric:  bool,
    is_multigraph: bool,
}

// ── Function signatures ───────────────────────────────────────────────────────

type FnCreateHandle     = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FnFreeHandle       = unsafe extern "C" fn(*mut c_void);
type FnDevArrayCreate   = unsafe extern "C" fn(
    handle: *mut c_void,
    n_elems: usize,
    dtype: i32,
    out: *mut *mut c_void,
    err: *mut *mut c_void,
) -> i32;
type FnDevArrayView     = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FnCopyFromHost     = unsafe extern "C" fn(
    handle: *mut c_void,
    arr: *mut c_void,
    host_ptr: *const u8,
    err: *mut *mut c_void,
) -> i32;
type FnCopyToHost       = unsafe extern "C" fn(
    handle: *mut c_void,
    host_ptr: *mut u8,
    arr_view: *mut c_void,
    err: *mut *mut c_void,
) -> i32;
type FnDevArrayFree     = unsafe extern "C" fn(*mut c_void);
type FnSgGraphCreate    = unsafe extern "C" fn(
    handle: *mut c_void,
    props: *const GraphProperties,
    src: *mut c_void,
    dst: *mut c_void,
    weights: *mut c_void,
    edge_ids: *mut c_void,
    edge_type_ids: *mut c_void,
    store_transposed: bool,
    num_vertices: usize,
    do_expensive_check: bool,
    out_graph: *mut *mut c_void,
    err: *mut *mut c_void,
) -> i32;
type FnSgGraphFree      = unsafe extern "C" fn(*mut c_void);
type FnBfs              = unsafe extern "C" fn(
    handle: *mut c_void,
    graph: *mut c_void,
    sources: *mut c_void,
    direction_optimizing: bool,
    depth_limit: u64,
    compute_predecessors: bool,
    do_expensive_check: bool,
    result: *mut *mut c_void,
    err: *mut *mut c_void,
) -> i32;
type FnSssp             = unsafe extern "C" fn(
    handle: *mut c_void,
    graph: *mut c_void,
    source: *mut c_void,
    cutoff: f64,
    compute_predecessors: bool,
    do_expensive_check: bool,
    result: *mut *mut c_void,
    err: *mut *mut c_void,
) -> i32;
type FnPagerank         = unsafe extern "C" fn(
    handle: *mut c_void,
    graph: *mut c_void,
    precomputed_vertex_out_weight_sums: *mut c_void,
    initial_guess_vertices: *const f64,
    initial_guess_values: *const f64,
    num_initial_vertices: usize,
    alpha: f64,
    epsilon: f64,
    max_iterations: usize,
    do_expensive_check: bool,
    result: *mut *mut c_void,
    err: *mut *mut c_void,
) -> i32;
type FnPathsGetVertices = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FnPathsGetDistances= unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FnPathsFree        = unsafe extern "C" fn(*mut c_void);
type FnCentGetVertices  = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FnCentGetValues    = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FnCentFree         = unsafe extern "C" fn(*mut c_void);
type FnErrMsg           = unsafe extern "C" fn(*const c_void) -> *const i8;
type FnErrFree          = unsafe extern "C" fn(*mut c_void);

// ── Session ───────────────────────────────────────────────────────────────────

/// Loaded cuGraph library with a RAFT resource handle.
///
/// One session per operation is fine — the handle is cheap to create.
pub struct CuGraphSession {
    _lib: Library,         // keep alive
    handle: ResourceHandle,
    // Function pointers loaded once
    fn_free_handle:        FnFreeHandle,
    fn_dev_array_create:   FnDevArrayCreate,
    fn_dev_array_view:     FnDevArrayView,
    fn_copy_from_host:     FnCopyFromHost,
    fn_copy_to_host:       FnCopyToHost,
    fn_dev_array_free:     FnDevArrayFree,
    fn_sg_graph_create:    FnSgGraphCreate,
    fn_sg_graph_free:      FnSgGraphFree,
    fn_bfs:                FnBfs,
    fn_sssp:               FnSssp,
    fn_pagerank:           FnPagerank,
    fn_paths_vertices:     FnPathsGetVertices,
    fn_paths_distances:    FnPathsGetDistances,
    fn_paths_free:         FnPathsFree,
    fn_cent_vertices:      FnCentGetVertices,
    fn_cent_values:        FnCentGetValues,
    fn_cent_free:          FnCentFree,
    fn_err_msg:            FnErrMsg,
    fn_err_free:           FnErrFree,
}

impl CuGraphSession {
    pub fn load(lib_path: &str) -> Result<Self, CuGraphError> {
        // Safety: loading a shared library is inherently unsafe but controlled here.
        let lib = unsafe {
            Library::new(lib_path).map_err(|e| CuGraphError::LibraryLoad {
                path: lib_path.to_string(),
                source: Box::new(e),
            })?
        };

        macro_rules! sym {
            ($name:literal, $ty:ty) => {
                unsafe {
                    let s: Symbol<$ty> = lib.get($name).map_err(|e| CuGraphError::LibraryLoad {
                        path: format!("{}::{}", lib_path, std::str::from_utf8($name).unwrap_or("?")),
                        source: Box::new(e),
                    })?;
                    *s  // copy the raw fn pointer out of Symbol to avoid lifetime issue
                }
            };
        }

        let fn_create_handle:     FnCreateHandle     = sym!(b"cugraph_create_resource_handle\0",  FnCreateHandle);
        let fn_free_handle:       FnFreeHandle        = sym!(b"cugraph_free_resource_handle\0",    FnFreeHandle);
        let fn_dev_array_create:  FnDevArrayCreate    = sym!(b"cugraph_type_erased_device_array_create\0", FnDevArrayCreate);
        let fn_dev_array_view:    FnDevArrayView      = sym!(b"cugraph_type_erased_device_array_view\0",   FnDevArrayView);
        let fn_copy_from_host:    FnCopyFromHost      = sym!(b"cugraph_type_erased_device_array_copy_from_host\0", FnCopyFromHost);
        let fn_copy_to_host:      FnCopyToHost        = sym!(b"cugraph_type_erased_device_array_view_copy_to_host\0", FnCopyToHost);
        let fn_dev_array_free:    FnDevArrayFree      = sym!(b"cugraph_type_erased_device_array_free\0",   FnDevArrayFree);
        let fn_sg_graph_create:   FnSgGraphCreate     = sym!(b"cugraph_sg_graph_create\0",    FnSgGraphCreate);
        let fn_sg_graph_free:     FnSgGraphFree       = sym!(b"cugraph_sg_graph_free\0",      FnSgGraphFree);
        let fn_bfs:               FnBfs               = sym!(b"cugraph_bfs\0",                FnBfs);
        let fn_sssp:              FnSssp              = sym!(b"cugraph_sssp\0",               FnSssp);
        let fn_pagerank:          FnPagerank          = sym!(b"cugraph_pagerank\0",           FnPagerank);
        let fn_paths_vertices:    FnPathsGetVertices  = sym!(b"cugraph_paths_result_get_vertices\0",  FnPathsGetVertices);
        let fn_paths_distances:   FnPathsGetDistances = sym!(b"cugraph_paths_result_get_distances\0", FnPathsGetDistances);
        let fn_paths_free:        FnPathsFree         = sym!(b"cugraph_paths_result_free\0",          FnPathsFree);
        let fn_cent_vertices:     FnCentGetVertices   = sym!(b"cugraph_centrality_result_get_vertices\0", FnCentGetVertices);
        let fn_cent_values:       FnCentGetValues     = sym!(b"cugraph_centrality_result_get_values\0",   FnCentGetValues);
        let fn_cent_free:         FnCentFree          = sym!(b"cugraph_centrality_result_free\0",         FnCentFree);
        let fn_err_msg:           FnErrMsg            = sym!(b"cugraph_error_message\0",  FnErrMsg);
        let fn_err_free:          FnErrFree           = sym!(b"cugraph_error_free\0",    FnErrFree);

        // Create RAFT resource handle (pass null for default CUDA stream)
        let raw_handle = unsafe { fn_create_handle(std::ptr::null_mut()) };
        if raw_handle.is_null() {
            return Err(CuGraphError::Cuda("cugraph_create_resource_handle returned null".into()));
        }

        Ok(Self {
            _lib: lib,
            handle: ResourceHandle(raw_handle),
            fn_free_handle,
            fn_dev_array_create,
            fn_dev_array_view,
            fn_copy_from_host,
            fn_copy_to_host,
            fn_dev_array_free,
            fn_sg_graph_create,
            fn_sg_graph_free,
            fn_bfs,
            fn_sssp,
            fn_pagerank,
            fn_paths_vertices,
            fn_paths_distances,
            fn_paths_free,
            fn_cent_vertices,
            fn_cent_values,
            fn_cent_free,
            fn_err_msg,
            fn_err_free,
        })
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn check(&self, code: i32, err_ptr: *mut c_void) -> Result<(), CuGraphError> {
        if code == ErrorCode::Success as i32 {
            return Ok(());
        }
        let message = if !err_ptr.is_null() {
            let msg_ptr = unsafe { (self.fn_err_msg)(err_ptr as *const c_void) };
            let s = if msg_ptr.is_null() {
                "unknown cuGraph error".to_string()
            } else {
                unsafe { std::ffi::CStr::from_ptr(msg_ptr) }
                    .to_string_lossy()
                    .into_owned()
            };
            unsafe { (self.fn_err_free)(err_ptr) };
            s
        } else {
            format!("cuGraph error code {}", code)
        };
        Err(CuGraphError::Api { code, message })
    }

    /// Allocate a device array and upload a host slice.
    unsafe fn upload<T>(&self, data: &[T], dtype: DataTypeId) -> Result<DevArray, CuGraphError> {
        let mut arr_ptr: *mut c_void = std::ptr::null_mut();
        let mut err_ptr: *mut c_void = std::ptr::null_mut();

        let code = (self.fn_dev_array_create)(
            self.handle.0,
            data.len(),
            dtype as i32,
            &mut arr_ptr,
            &mut err_ptr,
        );
        self.check(code, err_ptr)?;

        let mut err_ptr2: *mut c_void = std::ptr::null_mut();
        let code2 = (self.fn_copy_from_host)(
            self.handle.0,
            arr_ptr,
            data.as_ptr() as *const u8,
            &mut err_ptr2,
        );
        self.check(code2, err_ptr2)?;

        Ok(DevArray(arr_ptr))
    }

    /// Download a device array view into a Vec<T>.
    unsafe fn download<T: Default + Clone>(
        &self,
        view: *mut c_void,
        count: usize,
    ) -> Result<Vec<T>, CuGraphError> {
        let mut out = vec![T::default(); count];
        let mut err_ptr: *mut c_void = std::ptr::null_mut();
        let code = (self.fn_copy_to_host)(
            self.handle.0,
            out.as_mut_ptr() as *mut u8,
            view,
            &mut err_ptr,
        );
        self.check(code, err_ptr)?;
        Ok(out)
    }

    /// Build a cuGraph (COO from CSR offsets+col_idx) and run BFS.
    pub fn bfs(
        &self,
        offsets: &[u32],
        col_idx: &[u32],
        n_vertices: usize,
        source: u32,
        depth_limit: u64,
    ) -> Result<(Vec<u32>, Vec<u32>), CuGraphError> {
        // Convert CSR → COO edge list required by cugraph_sg_graph_create
        let mut src_coo: Vec<u32> = Vec::with_capacity(col_idx.len());
        let mut dst_coo: Vec<u32> = Vec::with_capacity(col_idx.len());
        for (u, window) in offsets.windows(2).enumerate() {
            for &v in &col_idx[window[0] as usize..window[1] as usize] {
                src_coo.push(u as u32);
                dst_coo.push(v);
            }
        }
        let weights_f32 = vec![1.0f32; src_coo.len()];

        unsafe {
            let da_src = self.upload(&src_coo, DataTypeId::Int32)?;
            let da_dst = self.upload(&dst_coo, DataTypeId::Int32)?;
            let da_w   = self.upload(&weights_f32, DataTypeId::Float32)?;
            let v_src  = (self.fn_dev_array_view)(da_src.0);
            let v_dst  = (self.fn_dev_array_view)(da_dst.0);
            let v_w    = (self.fn_dev_array_view)(da_w.0);

            let props = GraphProperties { is_symmetric: false, is_multigraph: false };
            let mut graph_ptr: *mut c_void = std::ptr::null_mut();
            let mut err_ptr:   *mut c_void = std::ptr::null_mut();
            let code = (self.fn_sg_graph_create)(
                self.handle.0, &props,
                v_src, v_dst, v_w,
                std::ptr::null_mut(), std::ptr::null_mut(),
                false, n_vertices, false,
                &mut graph_ptr, &mut err_ptr,
            );
            self.check(code, err_ptr)?;

            // source array
            let sources = [source];
            let da_src_v = self.upload(&sources, DataTypeId::Int32)?;
            let v_src_v  = (self.fn_dev_array_view)(da_src_v.0);

            let mut result_ptr: *mut c_void = std::ptr::null_mut();
            let mut err_ptr2:   *mut c_void = std::ptr::null_mut();
            let code2 = (self.fn_bfs)(
                self.handle.0, graph_ptr, v_src_v,
                false, depth_limit, false, false,
                &mut result_ptr, &mut err_ptr2,
            );
            self.check(code2, err_ptr2)?;

            // Extract results
            let v_verts = (self.fn_paths_vertices)(result_ptr);
            let v_dists = (self.fn_paths_distances)(result_ptr);
            let vertices: Vec<u32> = self.download(v_verts, n_vertices)?;
            let distances: Vec<u32> = self.download(v_dists, n_vertices)?;

            (self.fn_paths_free)(result_ptr);
            (self.fn_sg_graph_free)(graph_ptr);
            (self.fn_dev_array_free)(da_src.0);
            (self.fn_dev_array_free)(da_dst.0);
            (self.fn_dev_array_free)(da_w.0);
            (self.fn_dev_array_free)(da_src_v.0);

            Ok((vertices, distances))
        }
    }

    /// SSSP (weighted shortest paths) from a single source.
    pub fn sssp(
        &self,
        offsets: &[u32],
        col_idx: &[u32],
        weights: &[f32],
        n_vertices: usize,
        source: u32,
    ) -> Result<(Vec<u32>, Vec<f64>), CuGraphError> {
        let mut src_coo: Vec<u32> = Vec::with_capacity(col_idx.len());
        let mut dst_coo: Vec<u32> = Vec::with_capacity(col_idx.len());
        let mut w_coo:   Vec<f32> = Vec::with_capacity(col_idx.len());
        for (u, window) in offsets.windows(2).enumerate() {
            for (idx, &v) in col_idx[window[0] as usize..window[1] as usize].iter().enumerate() {
                src_coo.push(u as u32);
                dst_coo.push(v);
                w_coo.push(weights[window[0] as usize + idx]);
            }
        }

        unsafe {
            let da_src = self.upload(&src_coo, DataTypeId::Int32)?;
            let da_dst = self.upload(&dst_coo, DataTypeId::Int32)?;
            let da_w   = self.upload(&w_coo,   DataTypeId::Float32)?;
            let v_src  = (self.fn_dev_array_view)(da_src.0);
            let v_dst  = (self.fn_dev_array_view)(da_dst.0);
            let v_w    = (self.fn_dev_array_view)(da_w.0);

            let props = GraphProperties { is_symmetric: false, is_multigraph: false };
            let mut graph_ptr: *mut c_void = std::ptr::null_mut();
            let mut err_ptr:   *mut c_void = std::ptr::null_mut();
            let code = (self.fn_sg_graph_create)(
                self.handle.0, &props,
                v_src, v_dst, v_w,
                std::ptr::null_mut(), std::ptr::null_mut(),
                false, n_vertices, false,
                &mut graph_ptr, &mut err_ptr,
            );
            self.check(code, err_ptr)?;

            let sources = [source];
            let da_sv = self.upload(&sources, DataTypeId::Int32)?;
            let v_sv  = (self.fn_dev_array_view)(da_sv.0);

            let mut result_ptr: *mut c_void = std::ptr::null_mut();
            let mut err_ptr2:   *mut c_void = std::ptr::null_mut();
            let code2 = (self.fn_sssp)(
                self.handle.0, graph_ptr, v_sv,
                f64::MAX, false, false,
                &mut result_ptr, &mut err_ptr2,
            );
            self.check(code2, err_ptr2)?;

            let v_verts = (self.fn_paths_vertices)(result_ptr);
            let v_dists = (self.fn_paths_distances)(result_ptr);
            let vertices:  Vec<u32> = self.download(v_verts, n_vertices)?;
            let distances: Vec<f64> = self.download(v_dists, n_vertices)?;

            (self.fn_paths_free)(result_ptr);
            (self.fn_sg_graph_free)(graph_ptr);
            (self.fn_dev_array_free)(da_src.0);
            (self.fn_dev_array_free)(da_dst.0);
            (self.fn_dev_array_free)(da_w.0);
            (self.fn_dev_array_free)(da_sv.0);

            Ok((vertices, distances))
        }
    }

    /// PageRank via power iteration.
    pub fn pagerank(
        &self,
        offsets: &[u32],
        col_idx: &[u32],
        weights: &[f32],
        n_vertices: usize,
        alpha: f64,
        epsilon: f64,
        max_iter: usize,
    ) -> Result<(Vec<u32>, Vec<f64>), CuGraphError> {
        let mut src_coo: Vec<u32> = Vec::with_capacity(col_idx.len());
        let mut dst_coo: Vec<u32> = Vec::with_capacity(col_idx.len());
        let mut w_coo:   Vec<f32> = Vec::with_capacity(col_idx.len());
        for (u, window) in offsets.windows(2).enumerate() {
            for (idx, &v) in col_idx[window[0] as usize..window[1] as usize].iter().enumerate() {
                src_coo.push(u as u32);
                dst_coo.push(v);
                w_coo.push(weights[window[0] as usize + idx]);
            }
        }

        unsafe {
            let da_src = self.upload(&src_coo, DataTypeId::Int32)?;
            let da_dst = self.upload(&dst_coo, DataTypeId::Int32)?;
            let da_w   = self.upload(&w_coo,   DataTypeId::Float32)?;
            let v_src  = (self.fn_dev_array_view)(da_src.0);
            let v_dst  = (self.fn_dev_array_view)(da_dst.0);
            let v_w    = (self.fn_dev_array_view)(da_w.0);

            // PageRank needs store_transposed = true
            let props = GraphProperties { is_symmetric: false, is_multigraph: false };
            let mut graph_ptr: *mut c_void = std::ptr::null_mut();
            let mut err_ptr:   *mut c_void = std::ptr::null_mut();
            let code = (self.fn_sg_graph_create)(
                self.handle.0, &props,
                v_src, v_dst, v_w,
                std::ptr::null_mut(), std::ptr::null_mut(),
                true, // store_transposed for PageRank
                n_vertices, false,
                &mut graph_ptr, &mut err_ptr,
            );
            self.check(code, err_ptr)?;

            let mut result_ptr: *mut c_void = std::ptr::null_mut();
            let mut err_ptr2:   *mut c_void = std::ptr::null_mut();
            let code2 = (self.fn_pagerank)(
                self.handle.0, graph_ptr,
                std::ptr::null_mut(), // precomputed out-weight sums
                std::ptr::null(),     // initial guess vertices
                std::ptr::null(),     // initial guess values
                0,                    // num initial vertices
                alpha, epsilon, max_iter, false,
                &mut result_ptr, &mut err_ptr2,
            );
            self.check(code2, err_ptr2)?;

            let v_verts  = (self.fn_cent_vertices)(result_ptr);
            let v_scores = (self.fn_cent_values)(result_ptr);
            let vertices: Vec<u32> = self.download(v_verts,  n_vertices)?;
            let scores:   Vec<f64> = self.download(v_scores, n_vertices)?;

            (self.fn_cent_free)(result_ptr);
            (self.fn_sg_graph_free)(graph_ptr);
            (self.fn_dev_array_free)(da_src.0);
            (self.fn_dev_array_free)(da_dst.0);
            (self.fn_dev_array_free)(da_w.0);

            Ok((vertices, scores))
        }
    }
}

impl Drop for CuGraphSession {
    fn drop(&mut self) {
        if !self.handle.0.is_null() {
            unsafe { (self.fn_free_handle)(self.handle.0) };
        }
    }
}

// Safety: cuGraph resource handle is not thread-safe, but CuGraphSession is
// only used within a single-threaded context per operation (created, used, dropped).
unsafe impl Send for CuGraphSession {}
