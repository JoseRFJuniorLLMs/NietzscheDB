//! Role-Based Access Control types for NietzscheDB gRPC API.
//!
//! The [`Role`] enum is injected into `tonic::Request::extensions()` by the
//! auth interceptor in `nietzsche-server`. RPC handlers use [`require_writer`]
//! and [`require_admin`] to enforce access control.

use tonic::{Request, Status};

/// Access role resolved from the API key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Full access: all operations including admin (backup, restore, drop, sleep, zaratustra).
    Admin,
    /// Read + write: insert/delete/update nodes and edges, run queries.
    Writer,
    /// Read-only: get nodes/edges, KNN search, traversals, queries.
    Reader,
}

/// Reject the request if the caller has `Reader` role.
/// Writers and Admins pass through.
pub fn require_writer<T>(req: &Request<T>) -> Result<(), Status> {
    match req.extensions().get::<Role>() {
        Some(Role::Reader) => Err(Status::permission_denied("writer or admin role required")),
        _ => Ok(()),
    }
}

/// Reject the request unless the caller has `Admin` role.
pub fn require_admin<T>(req: &Request<T>) -> Result<(), Status> {
    match req.extensions().get::<Role>() {
        Some(Role::Admin) => Ok(()),
        Some(_) => Err(Status::permission_denied("admin role required")),
        None => Ok(()), // no auth configured
    }
}
