// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use rust_embed::Embed;

#[derive(Embed)]
#[folder = "../../dashboard/dist"]
struct DashboardAssets;

/// Serve the React dashboard's `index.html`.
pub async fn index() -> impl IntoResponse {
    match DashboardAssets::get("index.html") {
        Some(file) => Html(String::from_utf8_lossy(&file.data).into_owned()).into_response(),
        None => (StatusCode::NOT_FOUND, "dashboard not built — run `npm run build` in dashboard/").into_response(),
    }
}

/// Serve any static asset from the dashboard dist folder.
pub async fn static_asset(path: axum::extract::Path<String>) -> impl IntoResponse {
    let path = path.0;
    match DashboardAssets::get(&path) {
        Some(file) => {
            let mime = mime_guess::from_path(&path).first_or_octet_stream();
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, mime.as_ref())
                .body(axum::body::Body::from(file.data.to_vec()))
                .unwrap()
                .into_response()
        }
        // SPA fallback: serve index.html for client-side routes
        None => match DashboardAssets::get("index.html") {
            Some(file) => Html(String::from_utf8_lossy(&file.data).into_owned()).into_response(),
            None => (StatusCode::NOT_FOUND, "not found").into_response(),
        },
    }
}
