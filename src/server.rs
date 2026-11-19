use crate::audio::{
    atomic_write_bytes, embed_lyrics_in_place, embed_lyrics_to_file, inspect_metadata,
    write_preview_wav,
};
use crate::lrclib::fetch_from_lrclib;
use crate::pipeline::{AlignOptions, AlignOutput, ModelCache, run_alignment_pipeline};
use crate::whisper_align::WhisperSize;
use anyhow::Result;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

const INDEX_HTML: &str = include_str!("../web/index.html");

#[derive(Clone)]
struct AppState {
    model_cache: Arc<ModelCache>,
    jobs: Arc<Mutex<HashMap<String, Arc<JobRecord>>>>,
    temp_root: Arc<TempDir>,
}

struct JobRecord {
    tx: broadcast::Sender<JobEvent>,
    history: Mutex<Vec<JobEvent>>,
    result: Mutex<Option<AlignOutput>>,
    audio_path: PathBuf,
    audio_filename: String,
    preview_lock: tokio::sync::Mutex<()>,
}

#[derive(Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum JobEvent {
    Log { progress: f32, message: String },
    Done { output: AlignOutput },
    Error { message: String },
}

#[derive(Deserialize)]
struct LocateRequest {
    filename: String,
    size: Option<u64>,
}

#[derive(Deserialize)]
struct ReplaceRequest {
    target_path: String,
    #[serde(default)]
    write_sidecar_lrc: bool,
}

#[derive(Deserialize)]
struct LrcLibRequest {
    title: String,
    artist: Option<String>,
    album: Option<String>,
    duration_secs: Option<f64>,
}

pub async fn serve(port: u16, open_browser: bool) -> Result<()> {
    let temp_root = Arc::new(tempfile::tempdir()?);
    let state = AppState {
        model_cache: ModelCache::new(),
        jobs: Arc::new(Mutex::new(HashMap::new())),
        temp_root,
    };

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/capabilities", get(capabilities_handler))
        .route("/api/inspect", post(inspect_handler))
        .route("/api/locate", post(locate_handler))
        .route("/api/lrclib", post(lrclib_handler))
        .route("/api/align", post(align_handler))
        .route("/api/jobs/{id}/events", get(job_events_handler))
        .route("/api/jobs/{id}/preview.wav", get(preview_wav_handler))
        .route("/api/jobs/{id}/replace", post(replace_handler))
        .route("/api/jobs/{id}/download/{kind}", get(download_handler))
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(state);

    let bind_addr = format!("127.0.0.1:{port}");
    let listener = match tokio::net::TcpListener::bind(&bind_addr).await {
        Ok(l) => l,
        Err(_) => tokio::net::TcpListener::bind("127.0.0.1:0").await?,
    };
    let actual_addr = listener.local_addr()?;
    let url = format!("http://{actual_addr}");

    println!("lrc-align web UI running at {url}");
    println!("Press Ctrl+C to stop.");

    if open_browser {
        open_url(&url);
    }

    axum::serve(listener, app).await?;
    Ok(())
}

fn open_url(url: &str) {
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd").args(["/C", "start", url]).spawn();
}

async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn capabilities_handler() -> Json<serde_json::Value> {
    let cuda_compiled = cfg!(feature = "cuda");
    let metal_compiled = cfg!(feature = "metal");
    let (_, active_device) = crate::whisper_align::select_device(true);
    let gpu_ready = active_device != "CPU";
    Json(serde_json::json!({
        "cuda_compiled": cuda_compiled,
        "metal_compiled": metal_compiled,
        "gpu_ready": gpu_ready,
        "device": active_device,
    }))
}

async fn inspect_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut filename = String::from("audio.bin");
    let mut bytes = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        if field.name() == Some("audio") {
            if let Some(name) = field.file_name() {
                filename = sanitize_filename(name);
            }
            bytes = field
                .bytes()
                .await
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
                .to_vec();
        }
    }

    if bytes.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No audio file uploaded".into()));
    }

    let tmp_path = state
        .temp_root
        .path()
        .join(format!("inspect_{}_{}", Uuid::new_v4(), filename));
    tokio::fs::write(&tmp_path, &bytes)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let meta = tokio::task::spawn_blocking(move || {
        let res = inspect_metadata(&tmp_path);
        let _ = std::fs::remove_file(&tmp_path);
        res
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "filename": filename,
        "metadata": meta,
    })))
}

async fn lrclib_handler(
    Json(req): Json<LrcLibRequest>,
) -> Result<Json<crate::lrclib::LrcLibResult>, (StatusCode, String)> {
    let res = fetch_from_lrclib(
        &req.title,
        req.artist.as_deref(),
        req.album.as_deref(),
        req.duration_secs,
    )
    .await
    .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(Json(res))
}

async fn align_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut filename = String::from("audio.m4a");
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut lyrics_text: Option<String> = None;
    let mut model = String::from("medium");
    let mut prefer_gpu = true;
    let mut vocal_isolation = true;
    let mut pause_snapping = true;
    let mut language = String::from("auto");

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        match field.name() {
            Some("audio") => {
                if let Some(name) = field.file_name() {
                    filename = sanitize_filename(name);
                }
                audio_bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
                        .to_vec(),
                );
            }
            Some("lyrics") => {
                lyrics_text = Some(
                    field
                        .text()
                        .await
                        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?,
                );
            }
            Some("model") => {
                model = field.text().await.unwrap_or(model);
            }
            Some("prefer_gpu") => {
                prefer_gpu = field.text().await.unwrap_or_default() != "false";
            }
            Some("vocal_isolation") => {
                vocal_isolation = field.text().await.unwrap_or_default() != "false";
            }
            Some("pause_snapping") => {
                pause_snapping = field.text().await.unwrap_or_default() != "false";
            }
            Some("language") => {
                language = field.text().await.unwrap_or(language);
            }
            _ => {}
        }
    }

    let bytes = audio_bytes.ok_or((StatusCode::BAD_REQUEST, "Missing audio upload".into()))?;
    let job_id = Uuid::new_v4().to_string();
    let job_dir = state.temp_root.path().join(&job_id);
    tokio::fs::create_dir_all(&job_dir)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let audio_path = job_dir.join(&filename);
    tokio::fs::write(&audio_path, &bytes)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let (tx, _) = broadcast::channel::<JobEvent>(256);
    let record = Arc::new(JobRecord {
        tx: tx.clone(),
        history: Mutex::new(Vec::new()),
        result: Mutex::new(None),
        audio_path: audio_path.clone(),
        audio_filename: filename.clone(),
        preview_lock: tokio::sync::Mutex::new(()),
    });
    state.jobs.lock().unwrap().insert(job_id.clone(), record.clone());

    let opts = AlignOptions {
        model_size: WhisperSize::from_str_name(&model),
        prefer_gpu,
        vocal_isolation,
        pause_snapping,
        language: if language == "auto" { None } else { Some(language) },
        raw_lyrics_override: lyrics_text,
    };

    let cache = state.model_cache.clone();
    tokio::spawn(async move {
        let rec_for_log = record.clone();
        let emit = move |ev: JobEvent| {
            rec_for_log.history.lock().unwrap().push(ev.clone());
            let _ = rec_for_log.tx.send(ev);
        };

        let emit_prog = |p: f32, msg: String| {
            emit(JobEvent::Log {
                progress: p,
                message: msg,
            });
        };

        match run_alignment_pipeline(&audio_path, opts, Some(cache), emit_prog).await {
            Ok(out) => {
                *record.result.lock().unwrap() = Some(out.clone());
                let ev = JobEvent::Done { output: out };
                record.history.lock().unwrap().push(ev.clone());
                let _ = record.tx.send(ev);
            }
            Err(err) => {
                let ev = JobEvent::Error {
                    message: format!("{err:#}"),
                };
                record.history.lock().unwrap().push(ev.clone());
                let _ = record.tx.send(ev);
            }
        }
    });

    Ok(Json(serde_json::json!({ "job_id": job_id })))
}

async fn job_events_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let record = state
        .jobs
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .ok_or(StatusCode::NOT_FOUND)?;

    let rx = record.tx.subscribe();
    let history = record.history.lock().unwrap().clone();

    let replay = tokio_stream::iter(history.into_iter().map(|ev| {
        let json = serde_json::to_string(&ev).unwrap_or_default();
        Ok::<Event, Infallible>(Event::default().data(json))
    }));
    let live = BroadcastStream::new(rx).filter_map(|res| match res {
        Ok(ev) => {
            let json = serde_json::to_string(&ev).ok()?;
            Some(Ok::<Event, Infallible>(Event::default().data(json)))
        }
        Err(_) => None,
    });

    let stream = futures_util::StreamExt::chain(replay, live);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn download_handler(
    State(state): State<AppState>,
    AxumPath((id, kind)): AxumPath<(String, String)>,
) -> Result<Response, (StatusCode, String)> {
    let record = state
        .jobs
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, "Unknown job id".into()))?;

    let output = record
        .result
        .lock()
        .unwrap()
        .clone()
        .ok_or((StatusCode::CONFLICT, "Job not finished yet".into()))?;

    let stem = std::path::Path::new(&record.audio_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("lyrics");

    match kind.as_str() {
        "lrc" => {
            let fname = format!("{stem}.lrc");
            Ok(download_response(
                output.enhanced_lrc.into_bytes(),
                "text/plain; charset=utf-8",
                &fname,
            ))
        }
        "wlrc" => {
            let fname = format!("{stem}.wlrc");
            Ok(download_response(
                output.wlrc.into_bytes(),
                "text/plain; charset=utf-8",
                &fname,
            ))
        }
        "embedded" => {
            let src = record.audio_path.clone();
            let dst = src
                .parent()
                .unwrap()
                .join(format!("embedded_{}", record.audio_filename));
            let lrc = output.enhanced_lrc.clone();
            tokio::task::spawn_blocking(move || embed_lyrics_to_file(&src, &dst, &lrc).map(|_| dst))
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
                .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("Embed failed: {e}")))
                .and_then(|dst_path| {
                    let bytes = std::fs::read(&dst_path)
                        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                    Ok(download_response(
                        bytes,
                        "application/octet-stream",
                        &record.audio_filename,
                    ))
                })
        }
        _ => Err((StatusCode::BAD_REQUEST, "Unknown download kind".into())),
    }
}

async fn locate_handler(
    Json(req): Json<LocateRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let target_name = sanitize_filename(&req.filename);
    if target_name.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Missing filename".into()));
    }
    let expected_size = req.size;
    let matches = tokio::task::spawn_blocking(move || find_local_matches(&target_name, expected_size))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(serde_json::json!({ "matches": matches })))
}

async fn replace_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Json(req): Json<ReplaceRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let record = state
        .jobs
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, "Unknown job id".into()))?;

    let output = record
        .result
        .lock()
        .unwrap()
        .clone()
        .ok_or((StatusCode::CONFLICT, "Job not finished yet".into()))?;

    let raw_path = req.target_path.trim();
    if raw_path.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Missing target_path".into()));
    }
    let target_path = PathBuf::from(raw_path);
    if !target_path.is_absolute() {
        return Err((
            StatusCode::BAD_REQUEST,
            "target_path must be an absolute path".into(),
        ));
    }

    let expected_ext = std::path::Path::new(&record.audio_filename)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    let actual_ext = target_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    if expected_ext != actual_ext {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Target extension does not match uploaded file ({})",
                record.audio_filename
            ),
        ));
    }

    let write_sidecar = req.write_sidecar_lrc;
    let lrc_text = output.enhanced_lrc.clone();
    let target_clone = target_path.clone();

    let sidecar_written = tokio::task::spawn_blocking(move || -> anyhow::Result<Option<String>> {
        embed_lyrics_in_place(&target_clone, &lrc_text)?;
        if write_sidecar {
            let lrc_path = target_clone.with_extension("lrc");
            atomic_write_bytes(&lrc_path, lrc_text.as_bytes())?;
            return Ok(Some(lrc_path.to_string_lossy().into_owned()));
        }
        Ok(None)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}")))?;

    Ok(Json(serde_json::json!({
        "replaced_path": target_path.to_string_lossy(),
        "sidecar_lrc": sidecar_written,
    })))
}

fn find_local_matches(filename: &str, expected_size: Option<u64>) -> Vec<String> {
    let mut roots = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        for sub in ["Music", "apple-music", "Downloads", "Desktop"] {
            let p = home.join(sub);
            if p.is_dir() {
                roots.push(p);
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }

    let mut visited = std::collections::HashSet::new();
    let mut matches = Vec::new();

    for root in roots {
        let Ok(canon) = std::fs::canonicalize(&root) else {
            continue;
        };
        if !visited.insert(canon.clone()) {
            continue;
        }
        let mut budget = 60_000usize;
        scan_dir_for_file(&canon, filename, expected_size, 0, &mut budget, &mut matches);
        if matches.len() >= 10 {
            break;
        }
    }
    matches
}

fn scan_dir_for_file(
    dir: &std::path::Path,
    filename: &str,
    expected_size: Option<u64>,
    depth: usize,
    budget: &mut usize,
    matches: &mut Vec<String>,
) {
    if depth > 7 || *budget == 0 || matches.len() >= 10 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        if *budget == 0 || matches.len() >= 10 {
            break;
        }
        *budget -= 1;
        let path = entry.path();
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_file() {
            if entry.file_name().to_string_lossy() == filename {
                if let Some(want) = expected_size {
                    let Ok(meta) = entry.metadata() else {
                        continue;
                    };
                    if meta.len() != want {
                        continue;
                    }
                }
                if let Some(s) = path.to_str().map(str::to_string) {
                    if !matches.contains(&s) {
                        matches.push(s);
                    }
                }
            }
        } else if ft.is_dir() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if !s.starts_with('.')
                && !matches!(
                    s.as_ref(),
                    "target" | "node_modules" | "build" | "dist" | "venv" | "__pycache__"
                )
            {
                subdirs.push(path);
            }
        }
    }
    for sub in subdirs {
        if *budget == 0 || matches.len() >= 10 {
            break;
        }
        scan_dir_for_file(&sub, filename, expected_size, depth + 1, budget, matches);
    }
}

fn download_response(bytes: Vec<u8>, content_type: &str, filename: &str) -> Response {
    let safe_ascii: String = filename
        .chars()
        .map(|c| if c.is_ascii() && c != '"' { c } else { '_' })
        .collect();
    let encoded_utf8 = percent_encode_rfc5987(filename);
    let disp = format!("attachment; filename=\"{safe_ascii}\"; filename*=UTF-8''{encoded_utf8}");
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_DISPOSITION, disp)
        .body(Body::from(bytes))
        .unwrap()
}

fn percent_encode_rfc5987(input: &str) -> String {
    let mut out = String::new();
    for b in input.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(*b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn sanitize_filename(name: &str) -> String {
    std::path::Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.bin")
        .replace(['/', '\\', '\0'], "_")
}

async fn preview_wav_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    headers: HeaderMap,
) -> Result<Response, (StatusCode, String)> {
    let record = state
        .jobs
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .ok_or((StatusCode::NOT_FOUND, "Unknown job id".into()))?;

    let wav_path = record
        .audio_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("preview.wav");

    {
        let _guard = record.preview_lock.lock().await;
        if !wav_path.exists() {
            let src = record.audio_path.clone();
            let dst = wav_path.clone();
            tokio::task::spawn_blocking(move || write_preview_wav(&src, &dst))
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
                .map_err(|e| (StatusCode::UNPROCESSABLE_ENTITY, format!("Preview failed: {e:#}")))?;
        }
    }

    serve_wav_with_range(&wav_path, &headers).await
}

async fn serve_wav_with_range(
    path: &std::path::Path,
    headers: &HeaderMap,
) -> Result<Response, (StatusCode, String)> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};

    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let total = file
        .metadata()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .len();

    let range_hdr = headers
        .get(header::RANGE)
        .and_then(|v| v.to_str().ok());

    if let Some(hdr) = range_hdr {
        let Some((start, end)) = parse_byte_range(hdr, total) else {
            return Ok(Response::builder()
                .status(StatusCode::RANGE_NOT_SATISFIABLE)
                .header(header::CONTENT_RANGE, format!("bytes */{total}"))
                .body(Body::empty())
                .unwrap());
        };
        let len = end - start + 1;
        file.seek(std::io::SeekFrom::Start(start))
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let mut buf = Vec::with_capacity(len as usize);
        file.take(len)
            .read_to_end(&mut buf)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        return Ok(Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(header::CONTENT_TYPE, "audio/wav")
            .header(header::ACCEPT_RANGES, "bytes")
            .header(header::CONTENT_LENGTH, len.to_string())
            .header(header::CONTENT_RANGE, format!("bytes {start}-{end}/{total}"))
            .body(Body::from(buf))
            .unwrap());
    }

    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::ACCEPT_RANGES, "bytes")
        .header(header::CONTENT_LENGTH, bytes.len().to_string())
        .body(Body::from(bytes))
        .unwrap())
}

fn parse_byte_range(header: &str, total: u64) -> Option<(u64, u64)> {
    if total == 0 {
        return None;
    }
    let spec = header.strip_prefix("bytes=")?.split(',').next()?.trim();
    let (s_str, e_str) = spec.split_once('-')?;
    if s_str.is_empty() {
        let suffix: u64 = e_str.parse().ok()?;
        if suffix == 0 {
            return None;
        }
        let start = total.saturating_sub(suffix);
        return Some((start, total - 1));
    }
    let start: u64 = s_str.parse().ok()?;
    if start >= total {
        return None;
    }
    let end = if e_str.is_empty() {
        total - 1
    } else {
        e_str.parse::<u64>().ok()?.min(total - 1)
    };
    if end < start {
        return None;
    }
    Some((start, end))
}

#[cfg(test)]
mod tests {
    use super::parse_byte_range;

    #[test]
    fn parses_open_closed_and_suffix_byte_ranges() {
        assert_eq!(parse_byte_range("bytes=0-99", 1000), Some((0, 99)));
        assert_eq!(parse_byte_range("bytes=500-", 1000), Some((500, 999)));
        assert_eq!(parse_byte_range("bytes=-128", 1000), Some((872, 999)));
        assert_eq!(parse_byte_range("bytes=1000-", 1000), None);
        assert_eq!(parse_byte_range("bytes=50-10", 1000), None);
    }
}
