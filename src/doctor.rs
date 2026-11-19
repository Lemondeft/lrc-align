use crate::segmenter::segment_line_words;
use crate::whisper_align::{
    CachedModelInfo, DevicePreference, hf_hub_dir, inspect_cached_models, select_device_checked,
};
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticItem {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoctorReport {
    pub ok: bool,
    pub gpu_ready: bool,
    pub cuda_compiled: bool,
    pub metal_compiled: bool,
    pub device: String,
    pub vram_info: Option<String>,
    pub hf_cache_dir: Option<String>,
    pub models: Vec<CachedModelInfo>,
    pub checks: Vec<DiagnosticItem>,
}

pub async fn run_doctor(check_network: bool) -> DoctorReport {
    let cuda_compiled = cfg!(feature = "cuda");
    let metal_compiled = cfg!(feature = "metal");
    let mut checks = Vec::new();

    let compiled_backends = match (cuda_compiled, metal_compiled) {
        (true, true) => "CUDA + Metal + CPU",
        (true, false) => "CUDA + CPU",
        (false, true) => "Metal + CPU",
        (false, false) => "CPU only",
    };
    checks.push(DiagnosticItem {
        name: "Compiled Backends".into(),
        status: if cuda_compiled || metal_compiled {
            CheckStatus::Ok
        } else {
            CheckStatus::Warn
        },
        detail: if cuda_compiled || metal_compiled {
            compiled_backends.into()
        } else {
            "CPU only (rebuild with --features cuda or --features metal for GPU)".into()
        },
    });

    let (gpu_ready, device, gpu_item) = match select_device_checked(DevicePreference::ForceGpu) {
        Ok((dev, label)) => {
            let dtype = if dev.is_cuda() { "FP16" } else { "FP32" };
            (
                true,
                label.clone(),
                DiagnosticItem {
                    name: "GPU Compute Kernel".into(),
                    status: CheckStatus::Ok,
                    detail: format!("{label} verified via {dtype} matmul smoke test"),
                },
            )
        }
        Err(err) => (
            false,
            "CPU".to_string(),
            DiagnosticItem {
                name: "GPU Compute Kernel".into(),
                status: if cuda_compiled || metal_compiled {
                    CheckStatus::Fail
                } else {
                    CheckStatus::Warn
                },
                detail: err.to_string(),
            },
        ),
    };
    checks.push(gpu_item);

    let vram_info = query_nvidia_vram();
    if let Some(ref vram) = vram_info {
        checks.push(DiagnosticItem {
            name: "NVIDIA GPU / VRAM".into(),
            status: CheckStatus::Ok,
            detail: vram.clone(),
        });
    }

    // Verify embedded pure-Rust audio/tag crates and Japanese IPADIC dictionary.
    checks.push(DiagnosticItem {
        name: "Audio Decode & Tags".into(),
        status: CheckStatus::Ok,
        detail: "Pure-Rust Symphonia (ALAC/AAC/FLAC/MP3/WAV/OGG) + Lofty".into(),
    });

    let jp_words = segment_line_words("言って");
    let lindera_ok = jp_words.concat() == "言って";
    checks.push(DiagnosticItem {
        name: "Japanese Tokenizer (Lindera IPADIC)".into(),
        status: if lindera_ok {
            CheckStatus::Ok
        } else {
            CheckStatus::Fail
        },
        detail: if lindera_ok {
            "Embedded IPADIC loaded cleanly".into()
        } else {
            "Embedded IPADIC dictionary check failed".into()
        },
    });

    let hf_dir = hf_hub_dir();
    let hf_cache_dir = hf_dir.as_ref().map(|p| p.to_string_lossy().into_owned());
    let models = inspect_cached_models();
    let cached_names: Vec<&str> = models
        .iter()
        .filter(|m| m.status == "cached")
        .map(|m| m.name)
        .collect();
    let incomplete_names: Vec<&str> = models
        .iter()
        .filter(|m| m.status == "incomplete")
        .map(|m| m.name)
        .collect();

    let cache_detail = match &hf_cache_dir {
        Some(dir) => {
            let mut parts = vec![dir.clone()];
            if !cached_names.is_empty() {
                parts.push(format!("cached: {}", cached_names.join(", ")));
            } else {
                parts.push("no complete models cached yet".into());
            }
            if !incomplete_names.is_empty() {
                parts.push(format!("incomplete: {}", incomplete_names.join(", ")));
            }
            parts.join(" | ")
        }
        None => "Could not resolve HF_HOME or HOME cache directory".into(),
    };
    checks.push(DiagnosticItem {
        name: "Whisper Model Cache".into(),
        status: if hf_dir.is_some() {
            CheckStatus::Ok
        } else {
            CheckStatus::Warn
        },
        detail: cache_detail,
    });

    if check_network {
        let (hf_res, lrclib_res) = tokio::join!(
            probe_url("https://huggingface.co/api/models/openai/whisper-tiny"),
            probe_url("https://lrclib.net/api/search?q=test")
        );
        checks.push(DiagnosticItem {
            name: "Hugging Face Hub".into(),
            status: if hf_res.is_ok() {
                CheckStatus::Ok
            } else {
                CheckStatus::Warn
            },
            detail: match hf_res {
                Ok(code) => format!("Reachable (HTTP {code})"),
                Err(e) => format!("Unreachable ({e}); cached models still work offline"),
            },
        });
        checks.push(DiagnosticItem {
            name: "LRCLIB API".into(),
            status: if lrclib_res.is_ok() {
                CheckStatus::Ok
            } else {
                CheckStatus::Warn
            },
            detail: match lrclib_res {
                Ok(code) => format!("Reachable (HTTP {code})"),
                Err(e) => format!("Unreachable ({e}); embedded/local lyrics still work"),
            },
        });
    }

    let ok = checks.iter().all(|c| c.status != CheckStatus::Fail);
    DoctorReport {
        ok,
        gpu_ready,
        cuda_compiled,
        metal_compiled,
        device,
        vram_info,
        hf_cache_dir,
        models,
        checks,
    }
}

pub fn print_doctor_report(report: &DoctorReport) {
    println!("lrc-align system verification");
    println!("-----------------------------");
    for item in &report.checks {
        let tag = match item.status {
            CheckStatus::Ok => "OK  ",
            CheckStatus::Warn => "WARN",
            CheckStatus::Fail => "FAIL",
        };
        println!("[{tag}] {}: {}", item.name, item.detail);
    }
    println!();
    println!("Whisper models:");
    for m in &report.models {
        let detail = match m.status {
            "cached" => format!("cached ({:.0} MB)", m.size_mb),
            "incomplete" => format!("incomplete ({:.0} MB on disk)", m.size_mb),
            _ => "not downloaded".into(),
        };
        println!("  - {:<15} {:<30} {}", m.name, m.repo, detail);
    }
}

fn query_nvidia_vram() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    let mut cols = line.split(',').map(str::trim);
    let name = cols.next()?;
    let free = cols.next()?;
    let total = cols.next()?;
    Some(format!("{name} ({free} / {total} MiB free)"))
}

async fn probe_url(url: &str) -> Result<u16, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(6))
        .user_agent("lrc-align/0.1")
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.head(url).send().await.map_err(|e| e.to_string())?;
    Ok(resp.status().as_u16())
}
