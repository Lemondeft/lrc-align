use crate::whisper_model::WhisperModel;
use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as cw, Config};
use hf_hub::HFClient;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use realfft::RealFftPlanner;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WhisperSize {
    Tiny,
    Base,
    Small,
    Medium,
    LargeV3Turbo,
}

impl WhisperSize {
    pub fn repo(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai", "whisper-tiny"),
            Self::Base => ("openai", "whisper-base"),
            Self::Small => ("openai", "whisper-small"),
            Self::Medium => ("openai", "whisper-medium"),
            Self::LargeV3Turbo => ("openai", "whisper-large-v3-turbo"),
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Tiny => "openai/whisper-tiny (~150 MB)",
            Self::Base => "openai/whisper-base (~290 MB)",
            Self::Small => "openai/whisper-small (~970 MB)",
            Self::Medium => "openai/whisper-medium (~3.0 GB)",
            Self::LargeV3Turbo => "openai/whisper-large-v3-turbo (~1.6 GB)",
        }
    }

    pub fn from_str_name(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "tiny" => Self::Tiny,
            "base" => Self::Base,
            "small" => Self::Small,
            "large-v3-turbo" | "turbo" => Self::LargeV3Turbo,
            _ => Self::Medium,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlignedUnit {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct AlignmentResult {
    pub units: Vec<AlignedUnit>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct CoarseLine {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    alignment_heads: Vec<[usize; 2]>,
}

pub struct WhisperAligner {
    model: WhisperModel,
    tokenizer: Tokenizer,
    mel_filters: Vec<f32>,
    alignment_heads: Vec<(usize, usize)>,
    device: Device,
    dtype: DType,
    sot_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
    eot_token: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DevicePreference {
    Auto,
    ForceGpu,
    Cpu,
}

impl DevicePreference {
    pub fn from_str_name(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "force-gpu" | "force_gpu" | "gpu" | "cuda" | "metal" => Self::ForceGpu,
            "cpu" => Self::Cpu,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CachedModelInfo {
    pub name: &'static str,
    pub repo: String,
    pub status: &'static str,
    pub size_mb: f64,
}

pub fn hf_hub_dir() -> Option<PathBuf> {
    std::env::var_os("HF_HOME")
        .map(|p| PathBuf::from(p).join("hub"))
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|p| PathBuf::from(p).join(".cache/huggingface/hub"))
        })
}

/// Inspect all supported Whisper models in the local Hugging Face cache.
pub fn inspect_cached_models() -> Vec<CachedModelInfo> {
    let hub_dir = hf_hub_dir();
    let all = [
        ("tiny", WhisperSize::Tiny),
        ("base", WhisperSize::Base),
        ("small", WhisperSize::Small),
        ("medium", WhisperSize::Medium),
        ("large-v3-turbo", WhisperSize::LargeV3Turbo),
    ];

    let mut out = Vec::with_capacity(all.len());
    for (name, size) in all {
        let (owner, repo_name) = size.repo();
        let repo = format!("{owner}/{repo_name}");
        let Some(ref hub) = hub_dir else {
            out.push(CachedModelInfo {
                name,
                repo,
                status: "not_downloaded",
                size_mb: 0.0,
            });
            continue;
        };

        let model_dir = hub.join(format!("models--{owner}--{repo_name}"));
        let snap_dir = model_dir.join("snapshots");
        let complete_bytes: u64 = std::fs::read_dir(&snap_dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| std::fs::metadata(e.path().join("model.safetensors")).ok())
            .map(|m| m.len())
            .max()
            .unwrap_or(0);

        if complete_bytes > 1_000_000 {
            out.push(CachedModelInfo {
                name,
                repo,
                status: "cached",
                size_mb: complete_bytes as f64 / (1024.0 * 1024.0),
            });
            continue;
        }

        let blobs_dir = model_dir.join("blobs");
        let incomplete_bytes: u64 = std::fs::read_dir(&blobs_dir)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".incomplete"))
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        if incomplete_bytes > 0 {
            out.push(CachedModelInfo {
                name,
                repo,
                status: "incomplete",
                size_mb: incomplete_bytes as f64 / (1024.0 * 1024.0),
            });
        } else {
            out.push(CachedModelInfo {
                name,
                repo,
                status: "not_downloaded",
                size_mb: 0.0,
            });
        }
    }
    out
}

/// Scan the local Hugging Face cache and return Whisper model names whose `model.safetensors` is complete.
pub fn list_cached_models() -> Vec<&'static str> {
    inspect_cached_models()
        .into_iter()
        .filter(|m| m.status == "cached")
        .map(|m| m.name)
        .collect()
}

/// Verify that tensor allocation and matrix multiplication actually work on `device`.
pub fn gpu_smoke_test(device: &Device) -> Result<()> {
    let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
    let a = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], (2, 2), device)?.to_dtype(dtype)?;
    let b = Tensor::from_slice(&[5.0_f32, 6.0, 7.0, 8.0], (2, 2), device)?.to_dtype(dtype)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let vals: Vec<f32> = c.flatten_all()?.to_vec1()?;
    let expected = [19.0_f32, 22.0, 43.0, 50.0];
    for (got, want) in vals.iter().zip(expected.iter()) {
        if (got - want).abs() > 0.1 {
            return Err(anyhow!(
                "GPU compute verification mismatch: got {got}, expected {want}"
            ));
        }
    }
    Ok(())
}

pub fn select_device_checked(pref: DevicePreference) -> Result<(Device, String)> {
    if pref == DevicePreference::Cpu {
        return Ok((Device::Cpu, "CPU".to_string()));
    }

    #[allow(unused_assignments)]
    let mut last_err: Option<String> = None;

    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(dev) => match gpu_smoke_test(&dev) {
                Ok(()) => return Ok((dev, "CUDA GPU (device 0)".to_string())),
                Err(e) => last_err = Some(format!("CUDA compute kernel test failed: {e}")),
            },
            Err(e) => last_err = Some(format!("CUDA device 0 unavailable: {e}")),
        }
    }

    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(dev) => match gpu_smoke_test(&dev) {
                Ok(()) => return Ok((dev, "Apple Metal GPU (device 0)".to_string())),
                Err(e) => last_err = Some(format!("Metal compute kernel test failed: {e}")),
            },
            Err(e) => last_err = Some(format!("Metal device 0 unavailable: {e}")),
        }
    }

    if pref == DevicePreference::ForceGpu {
        let reason = last_err.unwrap_or_else(|| {
            "this binary was compiled without --features cuda or --features metal".to_string()
        });
        return Err(anyhow!(
            "Force GPU requested, but no working GPU backend is available ({reason}). Rebuild with `cargo build --release --features cuda` (NVIDIA) or `--features metal` (Apple Silicon)."
        ));
    }

    Ok((Device::Cpu, "CPU".to_string()))
}

pub fn select_device(prefer_gpu: bool) -> (Device, String) {
    let pref = if prefer_gpu {
        DevicePreference::Auto
    } else {
        DevicePreference::Cpu
    };
    select_device_checked(pref).unwrap_or((Device::Cpu, "CPU".to_string()))
}

/// Slaney-normalized mel filterbank matching OpenAI Whisper / faster-whisper.
pub fn compute_whisper_mel_filters(n_mels: usize) -> Vec<f32> {
    let sr = 16000.0_f64;
    let n_fft = cw::N_FFT;
    let n_bins = n_fft / 2 + 1; // 201
    let fftfreqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sr / n_fft as f64)
        .collect();

    let min_mel = 0.0_f64;
    let max_mel = 45.245640471924965_f64;
    let f_min = 0.0_f64;
    let f_sp = 200.0_f64 / 3.0;
    let min_log_hz = 1000.0_f64;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4_f64).ln() / 27.0;

    let n_pts = n_mels + 2;
    let mut freqs = Vec::with_capacity(n_pts);
    for i in 0..n_pts {
        let mel = min_mel + (max_mel - min_mel) * (i as f64) / ((n_pts - 1) as f64);
        let hz = if mel >= min_log_mel {
            min_log_hz * (logstep * (mel - min_log_mel)).exp()
        } else {
            f_min + f_sp * mel
        };
        freqs.push(hz);
    }

    let mut weights = vec![0.0_f32; n_mels * n_bins];
    for j in 0..n_mels {
        let f_left = freqs[j];
        let f_center = freqs[j + 1];
        let f_right = freqs[j + 2];
        let enorm = 2.0 / (f_right - f_left);
        let d_left = f_center - f_left;
        let d_right = f_right - f_center;
        for k in 0..n_bins {
            let f = fftfreqs[k];
            let lower = (f - f_left) / d_left;
            let upper = (f_right - f) / d_right;
            let w = lower.min(upper).max(0.0) * enorm;
            weights[j * n_bins + k] = w as f32;
        }
    }
    weights
}

fn default_alignment_heads(size: WhisperSize, cfg: &Config) -> Vec<(usize, usize)> {
    match size {
        WhisperSize::Tiny => vec![(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
        WhisperSize::Base => vec![
            (3, 1),
            (4, 2),
            (4, 3),
            (4, 7),
            (5, 1),
            (5, 2),
            (5, 4),
            (5, 6),
        ],
        WhisperSize::Small => vec![
            (5, 3),
            (5, 9),
            (8, 0),
            (8, 4),
            (8, 7),
            (9, 0),
            (9, 7),
            (9, 9),
        ],
        WhisperSize::Medium => vec![(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
        WhisperSize::LargeV3Turbo => vec![(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
    }
    .into_iter()
    .filter(|&(l, h)| l < cfg.decoder_layers && h < cfg.decoder_attention_heads)
    .collect()
}

struct HubProgress {
    tx: tokio::sync::mpsc::UnboundedSender<String>,
    last_pct: Arc<Mutex<u64>>,
    label: &'static str,
}

impl ProgressHandler for HubProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let (done, total) = match event {
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                ..
            }) => (*bytes_completed, *total_bytes),
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                let Some(f) = files.last() else { return };
                (f.bytes_completed, f.total_bytes)
            }
            _ => return,
        };
        if total == 0 || done == 0 {
            return;
        }
        let pct = (done.saturating_mul(100) / total).min(100);
        let mut last = self.last_pct.lock().unwrap();
        if pct >= *last + 2 || (pct == 100 && *last < 100) {
            *last = pct;
            let mb_done = done as f64 / (1024.0 * 1024.0);
            let mb_tot = total as f64 / (1024.0 * 1024.0);
            let _ = self.tx.send(format!(
                "Downloading {}: {:.1} / {:.1} MB ({pct}%)",
                self.label, mb_done, mb_tot
            ));
        }
    }
}

async fn download_with_progress<F>(
    client: &HFClient,
    owner: &str,
    repo_name: &str,
    filename: &'static str,
    log: &mut F,
) -> Result<PathBuf>
where
    F: FnMut(String),
{
    let repo = client.model(owner, repo_name);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let prog = Progress::new(HubProgress {
        tx,
        last_pct: Arc::new(Mutex::new(0)),
        label: filename,
    });
    let fut = async {
        repo.download_file()
            .filename(filename)
            .progress(prog)
            .send()
            .await
            .with_context(|| format!("Failed to download {filename} from {owner}/{repo_name}"))
    };
    let (res, _) = tokio::join!(fut, async {
        while let Some(msg) = rx.recv().await {
            log(msg);
        }
    });
    res
}

impl WhisperAligner {
    pub async fn load<F>(
        size: WhisperSize,
        device_pref: DevicePreference,
        mut log: F,
    ) -> Result<(Self, String)>
    where
        F: FnMut(String),
    {
        let (device, dev_label) = select_device_checked(device_pref)?;
        let (owner, repo_name) = size.repo();
        log(format!(
            "Preparing model {} on {} (cached automatically after first download)...",
            size.display_name(),
            dev_label
        ));

        let client = HFClient::new().context("Failed to initialize Hugging Face Hub client")?;

        log("Checking config.json & tokenizer.json...".to_string());
        let config_path: PathBuf =
            download_with_progress(&client, owner, repo_name, "config.json", &mut log).await?;
        let tokenizer_path: PathBuf =
            download_with_progress(&client, owner, repo_name, "tokenizer.json", &mut log).await?;

        let gen_heads: Vec<(usize, usize)> =
            match download_with_progress(&client, owner, repo_name, "generation_config.json", &mut log).await {
                Ok(p) => {
                    let raw = std::fs::read_to_string(&p).unwrap_or_default();
                    serde_json::from_str::<GenerationConfig>(&raw)
                        .map(|g| g.alignment_heads.into_iter().map(|[l, h]| (l, h)).collect())
                        .unwrap_or_default()
                }
                Err(_) => Vec::new(),
            };

        log("Checking model.safetensors (first run downloads weights)...".to_string());
        let weights_path: PathBuf =
            download_with_progress(&client, owner, repo_name, "model.safetensors", &mut log).await?;

        let dtype = if device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        log(format!(
            "Loading weights into memory ({})...",
            if dtype == DType::F16 { "FP16" } else { "FP32" }
        ));
        let cfg: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).context("Cannot read config.json")?,
        )
        .context("Cannot parse config.json")?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer.json: {e}"))?;

        let alignment_heads = if !gen_heads.is_empty() {
            gen_heads
        } else {
            default_alignment_heads(size, &cfg)
        };

        let mel_filters = compute_whisper_mel_filters(cfg.num_mel_bins);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device)
                .context("Failed to mmap model.safetensors")?
        };
        let model = WhisperModel::load(&vb, cfg).context("Failed to load WhisperModel")?;

        let tok_id = |s: &str| -> Result<u32> {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| anyhow!("Missing special token {s}"))
        };

        let sot_token = tok_id(cw::SOT_TOKEN)?;
        let transcribe_token = tok_id(cw::TRANSCRIBE_TOKEN)?;
        let no_timestamps_token = tok_id(cw::NO_TIMESTAMPS_TOKEN)?;
        let eot_token = tok_id(cw::EOT_TOKEN)?;

        log("Whisper model ready.".to_string());
        Ok((
            Self {
                model,
                tokenizer,
                mel_filters,
                alignment_heads,
                device,
                dtype,
                sot_token,
                transcribe_token,
                no_timestamps_token,
                eot_token,
            },
            dev_label,
        ))
    }

    pub fn language_token(&self, lang: &str) -> u32 {
        let tag = format!("<|{lang}|>");
        self.tokenizer
            .token_to_id(&tag)
            .or_else(|| self.tokenizer.token_to_id("<|en|>"))
            .unwrap_or(self.sot_token + 1)
    }

    fn build_mel_30s(&self, samples_16k: &[f32]) -> Result<(Tensor, usize)> {
        let copy_len = samples_16k.len().min(cw::N_SAMPLES);
        let n_mels = self.model.config.num_mel_bins;
        let mel_3000 = fast_whisper_mel_30s(&samples_16k[..copy_len], n_mels, &self.mel_filters)?;
        let mel = Tensor::from_vec(mel_3000, (1, n_mels, cw::N_FRAMES), &self.device)?
            .to_dtype(self.dtype)?;
        let num_audio_frames = (((copy_len as f64) / 16000.0) * 50.0).ceil() as usize;
        Ok((mel, num_audio_frames.clamp(1, 1500)))
    }

    /// Detect dominant language using acoustic Whisper logits combined with lyrics text priors.
    pub fn detect_language_with_hint(
        &mut self,
        samples_16k: &[f32],
        lyrics_hint: &str,
    ) -> Result<String> {
        if let Some(script_lang) = detect_script_language(lyrics_hint) {
            return Ok(script_lang.to_string());
        }
        if let Some(text_lang) = strong_latin_lyrics_language(lyrics_hint) {
            return Ok(text_lang.to_string());
        }

        let (mel, _) = self.build_mel_30s(samples_16k)?;
        self.model.reset_kv_cache();
        let xa = self.model.encoder.forward(&mel, true)?;
        let sot = Tensor::new(&[[self.sot_token]], &self.device)?;
        let (hidden, _) = self.model.decoder.forward_with_cross_attn(&sot, &xa, 0, true, false)?;
        let logits = self
            .model
            .decoder
            .final_linear(&hidden.i((.., 0, ..))?)?
            .to_dtype(DType::F32)?;
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;

        let text_bias = score_latin_lyrics_hint(lyrics_hint);
        let candidates = ["ja", "en", "zh", "ko", "id", "ms", "es", "fr", "de", "pt", "it"];
        let mut best_lang = "en";
        let mut best_val = f32::NEG_INFINITY;
        for &lang in &candidates {
            if let Some(id) = self.tokenizer.token_to_id(&format!("<|{lang}|>")) {
                if let Some(&v) = logits_vec.get(id as usize) {
                    let biased = v + text_bias(lang);
                    if biased > best_val {
                        best_val = biased;
                        best_lang = lang;
                    }
                }
            }
        }
        if best_lang == "ms" && text_bias("id") > 0.0 {
            return Ok("id".to_string());
        }
        Ok(best_lang.to_string())
    }

    /// Run forced cross-attention DTW alignment on a single audio window `samples_16k` (<= 30s).
    pub fn align_segment(
        &mut self,
        samples_16k: &[f32],
        clean_text: &str,
        lang: &str,
    ) -> Result<AlignmentResult> {
        if clean_text.is_empty() || samples_16k.is_empty() {
            return Ok(AlignmentResult {
                units: Vec::new(),
                confidence: 0.0,
            });
        }
        let encoding = self
            .tokenizer
            .encode(clean_text, false)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
        let text_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if text_tokens.is_empty() {
            return Ok(AlignmentResult {
                units: Vec::new(),
                confidence: 0.0,
            });
        }

        let max_ctx = self.model.config.max_target_positions;
        let lang_tok = self.language_token(lang);
        let sot_seq = [
            self.sot_token,
            lang_tok,
            self.transcribe_token,
            self.no_timestamps_token,
        ];
        if sot_seq.len() + text_tokens.len() + 1 > max_ctx {
            return Err(anyhow!(
                "Segment exceeds Whisper max token length ({max_ctx})"
            ));
        }

        let mut all_tokens = Vec::with_capacity(sot_seq.len() + text_tokens.len() + 1);
        all_tokens.extend_from_slice(&sot_seq);
        all_tokens.extend_from_slice(&text_tokens);
        all_tokens.push(self.eot_token);

        let (mel, num_frames) = self.build_mel_30s(samples_16k)?;
        self.model.reset_kv_cache();
        let xa = self.model.encoder.forward(&mel, true)?;
        let tok_tensor = Tensor::new(all_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let (_hidden, cross_qks) =
            self.model
                .decoder
                .forward_with_cross_attn(&tok_tensor, &xa, 0, true, true)?;

        let n_text_plus_eot = text_tokens.len() + 1;
        // Causal decoder query at no_timestamps (sot_seq.len() - 1) predicts text_tokens[0], shifting cross-attn by -1.
        let query_start = sot_seq.len() - 1;

        let mut avg_matrix = vec![0.0_f32; n_text_plus_eot * num_frames];
        let mut heads_used = 0usize;

        for &(layer, head) in &self.alignment_heads {
            if layer >= cross_qks.len() {
                continue;
            }
            let qk = cross_qks[layer]
                .i((0, head, query_start..query_start + n_text_plus_eot, 0..num_frames))?
                .contiguous()?
                .to_dtype(DType::F32)?;
            let probs = candle_nn::ops::softmax_last_dim(&qk)?;
            let raw: Vec<f32> = probs.flatten_all()?.to_vec1()?;

            // Z-score normalize across tokens (dim = -2) per time frame
            let mut normed = vec![0.0_f32; n_text_plus_eot * num_frames];
            for f in 0..num_frames {
                let mut sum = 0.0_f32;
                for t in 0..n_text_plus_eot {
                    sum += raw[t * num_frames + f];
                }
                let mean = sum / n_text_plus_eot as f32;
                let mut var = 0.0_f32;
                for t in 0..n_text_plus_eot {
                    let d = raw[t * num_frames + f] - mean;
                    var += d * d;
                }
                let std = (var / n_text_plus_eot as f32).sqrt().max(1e-6);
                for t in 0..n_text_plus_eot {
                    normed[t * num_frames + f] = (raw[t * num_frames + f] - mean) / std;
                }
            }

            // 1D median filter (width = 7) along time axis
            let filtered = median_filter_rows(&normed, n_text_plus_eot, num_frames, 7);
            for idx in 0..avg_matrix.len() {
                avg_matrix[idx] += filtered[idx];
            }
            heads_used += 1;
        }

        if heads_used == 0 {
            return Err(anyhow!("No valid alignment heads available"));
        }
        let inv_h = 1.0 / heads_used as f32;
        for v in &mut avg_matrix {
            *v *= inv_h;
        }

        let (jump_times, tok_conf) = dtw_with_pre_speech(&avg_matrix, text_tokens.len(), num_frames);

        // Merge BPE tokens into valid complete Unicode chunks (preventing split UTF-8 characters)
        let mut units = Vec::new();
        let mut acc_ids: Vec<u32> = Vec::new();
        let mut group_start_tok = 0usize;

        for (ti, &tok_id) in text_tokens.iter().enumerate() {
            acc_ids.push(tok_id);
            let decoded = self
                .tokenizer
                .decode(&acc_ids, true)
                .map_err(|e| anyhow!("Decode failed: {e}"))?;
            if !decoded.contains('\u{FFFD}') {
                let start = jump_times[group_start_tok];
                let end = jump_times[ti + 1];
                let conf_slice = &tok_conf[group_start_tok..=ti];
                let conf = conf_slice.iter().copied().sum::<f32>() / conf_slice.len().max(1) as f32;
                if !decoded.is_empty() {
                    units.push(AlignedUnit {
                        text: decoded,
                        start,
                        end: end.max(start),
                        confidence: conf,
                    });
                }
                acc_ids.clear();
                group_start_tok = ti + 1;
            }
        }

        if !acc_ids.is_empty() {
            let decoded = self
                .tokenizer
                .decode(&acc_ids, true)
                .unwrap_or_default()
                .replace('\u{FFFD}', "");
            if !decoded.is_empty() {
                let start = jump_times[group_start_tok];
                let end = jump_times[text_tokens.len()];
                let conf_slice = &tok_conf[group_start_tok..text_tokens.len()];
                let conf = conf_slice.iter().copied().sum::<f32>() / conf_slice.len().max(1) as f32;
                units.push(AlignedUnit {
                    text: decoded,
                    start,
                    end: end.max(start),
                    confidence: conf,
                });
            }
        }

        let seg_conf = if units.is_empty() {
            0.0
        } else {
            units.iter().map(|u| u.confidence).sum::<f32>() / units.len() as f32
        };

        Ok(AlignmentResult {
            units,
            confidence: seg_conf,
        })
    }

    /// Coarse greedy transcription when no lyrics exist at all.
    pub fn transcribe_coarse(&mut self, samples_16k: &[f32], lang: &str) -> Result<Vec<CoarseLine>> {
        let ts_begin = self
            .tokenizer
            .token_to_id("<|0.00|>")
            .ok_or_else(|| anyhow!("Missing timestamp token <|0.00|>"))?;
        let lang_tok = self.language_token(lang);
        let max_dur = samples_16k.len() as f64 / 16000.0;
        let mut lines = Vec::new();
        let mut seek_samples = 0usize;

        while seek_samples < samples_16k.len() {
            let chunk_start_sec = seek_samples as f64 / 16000.0;
            let chunk_end = (seek_samples + cw::N_SAMPLES).min(samples_16k.len());
            let (mel, _) = self.build_mel_30s(&samples_16k[seek_samples..chunk_end])?;
            self.model.reset_kv_cache();
            let xa = self.model.encoder.forward(&mel, true)?;

            let mut tokens = vec![self.sot_token, lang_tok, self.transcribe_token];
            let mut pos_offset = 0usize;
            for step in 0..220 {
                let inp = if step == 0 {
                    Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?
                } else {
                    Tensor::new(&[*tokens.last().unwrap()], &self.device)?.unsqueeze(0)?
                };
                let (hidden, _) = self.model.decoder.forward_with_cross_attn(
                    &inp,
                    &xa,
                    pos_offset,
                    step == 0,
                    false,
                )?;
                pos_offset = tokens.len();
                let last_hidden = hidden.i((.., hidden.dim(1)? - 1, ..))?;
                let logits: Vec<f32> = self
                    .model
                    .decoder
                    .final_linear(&last_hidden)?
                    .to_dtype(DType::F32)?
                    .squeeze(0)?
                    .to_vec1()?;

                // Pick argmax while suppressing non-transcribe control tokens except EOT/timestamps
                let mut best_id = self.eot_token;
                let mut best_logit = f32::NEG_INFINITY;
                for (id, &val) in logits.iter().enumerate() {
                    let uid = id as u32;
                    if step == 0 && uid < ts_begin {
                        continue; // Force initial timestamp token
                    }
                    if uid != self.eot_token && uid < ts_begin && self.model.config.suppress_tokens.contains(&uid) {
                        continue;
                    }
                    if val > best_logit {
                        best_logit = val;
                        best_id = uid;
                    }
                }
                tokens.push(best_id);
                if best_id == self.eot_token {
                    break;
                }
            }

            let mut cur_start: Option<f64> = None;
            let mut cur_text_ids: Vec<u32> = Vec::new();
            let mut last_ts_sec = 0.0_f64;

            for &tok in &tokens[3..] {
                if tok == self.eot_token {
                    break;
                }
                if tok >= ts_begin {
                    let ts_sec = (tok - ts_begin) as f64 * 0.02;
                    last_ts_sec = ts_sec;
                    if let Some(s) = cur_start {
                        if !cur_text_ids.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&cur_text_ids, true)
                                .unwrap_or_default()
                                .trim()
                                .to_string();
                            if !text.is_empty() && ts_sec > s {
                                lines.push(CoarseLine {
                                    start: chunk_start_sec + s,
                                    end: (chunk_start_sec + ts_sec).min(max_dur),
                                    text,
                                });
                            }
                            cur_text_ids.clear();
                            cur_start = Some(ts_sec);
                        } else {
                            cur_start = Some(ts_sec);
                        }
                    } else {
                        cur_start = Some(ts_sec);
                    }
                } else {
                    cur_text_ids.push(tok);
                }
            }

            let advance = if last_ts_sec > 2.0 {
                (last_ts_sec * 16000.0) as usize
            } else {
                cw::N_SAMPLES
            };
            seek_samples += advance.max(16000);
        }

        Ok(lines)
    }
}

fn median_filter_rows(data: &[f32], rows: usize, cols: usize, width: usize) -> Vec<f32> {
    let half = width / 2;
    let mut out = vec![0.0_f32; rows * cols];
    let mut win = Vec::with_capacity(width);
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            win.clear();
            for offset in 0..width {
                let idx = (c as isize + offset as isize - half as isize)
                    .clamp(0, (cols - 1) as isize) as usize;
                win.push(row[idx]);
            }
            win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            out[r * cols + c] = win[win.len() / 2];
        }
    }
    out
}

fn dtw(matrix: &[f32], n: usize, m: usize) -> (Vec<usize>, Vec<usize>) {
    let inf = f64::INFINITY;
    let mut cost = vec![inf; (n + 1) * (m + 1)];
    let mut trace = vec![0u8; (n + 1) * (m + 1)];
    let stride = m + 1;
    cost[0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let c = -(matrix[(i - 1) * m + (j - 1)] as f64);
            let c0 = cost[(i - 1) * stride + (j - 1)];
            let c1 = cost[(i - 1) * stride + j];
            let c2 = cost[i * stride + (j - 1)];

            let (best, dir) = if c0 <= c1 && c0 <= c2 {
                (c0, 0u8)
            } else if c1 <= c2 {
                (c1, 1u8)
            } else {
                (c2, 2u8)
            };
            cost[i * stride + j] = c + best;
            trace[i * stride + j] = dir;
        }
    }

    let mut ti = Vec::new();
    let mut tj = Vec::new();
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        ti.push(i.saturating_sub(1));
        tj.push(j.saturating_sub(1));
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            match trace[i * stride + j] {
                0 => {
                    i -= 1;
                    j -= 1;
                }
                1 => i -= 1,
                _ => j -= 1,
            }
        }
    }
    ti.reverse();
    tj.reverse();
    (ti, tj)
}

/// Prepend a neutral pre-speech baseline row (`0.05` Z-score threshold) before `text_tokens[0]`
/// so leading silence or breath in a padded window is absorbed instead of pinning word 0 to frame 0.
/// Returns `(token_start_times_plus_eot, per_token_confidences)`.
fn dtw_with_pre_speech(
    avg_matrix: &[f32],
    n_text_tokens: usize,
    num_frames: usize,
) -> (Vec<f64>, Vec<f32>) {
    let n_text_plus_eot = n_text_tokens + 1;
    let total_rows = 1 + n_text_plus_eot;
    let mut padded = Vec::with_capacity(total_rows * num_frames);
    padded.resize(num_frames, 0.05_f32);
    padded.extend_from_slice(avg_matrix);

    let (text_idx, time_idx) = dtw(&padded, total_rows, num_frames);

    // Track `[first_frame, last_frame]` for each text/eot row (`row 1..=n_text_plus_eot`).
    let mut first_frame = vec![None; n_text_plus_eot];
    let mut last_frame = vec![0usize; n_text_plus_eot];
    for (&r, &f) in text_idx.iter().zip(time_idx.iter()) {
        if r >= 1 {
            let tok = r - 1;
            if tok < n_text_plus_eot {
                if first_frame[tok].is_none() {
                    first_frame[tok] = Some(f);
                }
                last_frame[tok] = f;
            }
        }
    }

    let mut jump_times = Vec::with_capacity(n_text_plus_eot);
    let mut prev_t = 0.0_f64;
    for tok in 0..n_text_plus_eot {
        let t = first_frame[tok]
            .map(|f| f as f64 / 50.0)
            .unwrap_or(prev_t)
            .max(prev_t);
        prev_t = t;
        jump_times.push(t);
    }

    let mut confidences = Vec::with_capacity(n_text_tokens);
    for tok in 0..n_text_tokens {
        let f_start = first_frame[tok].unwrap_or(0);
        let f_end = last_frame[tok].max(f_start).min(num_frames - 1);
        let row = &avg_matrix[tok * num_frames..(tok + 1) * num_frames];
        let mut sum = 0.0_f32;
        let mut peak = f32::NEG_INFINITY;
        for &v in &row[f_start..=f_end] {
            sum += v;
            if v > peak {
                peak = v;
            }
        }
        let mean = sum / (f_end - f_start + 1) as f32;
        let score = 0.60 * mean + 0.40 * peak;
        let conf = (score / 1.90).clamp(0.05, 0.99);
        confidences.push(conf);
    }

    (jump_times, confidences)
}

fn detect_script_language(text: &str) -> Option<&'static str> {
    let mut jp = 0usize;
    let mut ko = 0usize;
    let mut han = 0usize;
    let mut ru = 0usize;
    for ch in text.chars() {
        match ch {
            '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}' => jp += 1,
            '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}' => ko += 1,
            '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' => han += 1,
            '\u{0400}'..='\u{04FF}' => ru += 1,
            _ => {}
        }
    }
    if jp >= 3 {
        Some("ja")
    } else if ko >= 3 {
        Some("ko")
    } else if han >= 4 {
        Some("zh")
    } else if ru >= 4 {
        Some("ru")
    } else {
        None
    }
}

fn count_latin_stopwords(text: &str) -> (usize, usize) {
    const ID_WORDS: &[&str] = &[
        "yang", "dan", "ini", "itu", "aku", "gue", "gak", "tak", "tidak", "kini", "untuk",
        "dalam", "dengan", "dari", "ke", "di", "ku", "kamu", "kau", "ingin", "lalu", "semua",
        "jadi", "buat", "atau", "juga", "pada", "sudah", "udah", "akan",
    ];
    const EN_WORDS: &[&str] = &[
        "the", "and", "you", "that", "this", "with", "for", "are", "was", "have", "not", "your",
        "our", "why", "cause", "when", "what", "will", "from", "into", "just", "know", "like",
        "love", "dont", "cant", "could", "would", "through", "again",
    ];

    let mut id_hits = 0usize;
    let mut en_hits = 0usize;
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '\'') {
        let w = raw.to_ascii_lowercase();
        if w.is_empty() {
            continue;
        }
        if ID_WORDS.contains(&w.as_str()) {
            id_hits += 1;
        }
        if EN_WORDS.contains(&w.as_str()) {
            en_hits += 1;
        }
    }
    (id_hits, en_hits)
}

fn strong_latin_lyrics_language(text: &str) -> Option<&'static str> {
    let (id_hits, en_hits) = count_latin_stopwords(text);
    if id_hits >= 3 && id_hits >= en_hits.saturating_mul(2).max(1) {
        Some("id")
    } else if en_hits >= 4 && en_hits >= id_hits.saturating_mul(3).max(1) {
        Some("en")
    } else {
        None
    }
}

fn score_latin_lyrics_hint(text: &str) -> impl Fn(&str) -> f32 {
    let (id_hits, en_hits) = count_latin_stopwords(text);
    move |lang: &str| match lang {
        "id" | "ms" if id_hits >= 2 && id_hits > en_hits => 10.0,
        "en" if en_hits >= 3 && en_hits > id_hits * 2 => 6.0,
        _ => 0.0,
    }
}

/// Compute Whisper 30s log-mel spectrogram by running STFT only over active audio frames
/// and filling the silent remainder with the exact clamped silence floor.
fn fast_whisper_mel_30s(samples: &[f32], n_mels: usize, filters: &[f32]) -> Result<Vec<f32>> {
    const N_FFT: usize = cw::N_FFT;
    const HOP: usize = cw::HOP_LENGTH;
    const N_FRAMES: usize = cw::N_FRAMES;
    const N_BINS: usize = N_FFT / 2 + 1;

    let active_frames = if samples.is_empty() {
        0
    } else {
        (samples.len() / HOP + 1).min(N_FRAMES)
    };

    let hann: Vec<f32> = (0..N_FFT)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos()))
        .collect();

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(N_FFT);
    let mut fft_in = vec![0.0_f32; N_FFT];
    let mut spectrum = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();
    let mut power = vec![0.0_f32; N_BINS];

    let mut mel = vec![-10.0_f32; n_mels * N_FRAMES];
    let mut global_max = -10.0_f32;

    for frame in 0..active_frames {
        let offset = frame * HOP;
        let avail = samples.len().saturating_sub(offset).min(N_FFT);
        for j in 0..avail {
            fft_in[j] = hann[j] * samples[offset + j];
        }
        fft_in[avail..N_FFT].fill(0.0);

        r2c.process_with_scratch(&mut fft_in, &mut spectrum, &mut scratch)
            .map_err(|e| anyhow!("Mel FFT failed: {e}"))?;

        power[0] = spectrum[0].norm_sqr();
        for k in 1..N_BINS - 1 {
            // Double non-DC/Nyquist bins to match candle's full complex FFT folding.
            power[k] = 2.0 * spectrum[k].norm_sqr();
        }
        power[N_BINS - 1] = spectrum[N_BINS - 1].norm_sqr();

        for m in 0..n_mels {
            let f_row = &filters[m * N_BINS..(m + 1) * N_BINS];
            let mut sum = 0.0_f32;
            for k in 0..N_BINS {
                sum += power[k] * f_row[k];
            }
            let log_v = sum.max(1e-10).log10();
            mel[m * N_FRAMES + frame] = log_v;
            if log_v > global_max {
                global_max = log_v;
            }
        }
    }

    let mmax = global_max - 8.0;
    let pad_val = (-10.0_f32).max(mmax) / 4.0 + 1.0;
    for m in 0..n_mels {
        let row = &mut mel[m * N_FRAMES..(m + 1) * N_FRAMES];
        for v in &mut row[..active_frames] {
            *v = (*v).max(mmax) / 4.0 + 1.0;
        }
        row[active_frames..].fill(pad_val);
    }

    Ok(mel)
}

#[cfg(test)]
mod tests {
    use super::{compute_whisper_mel_filters, dtw_with_pre_speech, fast_whisper_mel_30s};
    use candle_transformers::models::whisper as cw;

    #[test]
    fn pre_speech_anchor_finds_delayed_onset_instead_of_frame_zero() {
        let n_tokens = 2usize;
        let num_frames = 60usize;
        let mut avg = vec![-0.8_f32; (n_tokens + 1) * num_frames];
        // Silent intro for frames 0..25 (0.00s..0.50s), then tok 0 active at 25..40, tok 1 at 40..52, eot at 52..60.
        for f in 25..40 {
            avg[f] = 1.6;
        }
        for f in 40..52 {
            avg[num_frames + f] = 1.6;
        }
        for f in 52..60 {
            avg[2 * num_frames + f] = 1.6;
        }
        let (jumps, conf) = dtw_with_pre_speech(&avg, n_tokens, num_frames);
        assert!((jumps[0] - 0.50).abs() <= 0.04, "expected ~0.50s, got {}", jumps[0]);
        assert!((jumps[1] - 0.80).abs() <= 0.04, "expected ~0.80s, got {}", jumps[1]);
        assert!(conf[0] > 0.8 && conf[1] > 0.8);
    }

    #[test]
    fn fast_mel_matches_candle_pcm_to_mel() {
        let cfg = cw::Config {
            num_mel_bins: 80,
            max_source_positions: 1500,
            d_model: 512,
            encoder_attention_heads: 8,
            encoder_layers: 6,
            vocab_size: 51865,
            decoder_layers: 6,
            decoder_attention_heads: 8,
            max_target_positions: 448,
            suppress_tokens: Vec::new(),
        };
        let filters = compute_whisper_mel_filters(80);
        let active: Vec<f32> = (0..16000)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let mut padded = vec![0.0_f32; cw::N_SAMPLES];
        padded[..active.len()].copy_from_slice(&active);

        let ref_full = cw::audio::pcm_to_mel(&cfg, &padded, &filters);
        let ref_stride = ref_full.len() / 80;
        let fast = fast_whisper_mel_30s(&active, 80, &filters).unwrap();

        let mut max_diff = 0.0_f32;
        for m in 0..80 {
            for t in 0..cw::N_FRAMES {
                let d = (fast[m * cw::N_FRAMES + t] - ref_full[m * ref_stride + t]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        assert!(max_diff < 1e-4, "max_diff={max_diff}");
    }
}
