use crate::whisper_model::WhisperModel;
use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as cw, Config};
use hf_hub::HFClient;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
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
    sot_token: u32,
    transcribe_token: u32,
    no_timestamps_token: u32,
    eot_token: u32,
}

pub fn select_device(prefer_gpu: bool) -> (Device, String) {
    if prefer_gpu {
        #[cfg(feature = "cuda")]
        if let Ok(dev) = Device::new_cuda(0) {
            return (dev, "CUDA GPU (device 0)".to_string());
        }
        #[cfg(feature = "metal")]
        if let Ok(dev) = Device::new_metal(0) {
            return (dev, "Apple Metal GPU (device 0)".to_string());
        }
    }
    (Device::Cpu, "CPU".to_string())
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
    pub async fn load<F>(size: WhisperSize, prefer_gpu: bool, mut log: F) -> Result<(Self, String)>
    where
        F: FnMut(String),
    {
        let (device, dev_label) = select_device(prefer_gpu);
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

        log("Loading weights into memory...".to_string());
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
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
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
        let mut padded = vec![0.0_f32; cw::N_SAMPLES];
        let copy_len = samples_16k.len().min(cw::N_SAMPLES);
        padded[..copy_len].copy_from_slice(&samples_16k[..copy_len]);

        let full_mel = cw::audio::pcm_to_mel(&self.model.config, &padded, &self.mel_filters);
        let n_mels = self.model.config.num_mel_bins;
        let total_frames = full_mel.len() / n_mels;
        let mut mel_3000 = Vec::with_capacity(n_mels * cw::N_FRAMES);
        for m in 0..n_mels {
            let row_start = m * total_frames;
            mel_3000.extend_from_slice(&full_mel[row_start..row_start + cw::N_FRAMES]);
        }

        let mel = Tensor::from_vec(mel_3000, (1, n_mels, cw::N_FRAMES), &self.device)?;
        let num_audio_frames = (((copy_len as f64) / 16000.0) * 50.0).ceil() as usize;
        Ok((mel, num_audio_frames.clamp(1, 1500)))
    }

    /// Detect dominant language of an audio snippet (`"ja"`, `"en"`, etc.).
    pub fn detect_language(&mut self, samples_16k: &[f32]) -> Result<String> {
        let (mel, _) = self.build_mel_30s(samples_16k)?;
        self.model.reset_kv_cache();
        let xa = self.model.encoder.forward(&mel, true)?;
        let sot = Tensor::new(&[[self.sot_token]], &self.device)?;
        let (hidden, _) = self.model.decoder.forward_with_cross_attn(&sot, &xa, 0, true, false)?;
        let logits = self.model.decoder.final_linear(&hidden.i((.., 0, ..))?)?;
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;

        let candidates = ["ja", "en", "zh", "ko", "id", "es", "fr", "de"];
        let mut best_lang = "en";
        let mut best_val = f32::NEG_INFINITY;
        for &lang in &candidates {
            if let Some(id) = self.tokenizer.token_to_id(&format!("<|{lang}|>")) {
                if let Some(&v) = logits_vec.get(id as usize) {
                    if v > best_val {
                        best_val = v;
                        best_lang = lang;
                    }
                }
            }
        }
        Ok(best_lang.to_string())
    }

    /// Run forced cross-attention DTW alignment on a single audio window `samples_16k` (<= 30s).
    pub fn align_segment(
        &mut self,
        samples_16k: &[f32],
        clean_text: &str,
        lang: &str,
    ) -> Result<Vec<AlignedUnit>> {
        if clean_text.is_empty() || samples_16k.is_empty() {
            return Ok(Vec::new());
        }
        let encoding = self
            .tokenizer
            .encode(clean_text, false)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
        let text_tokens: Vec<u32> = encoding.get_ids().to_vec();
        if text_tokens.is_empty() {
            return Ok(Vec::new());
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

        let (text_idx, time_idx) = dtw(&avg_matrix, n_text_plus_eot, num_frames);

        let mut jump_times = Vec::with_capacity(n_text_plus_eot);
        for k in 0..text_idx.len() {
            if k == 0 || text_idx[k] != text_idx[k - 1] {
                jump_times.push(time_idx[k] as f64 / 50.0);
            }
        }
        while jump_times.len() < n_text_plus_eot {
            let last = *jump_times.last().unwrap_or(&0.0);
            jump_times.push(last);
        }

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
                if !decoded.is_empty() {
                    units.push(AlignedUnit {
                        text: decoded,
                        start,
                        end: end.max(start),
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
                units.push(AlignedUnit {
                    text: decoded,
                    start,
                    end: end.max(start),
                });
            }
        }

        Ok(units)
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
