# lrc-align

[![Built in Rust](https://img.shields.io/badge/Built%20in-Rust-dea584?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Edition 2024](https://img.shields.io/badge/Rust-Edition%202024-1f242d?style=for-the-badge)](https://doc.rust-lang.org/edition-guide/)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20Metal-76b900?style=for-the-badge)](https://github.com/huggingface/candle)
[![License: MIT](https://img.shields.io/badge/License-MIT-3b82f6?style=for-the-badge)](LICENSE)

**Pure-Rust, out-of-the-box word-by-word Enhanced LRC & WLRC forced aligner with a localhost notebook-style drag-and-drop web UI.**

Zero Python, zero FFmpeg, and zero external runtime dependencies. Drop any audio file in your browser, watch live GPU/CPU forced alignment via Whisper cross-attention DTW, preview word-sweeping karaoke playback, and export Enhanced `.lrc`, `.wlrc`, or overwrite your original audio file in-place with embedded synced lyrics.

---

## Highlights

- **Built 100% in Rust**: Single native binary with no Python environment, no `ffmpeg` binary, and no external `mecab` install.
- **Notebook-Style Localhost Web UI**: Four-cell workflow on `http://127.0.0.1:8765`:
  1. Drag & drop audio (`.m4a` ALAC/AAC, `.flac`, `.mp3`, `.wav`, `.ogg`) plus optional `.lrc` / `.txt`
  2. Configure Whisper model (`base`, `small`, `medium`, `large-v3-turbo`), language, vocal isolation, and pause snapping
  3. Live SSE progress logs (including Hugging Face weight download MB/percentage updates)
  4. Real-time word-by-word karaoke preview player (with automatic WAV byte-range transcoding for ALAC `.m4a` playback in browsers)
- **Smart Lyrics Input**: Automatically reads embedded audio tags (`lofty`), fetches synced/plain lyrics from LRCLIB, or auto-transcribes when no lyrics are provided.
- **Center-Vocal DSP & Pause Snapping**: Pure-Rust STFT mid-side Wiener masking (`realfft`), 16 kHz FFT resampling (`rubato`), and 10 ms vocal-band envelope valley snapping (`dsp.rs`).
- **Japanese + Multilingual Word Segmentation**: Embedded IPADIC morphological segmentation via `lindera` for natural Japanese karaoke units, alongside clean space-preserving tokenization for English and other languages.
- **In-Place Original File Overwrite**: Stage, embed, verify, and atomically replace the original audio file on disk (plus optional sibling `.lrc` output) from either the Web UI or CLI `--in-place`.
- **GPU Acceleration**: Out-of-the-box CPU support by default, with optional NVIDIA CUDA (`--features cuda`) and Apple Metal (`--features metal`).

---

## Quickstart

### Prerequisites
- [Rust toolchain](https://rustup.rs/) (`cargo` / `rustc`, Rust 1.85+ for Edition 2024)
- *(Optional)* NVIDIA CUDA Toolkit (`--features cuda`) or macOS Xcode Metal (`--features metal`)

### Launch Web UI

```bash
git clone https://github.com/Lemondeft/lrc-align.git
cd lrc-align

# Out-of-the-box CPU build and launch (opens http://127.0.0.1:8765)
cargo run --release

# NVIDIA CUDA GPU build and launch
cargo run --release --features cuda
# or
./run-gpu.sh

# Apple Silicon Metal GPU build and launch
cargo run --release --features metal
```

Models (`openai/whisper-{base,small,medium,large-v3-turbo}`) download automatically from Hugging Face Hub on first use and are cached locally in `~/.cache/huggingface/hub/`.

---

## CLI Usage

```bash
# Verify GPU compute kernel, embedded dependencies, cached models, and connectivity
lrc-align doctor
lrc-align doctor --require-gpu

# Self-update from GitHub, rebuild with current GPU features, and reinstall
lrc-align -U
lrc-align -U --force-gpu

# Launch localhost Web UI and fail fast if GPU is unavailable
lrc-align --force-gpu

# Align audio using embedded lyrics (or auto-transcribe if none)
lrc-align align song.m4a --model base --force-gpu

# Specify external lyrics, Enhanced LRC output, WLRC output, and timing offset
lrc-align align song.m4a --lyrics lyrics.txt --output song.lrc --wlrc song.wlrc --offset-ms -40

# Embed aligned Enhanced LRC directly into the original audio file in-place
lrc-align align song.m4a --model medium --in-place

# Force CPU execution even on a CUDA build
lrc-align align song.m4a --model base --cpu
```

---

## Output Formats

| Format | Extension | Description |
|---|---|---|
| **Enhanced LRC** | `.lrc` | Line timestamps + per-word inline timestamps (`[00:09.76]<00:09.76>High <00:10.10>dive...`) |
| **WLRC** | `.wlrc` | Word-per-line timestamps (`[00:09.76]High`) for terminal visualizers |
| **Embedded Audio** | `.m4a`, `.flac`, `.mp3`, etc. | Writes synced lyrics directly into audio file metadata tags |

---

## Architecture & Pure-Rust Stack

| Stage | Crate / Module | Purpose |
|---|---|---|
| Audio Decode & Tagging | `symphonia`, `lofty` ([`audio.rs`](src/audio.rs)) | Lossless/lossy decoding, metadata inspection, tag embedding, WAV preview |
| Vocal Isolation & Envelope | `realfft`, `rubato` ([`dsp.rs`](src/dsp.rs)) | Center-panned STFT Wiener vocal mask, 16 kHz resampling, 10 ms vocal envelope |
| Forced Alignment | `candle-core`, `candle-nn`, `tokenizers`, `hf-hub` ([`whisper_align.rs`](src/whisper_align.rs), [`whisper_model.rs`](src/whisper_model.rs)) | Whisper cross-attention DTW word alignment & language detection |
| Segmentation & Formatting | `lindera` ([`segmenter.rs`](src/segmenter.rs)) | Japanese IPADIC morphological word splitting & LRC/WLRC formatting |
| Lyrics Lookup | `reqwest` ([`lrclib.rs`](src/lrclib.rs)) | LRCLIB synced/plain lyrics fetch fallback |
| Web Server & UI | `axum`, `tokio` ([`server.rs`](src/server.rs), [`web/index.html`](web/index.html)) | Localhost notebook UI, SSE progress streaming, HTTP Range WAV preview |

---

## License

MIT License. See [LICENSE](LICENSE).
