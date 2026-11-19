mod audio;
mod doctor;
mod dsp;
mod lrclib;
mod pipeline;
mod segmenter;
mod server;
mod updater;
mod whisper_align;
mod whisper_model;

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};
use pipeline::{AlignOptions, run_alignment_pipeline};
use std::path::PathBuf;
use whisper_align::{DevicePreference, WhisperSize};

#[derive(Parser)]
#[command(
    name = "lrc-align",
    version,
    about = "Word-by-word Enhanced LRC & WLRC aligner with a localhost notebook web UI"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Port for the localhost web UI (default mode)
    #[arg(long, default_value_t = 8765)]
    port: u16,

    /// Do not auto-open the browser
    #[arg(long)]
    no_open: bool,

    /// Require a working GPU (CUDA/Metal) and fail instead of falling back to CPU
    #[arg(long)]
    force_gpu: bool,

    /// Self-update lrc-align from GitHub (git pull, rebuild with current GPU features, reinstall)
    #[arg(short = 'U', long = "update")]
    update: bool,

    /// Force CPU-only rebuild when combined with -U / --update
    #[arg(long, conflicts_with = "force_gpu")]
    cpu: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the localhost drag-and-drop web UI
    Serve {
        #[arg(long, default_value_t = 8765)]
        port: u16,
        #[arg(long)]
        no_open: bool,
        /// Require a working GPU (CUDA/Metal) and fail instead of falling back to CPU
        #[arg(long)]
        force_gpu: bool,
    },
    /// Verify compiled backends, GPU compute kernel, embedded dependencies, and cached models
    Doctor {
        /// Exit with non-zero status if a working GPU is not available
        #[arg(long)]
        require_gpu: bool,
        /// Skip online reachability checks (Hugging Face Hub and LRCLIB)
        #[arg(long)]
        offline: bool,
    },
    /// Align an audio file directly from the CLI
    Align {
        /// Path to audio file (.m4a, .flac, .mp3, .wav, .ogg)
        audio: PathBuf,
        /// Optional path to input .lrc or plain lyrics .txt (defaults to embedded lyrics)
        #[arg(short, long)]
        lyrics: Option<PathBuf>,
        /// Output .lrc path (Enhanced LRC)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Also write .wlrc (word-per-line for terminal visualizers)
        #[arg(long)]
        wlrc: Option<PathBuf>,
        /// Overwrite original audio file with embedded Enhanced LRC (delete old, place new)
        #[arg(long)]
        in_place: bool,
        /// Whisper model size: tiny, base, small, medium, large-v3-turbo
        #[arg(short, long, default_value = "medium")]
        model: String,
        /// Language code (auto, ja, en, ...)
        #[arg(long, default_value = "auto")]
        lang: String,
        /// Shift all output timestamps by milliseconds (negative values make lyrics appear earlier)
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        offset_ms: f64,
        /// Disable GPU acceleration
        #[arg(long, conflicts_with = "force_gpu")]
        cpu: bool,
        /// Require a working GPU (CUDA/Metal) and fail instead of falling back to CPU
        #[arg(long, conflicts_with = "cpu")]
        force_gpu: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.update {
        return updater::run_self_update(cli.force_gpu, cli.cpu);
    }
    match cli.command {
        None => server::serve(cli.port, !cli.no_open, cli.force_gpu).await,
        Some(Commands::Serve {
            port,
            no_open,
            force_gpu,
        }) => server::serve(port, !no_open, force_gpu).await,
        Some(Commands::Doctor {
            require_gpu,
            offline,
        }) => {
            let report = doctor::run_doctor(!offline).await;
            doctor::print_doctor_report(&report);
            if require_gpu && !report.gpu_ready {
                return Err(anyhow!(
                    "GPU verification failed (--require-gpu was specified)"
                ));
            }
            if !report.ok {
                return Err(anyhow!("One or more required system checks failed"));
            }
            Ok(())
        }
        Some(Commands::Align {
            audio,
            lyrics,
            output,
            wlrc,
            in_place,
            model,
            lang,
            offset_ms,
            cpu,
            force_gpu,
        }) => {
            let device_pref = if force_gpu {
                DevicePreference::ForceGpu
            } else if cpu {
                DevicePreference::Cpu
            } else {
                DevicePreference::Auto
            };
            let lyrics_override = match lyrics {
                Some(p) => Some(std::fs::read_to_string(&p)?),
                None => None,
            };
            let opts = AlignOptions {
                model_size: WhisperSize::from_str_name(&model),
                device_pref,
                vocal_isolation: true,
                pause_snapping: true,
                language: if lang == "auto" { None } else { Some(lang) },
                raw_lyrics_override: lyrics_override,
                offset_ms,
            };
            let out = run_alignment_pipeline(&audio, opts, None, |p, msg| {
                println!("[{:3}%] {msg}", (p * 100.0).round() as u32);
            })
            .await?;

            let lrc_path = output.unwrap_or_else(|| audio.with_extension("lrc"));
            std::fs::write(&lrc_path, &out.enhanced_lrc)?;
            println!("Wrote Enhanced LRC -> {}", lrc_path.display());

            if let Some(wlrc_path) = wlrc {
                std::fs::write(&wlrc_path, &out.wlrc)?;
                println!("Wrote WLRC -> {}", wlrc_path.display());
            }
            if in_place {
                audio::embed_lyrics_in_place(&audio, &out.enhanced_lrc)?;
                println!("Replaced original audio in-place -> {}", audio.display());
            }
            Ok(())
        }
    }
}
