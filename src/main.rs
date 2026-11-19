mod audio;
mod dsp;
mod lrclib;
mod pipeline;
mod segmenter;
mod server;
mod whisper_align;
mod whisper_model;

use anyhow::Result;
use clap::{Parser, Subcommand};
use pipeline::{AlignOptions, run_alignment_pipeline};
use std::path::PathBuf;
use whisper_align::WhisperSize;

#[derive(Parser)]
#[command(
    name = "lrc-align",
    version,
    about = "Word-by-word Enhanced LRC & WLRC aligner with a localhost Colab-style web UI"
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
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the localhost drag-and-drop web UI
    Serve {
        #[arg(long, default_value_t = 8765)]
        port: u16,
        #[arg(long)]
        no_open: bool,
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
        /// Also write .wlrc (word-per-line for tacos-terminal-lyrics)
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
        /// Disable GPU acceleration
        #[arg(long)]
        cpu: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => server::serve(cli.port, !cli.no_open).await,
        Some(Commands::Serve { port, no_open }) => server::serve(port, !no_open).await,
        Some(Commands::Align {
            audio,
            lyrics,
            output,
            wlrc,
            in_place,
            model,
            lang,
            cpu,
        }) => {
            let lyrics_override = match lyrics {
                Some(p) => Some(std::fs::read_to_string(&p)?),
                None => None,
            };
            let opts = AlignOptions {
                model_size: WhisperSize::from_str_name(&model),
                prefer_gpu: !cpu,
                vocal_isolation: true,
                pause_snapping: true,
                language: if lang == "auto" { None } else { Some(lang) },
                raw_lyrics_override: lyrics_override,
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
