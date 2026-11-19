use anyhow::{Context, Result, anyhow};
use lofty::config::WriteOptions;
use lofty::file::{AudioFile, TaggedFileExt};
use lofty::probe::Probe;
use lofty::tag::{ItemKey, Tag, TagExt, TagType};
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use symphonia::core::audio::sample::Sample;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

#[derive(Debug, Clone, Serialize)]
pub struct AudioMetadata {
    pub title: Option<String>,
    pub artist: Option<String>,
    pub album: Option<String>,
    pub lyrics: Option<String>,
    pub duration_secs: f64,
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
}

pub struct DecodedAudio {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub is_stereo: bool,
    pub sample_rate: u32,
    pub duration_secs: f64,
}

/// Inspect audio tags and properties using pure-Rust `lofty`.
pub fn inspect_metadata(path: &Path) -> Result<AudioMetadata> {
    let tagged_file = Probe::open(path)
        .with_context(|| format!("Failed to open audio file {:?}", path))?
        .guess_file_type()
        .context("Failed to detect audio format")?
        .read()
        .context("Failed to read audio metadata")?;

    let props = tagged_file.properties();
    let tag = tagged_file.primary_tag().or_else(|| tagged_file.first_tag());

    let get_str = |key: ItemKey| -> Option<String> {
        tag.and_then(|t| t.get_string(key).map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
    };

    let lyrics = get_str(ItemKey::Lyrics).or_else(|| get_str(ItemKey::UnsyncLyrics));

    Ok(AudioMetadata {
        title: get_str(ItemKey::TrackTitle),
        artist: get_str(ItemKey::TrackArtist),
        album: get_str(ItemKey::AlbumTitle),
        lyrics,
        duration_secs: props.duration().as_secs_f64(),
        sample_rate: props.sample_rate(),
        channels: props.channels(),
    })
}

/// Copy `src_path` to `dst_path` and embed `lyrics` into the audio file's metadata.
pub fn embed_lyrics_to_file(src_path: &Path, dst_path: &Path, lyrics: &str) -> Result<()> {
    std::fs::copy(src_path, dst_path)
        .with_context(|| format!("Failed to copy {:?} to {:?}", src_path, dst_path))?;
    write_lyrics_tag(dst_path, lyrics)
}

/// Embed `lyrics` into `target_path` by writing a verified temp copy in the same directory,
/// deleting the old file, and moving the new file into `target_path`.
pub fn embed_lyrics_in_place(target_path: &Path, lyrics: &str) -> Result<()> {
    let meta = std::fs::symlink_metadata(target_path)
        .with_context(|| format!("Original file not found: {}", target_path.display()))?;
    if !meta.file_type().is_file() {
        return Err(anyhow!(
            "Target path is not a regular file: {}",
            target_path.display()
        ));
    }

    let tmp_path = same_dir_temp_path(target_path);
    std::fs::copy(target_path, &tmp_path)
        .with_context(|| format!("Failed to stage temp copy at {}", tmp_path.display()))?;

    let staged = (|| -> Result<()> {
        write_lyrics_tag(&tmp_path, lyrics)?;
        let _ = inspect_metadata(&tmp_path).context("Staged audio verification failed")?;
        replace_file_with_staged(&tmp_path, target_path)?;
        Ok(())
    })();

    if staged.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    staged
}

/// Atomically replace `target_path` (deleting the old file if present) with `bytes`.
pub fn atomic_write_bytes(target_path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp_path = same_dir_temp_path(target_path);
    let res = (|| -> Result<()> {
        std::fs::write(&tmp_path, bytes)
            .with_context(|| format!("Failed to write temp file {}", tmp_path.display()))?;
        replace_file_with_staged(&tmp_path, target_path)?;
        Ok(())
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    res
}

fn same_dir_temp_path(target_path: &Path) -> std::path::PathBuf {
    let parent = target_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let ext = target_path
        .extension()
        .and_then(|e| e.to_str())
        .filter(|e| !e.is_empty())
        .unwrap_or("tmp");
    let id = uuid::Uuid::new_v4().simple();
    // Preserve extension so lofty::Probe::guess_file_type recognizes the container format.
    parent.join(format!(".lrcalign_tmp_{id}.{ext}"))
}

fn replace_file_with_staged(staged_path: &Path, target_path: &Path) -> Result<()> {
    match std::fs::rename(staged_path, target_path) {
        Ok(()) => Ok(()),
        Err(_) if target_path.exists() => {
            std::fs::remove_file(target_path)
                .with_context(|| format!("Failed to remove old file {}", target_path.display()))?;
            std::fs::rename(staged_path, target_path).with_context(|| {
                format!(
                    "Failed to move {} into {}",
                    staged_path.display(),
                    target_path.display()
                )
            })
        }
        Err(err) => Err(err).with_context(|| {
            format!(
                "Failed to move {} into {}",
                staged_path.display(),
                target_path.display()
            )
        }),
    }
}

fn write_lyrics_tag(path: &Path, lyrics: &str) -> Result<()> {
    let mut tagged_file = Probe::open(path)
        .context("Failed to open audio file for tagging")?
        .guess_file_type()
        .context("Failed to guess audio file type")?
        .read()
        .context("Failed to read tags for embedding")?;

    let target_tag_type = tagged_file.primary_tag_type();

    if tagged_file.tag(target_tag_type).is_none() {
        tagged_file.insert_tag(Tag::new(target_tag_type));
    }

    let tag = tagged_file
        .tag_mut(target_tag_type)
        .ok_or_else(|| anyhow!("Could not obtain writable tag for {:?}", target_tag_type))?;

    let key = if target_tag_type == TagType::Id3v2 {
        ItemKey::UnsyncLyrics
    } else {
        ItemKey::Lyrics
    };
    tag.insert_text(key, lyrics.to_string());
    tag.save_to_path(path, WriteOptions::default())
        .context("Failed to save embedded lyrics tag")?;

    Ok(())
}

/// Decode an audio file into normalized `f32` channels using pure-Rust `symphonia`.
pub fn decode_audio(path: &Path) -> Result<DecodedAudio> {
    let file = Box::new(File::open(path).with_context(|| format!("Cannot open {:?}", path))?);
    let mss = MediaSourceStream::new(file, Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let fmt_opts = FormatOptions::default();
    let meta_opts = MetadataOptions::default();
    let dec_opts = AudioDecoderOptions::default();

    let mut format = symphonia::default::get_probe()
        .probe(&hint, mss, fmt_opts, meta_opts)
        .context("Unsupported or unrecognized audio container")?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| anyhow!("No audio track found in file"))?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| anyhow!("Missing audio codec parameters"))?
        .clone();

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&audio_params, &dec_opts)
        .context("Unsupported audio codec")?;

    let mut left: Vec<f32> = Vec::new();
    let mut right: Vec<f32> = Vec::new();
    let mut interleaved: Vec<f32> = Vec::new();
    let mut sample_rate: u32 = audio_params.sample_rate.unwrap_or(44100);
    let mut detected_channels: usize = audio_params.channels.map(|c| c.count()).unwrap_or(2);

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::ResetRequired) => break,
            Err(err) => return Err(err).context("Error reading audio packet"),
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = audio_buf.spec();
                sample_rate = spec.rate();
                let ch = spec.channels().count().max(1);
                detected_channels = ch;
                let total = audio_buf.samples_interleaved();
                if total == 0 {
                    continue;
                }
                interleaved.resize(total, f32::MID);
                audio_buf.copy_to_slice_interleaved(&mut interleaved);

                let frames = total / ch;
                left.reserve(frames);
                right.reserve(frames);
                for f in 0..frames {
                    let base = f * ch;
                    if ch == 1 {
                        let s = interleaved[base];
                        left.push(s);
                        right.push(s);
                    } else if ch == 2 {
                        left.push(interleaved[base]);
                        right.push(interleaved[base + 1]);
                    } else {
                        let l = interleaved[base];
                        let r = interleaved[base + 1];
                        let c = interleaved[base + 2] * 0.7071;
                        left.push(l + c);
                        right.push(r + c);
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(err) => return Err(err).context("Error decoding audio packet"),
        }
    }

    if left.is_empty() {
        return Err(anyhow!("Decoded zero audio samples"));
    }

    let duration_secs = left.len() as f64 / sample_rate as f64;
    Ok(DecodedAudio {
        left,
        right,
        is_stereo: detected_channels >= 2,
        sample_rate,
        duration_secs,
    })
}

/// Transcode `audio_path` via Symphonia into a browser-compatible 16-bit PCM WAV file at `wav_path`.
pub fn write_preview_wav(audio_path: &Path, wav_path: &Path) -> Result<()> {
    let decoded = decode_audio(audio_path)?;
    let channels: u16 = if decoded.is_stereo { 2 } else { 1 };
    write_pcm16_wav(
        wav_path,
        decoded.sample_rate,
        channels,
        &decoded.left,
        &decoded.right,
    )
}

fn write_pcm16_wav(
    wav_path: &Path,
    sample_rate: u32,
    channels: u16,
    left: &[f32],
    right: &[f32],
) -> Result<()> {
    let tmp_path = same_dir_temp_path(wav_path);
    let res = (|| -> Result<()> {
        let num_frames = left.len();
        let bytes_per_sample = 2u32;
        let block_align = channels * bytes_per_sample as u16;
        let byte_rate = sample_rate * block_align as u32;
        let data_size = (num_frames as u64)
            .saturating_mul(block_align as u64)
            .min(u32::MAX as u64 - 36) as u32;
        let max_frames = (data_size / block_align as u32) as usize;

        {
            let file = File::create(&tmp_path)
                .with_context(|| format!("Failed to create preview WAV {}", tmp_path.display()))?;
            let mut w = BufWriter::new(file);
            w.write_all(b"RIFF")?;
            w.write_all(&(36 + data_size).to_le_bytes())?;
            w.write_all(b"WAVEfmt ")?;
            w.write_all(&16u32.to_le_bytes())?;
            w.write_all(&1u16.to_le_bytes())?;
            w.write_all(&channels.to_le_bytes())?;
            w.write_all(&sample_rate.to_le_bytes())?;
            w.write_all(&byte_rate.to_le_bytes())?;
            w.write_all(&block_align.to_le_bytes())?;
            w.write_all(&16u16.to_le_bytes())?;
            w.write_all(b"data")?;
            w.write_all(&data_size.to_le_bytes())?;

            let to_i16 = |v: f32| -> [u8; 2] {
                let s = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                s.to_le_bytes()
            };
            for i in 0..max_frames {
                w.write_all(&to_i16(left[i]))?;
                if channels == 2 {
                    let rv = right.get(i).copied().unwrap_or(left[i]);
                    w.write_all(&to_i16(rv))?;
                }
            }
            w.flush()?;
        }

        replace_file_with_staged(&tmp_path, wav_path)?;
        Ok(())
    })();

    if res.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::{atomic_write_bytes, write_pcm16_wav};

    #[test]
    fn atomic_write_replaces_existing_file_and_cleans_temp() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("track.lrc");
        std::fs::write(&target, b"old content").unwrap();

        atomic_write_bytes(&target, b"new content").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new content");

        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".lrcalign_tmp_"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn write_pcm16_wav_creates_valid_header_and_payload_size() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("preview.wav");
        let left = vec![0.0_f32, 0.5, -0.5, 1.0];
        let right = vec![0.0_f32, -0.5, 0.5, -1.0];

        write_pcm16_wav(&wav, 44100, 2, &left, &right).unwrap();
        let bytes = std::fs::read(&wav).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..16], b"WAVEfmt ");
        assert_eq!(bytes.len(), 44 + 4 * 2 * 2);
    }
}
