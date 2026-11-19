use crate::audio::{AudioMetadata, decode_audio, inspect_metadata};
use crate::dsp::{
    compute_vocal_envelope_10ms, detect_vocal_islands, find_active_end, isolate_center_vocals,
    resample_to_16k,
};
use crate::segmenter::{
    AlignedLine, InputLine, WordSpan, format_enhanced_lrc, format_wlrc, has_japanese, parse_lrc,
    segment_line_words, strip_punct_and_space,
};
use crate::whisper_align::{AlignedUnit, WhisperAligner, WhisperSize};
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct AlignOptions {
    pub model_size: WhisperSize,
    pub prefer_gpu: bool,
    pub vocal_isolation: bool,
    pub pause_snapping: bool,
    pub language: Option<String>,
    pub raw_lyrics_override: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AlignOutput {
    pub metadata: AudioMetadata,
    pub device_used: String,
    pub language: String,
    pub enhanced_lrc: String,
    pub wlrc: String,
    pub lines: Vec<AlignedLine>,
    pub elapsed_secs: f64,
}

#[derive(Default)]
pub struct ModelCache {
    inner: Mutex<Option<(WhisperSize, bool, WhisperAligner)>>,
}

impl ModelCache {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

pub async fn run_alignment_pipeline<F>(
    audio_path: &Path,
    opts: AlignOptions,
    cache: Option<Arc<ModelCache>>,
    mut on_progress: F,
) -> Result<AlignOutput>
where
    F: FnMut(f32, String) + Send,
{
    let t0_all = std::time::Instant::now();
    on_progress(0.03, "Reading audio metadata...".into());
    let metadata = inspect_metadata(audio_path).unwrap_or(AudioMetadata {
        title: audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string()),
        artist: None,
        album: None,
        lyrics: None,
        duration_secs: 0.0,
        sample_rate: None,
        channels: None,
    });

    on_progress(0.08, "Decoding audio in pure Rust (Symphonia)...".into());
    let decoded = decode_audio(audio_path)?;
    let duration = decoded.duration_secs;
    on_progress(
        0.16,
        format!(
            "Decoded {:.2}s audio @ {} Hz ({})",
            duration,
            decoded.sample_rate,
            if decoded.is_stereo { "stereo" } else { "mono" }
        ),
    );

    let mono_vocal = if opts.vocal_isolation && decoded.is_stereo {
        on_progress(0.22, "Isolating center-panned vocals (STFT Wiener mask)...".into());
        isolate_center_vocals(
            &decoded.left,
            &decoded.right,
            decoded.is_stereo,
            decoded.sample_rate,
        )?
    } else if decoded.is_stereo {
        decoded
            .left
            .iter()
            .zip(decoded.right.iter())
            .map(|(l, r)| 0.5 * (l + r))
            .collect()
    } else {
        decoded.left
    };

    on_progress(0.30, "Resampling to 16 kHz & extracting 10 ms vocal-band envelope...".into());
    let vocal_16k = resample_to_16k(&mono_vocal, decoded.sample_rate)?;
    let env10 = compute_vocal_envelope_10ms(&vocal_16k);

    let mut cached_aligner: Option<WhisperAligner> = None;
    if let Some(ref c) = cache {
        let mut guard = c.inner.lock().unwrap();
        if let Some((sz, gpu, _)) = guard.as_ref() {
            if *sz == opts.model_size && *gpu == opts.prefer_gpu {
                cached_aligner = guard.take().map(|(_, _, a)| a);
            } else {
                *guard = None;
            }
        }
    }

    let (mut aligner, device_used) = match cached_aligner {
        Some(a) => {
            let label = if opts.prefer_gpu { "Cached GPU/CPU model" } else { "Cached CPU model" };
            on_progress(0.45, format!("Reusing cached {} model...", opts.model_size.display_name()));
            (a, label.to_string())
        }
        None => {
            on_progress(0.35, format!("Loading {}...", opts.model_size.display_name()));
            WhisperAligner::load(opts.model_size, opts.prefer_gpu, |msg| {
                on_progress(0.40, msg);
            })
            .await?
        }
    };

    let raw_source = opts
        .raw_lyrics_override
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| metadata.lyrics.clone());

    let (is_synced, mut input_lines) = match raw_source {
        Some(ref txt) => parse_lrc(txt),
        None => (false, Vec::new()),
    };

    let sample_for_lang = {
        let start = (vocal_16k.len() / 5).min(vocal_16k.len());
        let end = (start + 30 * 16000).min(vocal_16k.len());
        &vocal_16k[start..end]
    };
    let lang = if let Some(l) = opts.language.as_deref().filter(|s| !s.is_empty() && *s != "auto") {
        l.to_string()
    } else if input_lines.iter().any(|l| has_japanese(&l.text)) {
        "ja".to_string()
    } else {
        aligner.detect_language(sample_for_lang).unwrap_or_else(|_| "en".to_string())
    };
    on_progress(0.48, format!("Using language '{lang}' on {device_used}"));

    if input_lines.is_empty() {
        on_progress(0.52, "No lyrics provided — running automatic Whisper transcription...".into());
        let coarse = aligner.transcribe_coarse(&vocal_16k, &lang)?;
        if coarse.is_empty() {
            return Err(anyhow!("Could not detect any sung/spoken lyrics in audio"));
        }
        input_lines = coarse
            .into_iter()
            .map(|c| InputLine {
                timestamp: c.start,
                text: c.text,
            })
            .collect();
    } else if !is_synced {
        on_progress(
            0.52,
            "Untimed plain lyrics detected — locating coarse line timestamps via Whisper...".into(),
        );
        let coarse = aligner.transcribe_coarse(&vocal_16k, &lang)?;
        input_lines = map_plain_lines_to_coarse(&input_lines, &coarse, &env10, duration);
    }

    let total_lines = input_lines.len().max(1);
    let mut aligned_lines = Vec::with_capacity(input_lines.len());

    for (i, line) in input_lines.iter().enumerate() {
        let frac = 0.55 + 0.40 * (i as f32 / total_lines as f32);
        on_progress(
            frac,
            format!("Aligning line {}/{}: {}", i + 1, total_lines, line.text),
        );

        let words = segment_line_words(&line.text);
        if words.is_empty() {
            continue;
        }

        let t_s = line.timestamp.max(0.0).min(duration);
        let t_next = if i + 1 < input_lines.len() {
            input_lines[i + 1].timestamp
        } else {
            duration
        };
        let unit_count = words.len().max(line.text.chars().count() / 3);
        let max_win = if t_next - t_s > 9.0 {
            (2.2 + unit_count as f64 * 0.58).clamp(4.0, 8.5)
        } else {
            15.0
        };
        let t_max = (t_next - 0.02).min(t_s + max_win).max(t_s + 0.3);
        let t_end_active = find_active_end(&env10, t_s, t_max).max(t_s + 0.25);

        let islands = if opts.pause_snapping {
            detect_vocal_islands(&env10, t_s, t_end_active, 250)
        } else {
            vec![(t_s, t_end_active)]
        };

        let mut word_spans = align_line_with_islands(
            &mut aligner,
            &vocal_16k,
            &env10,
            t_s,
            t_end_active,
            &line.text,
            &words,
            &islands,
            &lang,
            opts.pause_snapping,
        )?;

        // Clamp within [t_s, t_max] and enforce strictly monotonic order
        let first_start = islands.first().map(|&(s, _)| s).unwrap_or(t_s).max(t_s);
        let mut cursor = first_start;
        for (wi, w) in word_spans.iter_mut().enumerate() {
            w.start = w.start.max(cursor).min(t_max);
            if wi == 0 {
                w.start = first_start;
            }
            w.end = w.end.max(w.start + 0.02).min(t_max);
            cursor = w.end;
        }

        let line_end = word_spans.last().map(|w| w.end).unwrap_or(t_end_active);
        aligned_lines.push(AlignedLine {
            start: t_s,
            end: line_end,
            text: line.text.clone(),
            words: word_spans,
        });
    }

    if let Some(ref c) = cache {
        let mut guard = c.inner.lock().unwrap();
        *guard = Some((opts.model_size, opts.prefer_gpu, aligner));
    }

    let enhanced_lrc = format_enhanced_lrc(
        &aligned_lines,
        metadata.title.as_deref(),
        metadata.artist.as_deref(),
        metadata.album.as_deref(),
    );
    let wlrc = format_wlrc(
        &aligned_lines,
        metadata.title.as_deref(),
        metadata.artist.as_deref(),
        metadata.album.as_deref(),
    );

    let elapsed = t0_all.elapsed().as_secs_f64();
    on_progress(
        1.0,
        format!(
            "Done! Aligned {} lines in {:.2}s.",
            aligned_lines.len(),
            elapsed
        ),
    );

    Ok(AlignOutput {
        metadata,
        device_used,
        language: lang,
        enhanced_lrc,
        wlrc,
        lines: aligned_lines,
        elapsed_secs: elapsed,
    })
}

#[allow(clippy::too_many_arguments)]
fn align_line_with_islands(
    aligner: &mut WhisperAligner,
    vocal_16k: &[f32],
    env10: &[f32],
    t_s: f64,
    t_end_active: f64,
    orig_line: &str,
    words: &[String],
    islands: &[(f64, f64)],
    lang: &str,
    pause_snapping: bool,
) -> Result<Vec<WordSpan>> {
    // If line has punctuation/space-separated clauses whose count matches detected vocal islands (2..=4),
    // align each clause independently on its exact active vocal island!
    let clauses = split_words_by_clause_breaks(words);
    let max_clause_gap_ok = islands.windows(2).all(|w| w[1].0 - w[0].1 <= 1.5);
    if pause_snapping
        && islands.len() >= 2
        && islands.len() == clauses.len()
        && islands.len() <= 4
        && max_clause_gap_ok
    {
        let mut out = Vec::with_capacity(words.len());
        for ((sub_s, sub_e), clause_words) in islands.iter().zip(clauses.iter()) {
            if clause_words.len() == 1 {
                out.push(WordSpan {
                    text: clause_words[0].clone(),
                    start: *sub_s,
                    end: *sub_e,
                });
            } else {
                let joined: String = clause_words.iter().map(String::as_str).collect();
                let clean = clean_for_whisper(&joined);
                let seg = slice_16k(vocal_16k, *sub_s, *sub_e);
                let units = aligner.align_segment(seg, &clean, lang)?;
                let mut spans = map_units_to_words(&units, clause_words, *sub_s, *sub_e);
                if let Some(first) = spans.first_mut() {
                    first.start = *sub_s;
                }
                if let Some(last) = spans.last_mut() {
                    last.end = last.end.min(*sub_e);
                }
                out.extend(spans);
            }
        }
        return Ok(out);
    }

    let first_s = islands.first().map(|&(s, _)| s).unwrap_or(t_s);
    let clean = clean_for_whisper(orig_line);
    let seg = slice_16k(vocal_16k, first_s, t_end_active);
    let units = aligner.align_segment(seg, &clean, lang)?;
    let mut spans = map_units_to_words(&units, words, first_s, t_end_active);

    if pause_snapping && islands.len() >= 2 && spans.len() >= 2 {
        for g_idx in 0..islands.len() - 1 {
            let gap_start = islands[g_idx].1;
            let gap_end = islands[g_idx + 1].0;
            if !(0.18..=1.5).contains(&(gap_end - gap_start)) {
                continue;
            }
            let gap_mid = 0.5 * (gap_start + gap_end);
            let mut best_k = None;
            let mut best_dist = 0.55_f64;
            for k in 0..spans.len() - 1 {
                let boundary = 0.5 * (spans[k].end + spans[k + 1].start);
                let dist = (boundary - gap_mid).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = Some(k);
                }
            }
            if let Some(k) = best_k {
                if gap_start > spans[k].start + 0.04 && gap_end < spans[k + 1].end - 0.04 {
                    spans[k].end = gap_start;
                    spans[k + 1].start = gap_end;
                }
            }
        }
    }

    if let Some(last) = spans.last_mut() {
        let active_tail = find_active_end(env10, last.start, t_end_active);
        last.end = last.end.min(active_tail).max(last.start + 0.03);
    }

    Ok(spans)
}

fn split_words_by_clause_breaks(words: &[String]) -> Vec<Vec<String>> {
    let is_cjk_line = words.iter().any(|w| {
        w.chars().any(|c| {
            matches!(
                c,
                '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}' | '\u{4E00}'..='\u{9FFF}'
            )
        })
    });
    let mut clauses = Vec::new();
    let mut cur = Vec::new();
    for (idx, w) in words.iter().enumerate() {
        cur.push(w.clone());
        if idx + 1 == words.len() {
            break;
        }
        let trimmed = w.trim_end();
        let punct_break = trimmed.ends_with('、')
            || trimmed.ends_with(',')
            || trimmed.ends_with('。')
            || trimmed.ends_with(';')
            || trimmed.ends_with('?')
            || trimmed.ends_with('!')
            || trimmed.ends_with('？')
            || trimmed.ends_with('！');
        let cjk_space_break = is_cjk_line && (w.ends_with(' ') || w.ends_with('　'));
        if (punct_break || cjk_space_break) && !cur.is_empty() {
            clauses.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        clauses.push(cur);
    }
    clauses
}

/// Keep word spaces and intra-word apostrophes so Whisper BPE sees natural words instead of merged tokens.
fn clean_for_whisper(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = true;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else if ch == '\'' || !crate::segmenter::is_ignorable_punct_or_space(ch) {
            out.push(ch);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn slice_16k(audio_16k: &[f32], t0: f64, t1: f64) -> &[f32] {
    let i0 = ((t0 * 16000.0).floor() as usize).min(audio_16k.len());
    let i1 = ((t1 * 16000.0).ceil() as usize).min(audio_16k.len());
    if i1 <= i0 {
        &[]
    } else {
        &audio_16k[i0..i1]
    }
}

/// Map aligned Unicode token chunks sequentially onto target words.
fn map_units_to_words(
    units: &[AlignedUnit],
    words: &[String],
    offset_sec: f64,
    fallback_end: f64,
) -> Vec<WordSpan> {
    let mut char_times: Vec<(char, f64, f64)> = Vec::new();
    for u in units {
        let clean_u = strip_punct_and_space(&u.text);
        let chars: Vec<char> = clean_u.chars().collect();
        if chars.is_empty() {
            continue;
        }
        let dt = (u.end - u.start).max(0.01) / chars.len() as f64;
        for (idx, &ch) in chars.iter().enumerate() {
            char_times.push((
                ch,
                offset_sec + u.start + idx as f64 * dt,
                offset_sec + u.start + (idx + 1) as f64 * dt,
            ));
        }
    }

    let total_clean_chars: usize = words.iter().map(|w| strip_punct_and_space(w).chars().count()).sum();
    if char_times.is_empty() || char_times.len() != total_clean_chars {
        // Fallback: proportional interpolation across [offset_sec, fallback_end] if tokenizer altered chars (e.g. NFKC/numbers)
        return proportional_map(units, words, offset_sec, fallback_end);
    }

    let mut ci = 0usize;
    let mut spans = Vec::with_capacity(words.len());
    for w in words {
        let clean_w = strip_punct_and_space(w);
        let count = clean_w.chars().count();
        if count == 0 || ci + count > char_times.len() {
            let prev_end = spans.last().map(|s: &WordSpan| s.end).unwrap_or(offset_sec);
            spans.push(WordSpan {
                text: w.clone(),
                start: prev_end,
                end: (prev_end + 0.1).min(fallback_end),
            });
            continue;
        }
        let s = char_times[ci].1;
        let e = char_times[ci + count - 1].2;
        ci += count;
        spans.push(WordSpan {
            text: w.clone(),
            start: s,
            end: e.max(s + 0.02),
        });
    }
    spans
}

fn proportional_map(
    units: &[AlignedUnit],
    words: &[String],
    offset_sec: f64,
    fallback_end: f64,
) -> Vec<WordSpan> {
    let t0 = units
        .first()
        .map(|u| offset_sec + u.start)
        .unwrap_or(offset_sec);
    let t1 = units
        .last()
        .map(|u| offset_sec + u.end)
        .unwrap_or(fallback_end)
        .max(t0 + 0.05);
    let total_chars: usize = words
        .iter()
        .map(|w| strip_punct_and_space(w).chars().count().max(1))
        .sum::<usize>()
        .max(1);
    let mut acc = 0usize;
    words
        .iter()
        .map(|w| {
            let c = strip_punct_and_space(w).chars().count().max(1);
            let s = t0 + (t1 - t0) * (acc as f64 / total_chars as f64);
            acc += c;
            let e = t0 + (t1 - t0) * (acc as f64 / total_chars as f64);
            WordSpan {
                text: w.clone(),
                start: s,
                end: e.max(s + 0.02),
            }
        })
        .collect()
}

/// Anchor untimed plain lyrics lines to coarse Whisper transcription timestamps & vocal activity.
fn map_plain_lines_to_coarse(
    plain: &[InputLine],
    coarse: &[crate::whisper_align::CoarseLine],
    _env10: &[f32],
    duration: f64,
) -> Vec<InputLine> {
    if coarse.is_empty() || plain.is_empty() {
        let step = duration / plain.len().max(1) as f64;
        return plain
            .iter()
            .enumerate()
            .map(|(i, l)| InputLine {
                timestamp: i as f64 * step,
                text: l.text.clone(),
            })
            .collect();
    }

    let first_vocal = coarse.first().map(|c| c.start).unwrap_or(0.0);
    let last_vocal = coarse.last().map(|c| c.end).unwrap_or(duration);
    let span = (last_vocal - first_vocal).max(1.0);
    let total_chars: usize = plain
        .iter()
        .map(|l| l.text.chars().count().max(1))
        .sum::<usize>()
        .max(1);

    let mut acc = 0usize;
    plain
        .iter()
        .map(|l| {
            let t = first_vocal + span * (acc as f64 / total_chars as f64);
            acc += l.text.chars().count().max(1);
            InputLine {
                timestamp: t,
                text: l.text.clone(),
            }
        })
        .collect()
}
