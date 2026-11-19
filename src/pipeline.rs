use crate::audio::{AudioMetadata, decode_audio, inspect_metadata};
use crate::dsp::{
    compute_vocal_envelope_10ms, detect_vocal_islands, find_active_end, isolate_center_vocals,
    resample_to_16k,
};
use crate::segmenter::{
    AlignedLine, InputLine, WordSpan, format_enhanced_lrc, format_wlrc, parse_lrc,
    segment_line_words, shift_aligned_lines, split_role_prefix, strip_punct_and_space,
};
use crate::whisper_align::{AlignedUnit, AlignmentResult, DevicePreference, WhisperAligner, WhisperSize};
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct AlignOptions {
    pub model_size: WhisperSize,
    pub device_pref: DevicePreference,
    pub vocal_isolation: bool,
    pub pause_snapping: bool,
    pub language: Option<String>,
    pub raw_lyrics_override: Option<String>,
    pub offset_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AlignOutput {
    pub metadata: AudioMetadata,
    pub device_used: String,
    pub language: String,
    pub confidence: f32,
    pub enhanced_lrc: String,
    pub wlrc: String,
    pub lines: Vec<AlignedLine>,
    pub elapsed_secs: f64,
}

#[derive(Default)]
pub struct ModelCache {
    inner: Mutex<Option<(WhisperSize, DevicePreference, String, WhisperAligner)>>,
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

    let raw_mono = if decoded.is_stereo {
        decoded
            .left
            .iter()
            .zip(decoded.right.iter())
            .map(|(l, r)| 0.5 * (l + r))
            .collect()
    } else {
        decoded.left.clone()
    };

    let mono_vocal = if opts.vocal_isolation && decoded.is_stereo {
        on_progress(0.22, "Isolating center-panned vocals (STFT Wiener mask)...".into());
        isolate_center_vocals(
            &decoded.left,
            &decoded.right,
            decoded.is_stereo,
            decoded.sample_rate,
        )?
    } else {
        raw_mono.clone()
    };

    on_progress(0.30, "Resampling to 16 kHz & extracting 10 ms vocal-band envelope...".into());
    let vocal_16k = resample_to_16k(&mono_vocal, decoded.sample_rate)?;
    let raw_16k_opt = if opts.vocal_isolation && decoded.is_stereo {
        Some(resample_to_16k(&raw_mono, decoded.sample_rate)?)
    } else {
        None
    };
    let raw_ref = raw_16k_opt.as_deref().unwrap_or(&vocal_16k);
    let env10 = compute_vocal_envelope_10ms(&vocal_16k);

    let mut cached_aligner: Option<(String, WhisperAligner)> = None;
    if let Some(ref c) = cache {
        let mut guard = c.inner.lock().unwrap();
        if let Some((sz, pref, _, _)) = guard.as_ref() {
            if *sz == opts.model_size && *pref == opts.device_pref {
                cached_aligner = guard.take().map(|(_, _, dev, a)| (dev, a));
            } else {
                *guard = None;
            }
        }
    }

    let (mut aligner, device_used) = match cached_aligner {
        Some((dev_label, a)) => {
            on_progress(
                0.45,
                format!(
                    "Reusing cached {} model on {}...",
                    opts.model_size.display_name(),
                    dev_label
                ),
            );
            (a, dev_label)
        }
        None => {
            on_progress(0.35, format!("Loading {}...", opts.model_size.display_name()));
            WhisperAligner::load(opts.model_size, opts.device_pref, |msg| {
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

    let lyrics_hint: String = input_lines
        .iter()
        .map(|l| split_role_prefix(&l.text).1)
        .collect::<Vec<_>>()
        .join("\n");
    let sample_for_lang = best_30s_vocal_window(&env10, raw_ref);
    let lang = if let Some(l) = opts.language.as_deref().filter(|s| !s.is_empty() && *s != "auto") {
        l.to_string()
    } else {
        aligner
            .detect_language_with_hint(sample_for_lang, &lyrics_hint)
            .unwrap_or_else(|_| "en".to_string())
    };
    on_progress(0.48, format!("Using language '{lang}' on {device_used}"));

    if input_lines.is_empty() {
        on_progress(0.52, "No lyrics provided: running automatic Whisper transcription...".into());
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
            "Untimed plain lyrics detected: locating coarse line timestamps via Whisper...".into(),
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

        let (role_prefix, body_text) = split_role_prefix(&line.text);
        let words = segment_line_words(body_text);
        if words.is_empty() {
            continue;
        }

        let prev_end = aligned_lines.last().map(|l: &AlignedLine| l.end).unwrap_or(0.0);
        let t_s = line.timestamp.max(0.0).min(duration);
        let t_next = if i + 1 < input_lines.len() {
            input_lines[i + 1].timestamp
        } else {
            duration
        };
        let pre_roll = if is_synced {
            0.0
        } else {
            (t_s - prev_end).clamp(0.0, 0.30)
        };
        let ctx_s = (t_s - pre_roll).max(prev_end);
        let unit_count = words.len().max(body_text.chars().count() / 3);
        let max_win = if t_next - t_s > 9.0 {
            (2.2 + unit_count as f64 * 0.58).clamp(4.0, 8.5)
        } else {
            15.0
        };
        let t_max = (t_next - 0.02).min(t_s + max_win).max(t_s + 0.3);
        let t_end_active = find_active_end(&env10, t_s, t_max).max(t_s + 0.25);

        let islands = if opts.pause_snapping {
            detect_vocal_islands(&env10, ctx_s, t_end_active, 250)
        } else {
            vec![(ctx_s, t_end_active)]
        };

        let mut word_spans = align_line_with_islands(
            &mut aligner,
            &vocal_16k,
            raw_16k_opt.as_deref(),
            &env10,
            ctx_s,
            t_s,
            t_end_active,
            body_text,
            &words,
            &islands,
            &lang,
            opts.pause_snapping,
        )?;

        // Clamp within [ctx_s, t_max] and enforce strictly monotonic order.
        let line_anchor_floor = if is_synced { t_s.max(prev_end) } else { ctx_s };
        let mut cursor = line_anchor_floor;
        for (wi, w) in word_spans.iter_mut().enumerate() {
            w.start = w.start.max(cursor).min(t_max - 0.02);
            if wi == 0 && is_synced && w.start - line_anchor_floor > 0.35 {
                // Keep synced line starts tight unless Whisper/pre-speech detected only a small silence offset.
                w.start = line_anchor_floor;
            }
            w.end = w.end.max(w.start + 0.02).min(t_max);
            cursor = w.end;
        }

        if !role_prefix.is_empty() {
            if let Some(first) = word_spans.first_mut() {
                first.text = format!("{role_prefix}{}", first.text);
            }
        }

        let line_start = word_spans.first().map(|w| w.start).unwrap_or(t_s);
        let line_end = word_spans.last().map(|w| w.end).unwrap_or(t_end_active);
        let line_conf = if word_spans.is_empty() {
            0.0
        } else {
            word_spans.iter().map(|w| w.confidence).sum::<f32>() / word_spans.len() as f32
        };
        aligned_lines.push(AlignedLine {
            start: line_start,
            end: line_end,
            text: line.text.clone(),
            confidence: line_conf,
            words: word_spans,
        });
    }

    if let Some(ref c) = cache {
        let mut guard = c.inner.lock().unwrap();
        *guard = Some((opts.model_size, opts.device_pref, device_used.clone(), aligner));
    }

    let aligned_lines = shift_aligned_lines(&aligned_lines, opts.offset_ms);
    let total_words: usize = aligned_lines.iter().map(|l| l.words.len()).sum();
    let overall_confidence = if total_words == 0 {
        0.0
    } else {
        aligned_lines
            .iter()
            .flat_map(|l| l.words.iter())
            .map(|w| w.confidence)
            .sum::<f32>()
            / total_words as f32
    };

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
            "Done! Aligned {} lines in {:.2}s ({:.0}% confidence).",
            aligned_lines.len(),
            elapsed,
            overall_confidence * 100.0
        ),
    );

    Ok(AlignOutput {
        metadata,
        device_used,
        language: lang,
        confidence: overall_confidence,
        enhanced_lrc,
        wlrc,
        lines: aligned_lines,
        elapsed_secs: elapsed,
    })
}

fn best_30s_vocal_window<'a>(env10: &[f32], audio_16k: &'a [f32]) -> &'a [f32] {
    let win_samples = 30 * 16000;
    if audio_16k.len() <= win_samples || env10.len() <= 3000 {
        return audio_16k;
    }
    let env_win = 3000usize;
    let mut cur_sum: f32 = env10[..env_win].iter().sum();
    let mut best_sum = cur_sum;
    let mut best_idx = 0usize;
    for i in env_win..env10.len() {
        cur_sum += env10[i] - env10[i - env_win];
        if cur_sum > best_sum {
            best_sum = cur_sum;
            best_idx = i + 1 - env_win;
        }
    }
    let start_sample = (best_idx * 160).min(audio_16k.len().saturating_sub(win_samples));
    let end_sample = (start_sample + win_samples).min(audio_16k.len());
    &audio_16k[start_sample..end_sample]
}

fn align_segment_best_source(
    aligner: &mut WhisperAligner,
    vocal_16k: &[f32],
    raw_16k: Option<&[f32]>,
    t0: f64,
    t1: f64,
    clean: &str,
    lang: &str,
) -> Result<AlignmentResult> {
    let seg_vocal = slice_16k(vocal_16k, t0, t1);
    let best = aligner.align_segment(seg_vocal, clean, lang)?;
    if best.confidence < 0.55 {
        if let Some(raw) = raw_16k {
            let seg_raw = slice_16k(raw, t0, t1);
            if let Ok(raw_res) = aligner.align_segment(seg_raw, clean, lang) {
                if raw_res.confidence > best.confidence {
                    return Ok(raw_res);
                }
            }
        }
    }
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
fn align_line_with_islands(
    aligner: &mut WhisperAligner,
    vocal_16k: &[f32],
    raw_16k: Option<&[f32]>,
    env10: &[f32],
    ctx_s: f64,
    _t_s: f64,
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
                    confidence: 0.70,
                });
            } else {
                let joined: String = clause_words.iter().map(String::as_str).collect();
                let clean = clean_for_whisper(&joined);
                let res = align_segment_best_source(
                    aligner, vocal_16k, raw_16k, *sub_s, *sub_e, &clean, lang,
                )?;
                let mut spans = map_units_to_words(&res.units, clause_words, *sub_s, *sub_e, res.confidence);
                if let Some(last) = spans.last_mut() {
                    last.end = last.end.min(*sub_e);
                }
                out.extend(spans);
            }
        }
        return Ok(out);
    }

    let first_s = islands.first().map(|&(s, _)| s).unwrap_or(ctx_s);
    let clean = clean_for_whisper(orig_line);
    let res = align_segment_best_source(
        aligner, vocal_16k, raw_16k, first_s, t_end_active, &clean, lang,
    )?;
    let mut spans = map_units_to_words(&res.units, words, first_s, t_end_active, res.confidence);

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
    fallback_conf: f32,
) -> Vec<WordSpan> {
    let mut char_times: Vec<(char, f64, f64, f32)> = Vec::new();
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
                u.confidence,
            ));
        }
    }

    let total_clean_chars: usize = words.iter().map(|w| strip_punct_and_space(w).chars().count()).sum();
    if char_times.is_empty() || char_times.len() != total_clean_chars {
        // Fallback: proportional interpolation across [offset_sec, fallback_end] if tokenizer altered chars (e.g. NFKC/numbers)
        return proportional_map(units, words, offset_sec, fallback_end, fallback_conf * 0.75);
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
                confidence: fallback_conf * 0.7,
            });
            continue;
        }
        let s = char_times[ci].1;
        let e = char_times[ci + count - 1].2;
        let conf_sum: f32 = char_times[ci..ci + count].iter().map(|x| x.3).sum();
        let conf = conf_sum / count as f32;
        ci += count;
        spans.push(WordSpan {
            text: w.clone(),
            start: s,
            end: e.max(s + 0.02),
            confidence: conf,
        });
    }
    spans
}

fn proportional_map(
    units: &[AlignedUnit],
    words: &[String],
    offset_sec: f64,
    fallback_end: f64,
    conf: f32,
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
                confidence: conf,
            }
        })
        .collect()
}

/// Anchor untimed plain lyrics lines to coarse Whisper transcription timestamps,
/// skipping instrumental gaps by matching normalized character bigrams monotonically.
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

    // Build a continuous active-speech character stream `(char, timestamp)` from coarse segments.
    let mut stream_chars: Vec<char> = Vec::new();
    let mut stream_times: Vec<f64> = Vec::new();
    for seg in coarse {
        let clean: Vec<char> = strip_punct_and_space(&seg.text)
            .to_lowercase()
            .chars()
            .collect();
        if clean.is_empty() {
            continue;
        }
        let span = (seg.end - seg.start).max(0.05);
        for (idx, ch) in clean.iter().enumerate() {
            stream_chars.push(*ch);
            stream_times.push(seg.start + span * (idx as f64 / clean.len() as f64));
        }
    }

    if stream_chars.is_empty() {
        let first_vocal = coarse.first().map(|c| c.start).unwrap_or(0.0);
        let last_vocal = coarse.last().map(|c| c.end).unwrap_or(duration);
        let span = (last_vocal - first_vocal).max(1.0);
        return plain
            .iter()
            .enumerate()
            .map(|(i, l)| InputLine {
                timestamp: first_vocal + span * (i as f64 / plain.len().max(1) as f64),
                text: l.text.clone(),
            })
            .collect();
    }

    let n_stream = stream_chars.len();
    let line_norms: Vec<Vec<char>> = plain
        .iter()
        .map(|l| strip_punct_and_space(&l.text).to_lowercase().chars().collect())
        .collect();
    let total_plain_chars: usize = line_norms.iter().map(|v| v.len().max(1)).sum::<usize>().max(1);

    let mut anchors: Vec<Option<usize>> = vec![None; plain.len()];
    let mut cursor = 0usize;
    let mut plain_acc = 0usize;

    for (idx, chars) in line_norms.iter().enumerate() {
        let expected_pos = (plain_acc * n_stream) / total_plain_chars;
        plain_acc += chars.len().max(1);

        if chars.len() >= 3 {
            let win_len = chars.len().min(n_stream);
            let search_start = cursor.min(n_stream.saturating_sub(win_len));
            let search_end = (expected_pos + (n_stream / 3).max(win_len * 4))
                .min(n_stream.saturating_sub(win_len));

            let mut best_pos = None;
            let mut best_score = 0.35_f64;
            if search_start <= search_end {
                for pos in search_start..=search_end {
                    let s = char_bigram_similarity(chars, &stream_chars[pos..pos + win_len]);
                    if s > best_score {
                        best_score = s;
                        best_pos = Some(pos);
                    }
                }
            }
            if let Some(p) = best_pos {
                anchors[idx] = Some(p);
                cursor = (p + chars.len() / 2).min(n_stream.saturating_sub(1));
            }
        }
    }

    // Fill unanchored lines by interpolating stream positions between nearest anchors.
    let mut resolved_pos = vec![0usize; plain.len()];
    let mut last_anchor: Option<(usize, usize)> = None;
    for i in 0..plain.len() {
        if let Some(p) = anchors[i] {
            let prev_i = last_anchor.map(|(li, _)| li + 1).unwrap_or(0);
            let prev_p = last_anchor.map(|(_, lp)| lp).unwrap_or(0);
            fill_stream_positions(&mut resolved_pos, &line_norms, prev_i, i, prev_p, p);
            resolved_pos[i] = p;
            last_anchor = Some((i, p));
        }
    }
    let tail_start = last_anchor.map(|(li, _)| li + 1).unwrap_or(0);
    let tail_p = last_anchor.map(|(_, lp)| lp).unwrap_or(0);
    fill_stream_positions(
        &mut resolved_pos,
        &line_norms,
        tail_start,
        plain.len(),
        tail_p,
        n_stream.saturating_sub(1),
    );

    let mut prev_t = 0.0_f64;
    plain
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let p = resolved_pos[i].min(stream_times.len() - 1);
            let t = stream_times[p].max(prev_t);
            prev_t = t + 0.05;
            InputLine {
                timestamp: t.min(duration),
                text: l.text.clone(),
            }
        })
        .collect()
}

fn fill_stream_positions(
    out: &mut [usize],
    norms: &[Vec<char>],
    start_idx: usize,
    end_idx: usize,
    start_pos: usize,
    end_pos: usize,
) {
    if start_idx >= end_idx {
        return;
    }
    let span = end_pos.saturating_sub(start_pos);
    let weight_sum: usize = norms[start_idx..end_idx]
        .iter()
        .map(|v| v.len().max(1))
        .sum::<usize>()
        .max(1);
    let mut acc = 0usize;
    for i in start_idx..end_idx {
        out[i] = start_pos + (span * acc) / weight_sum;
        acc += norms[i].len().max(1);
    }
}

fn char_bigram_similarity(a: &[char], b: &[char]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let mut hits = 0usize;
    for pair in a.windows(2) {
        if b.windows(2).any(|bp| bp == pair) {
            hits += 1;
        }
    }
    (2.0 * hits as f64) / ((a.len() - 1 + b.len() - 1) as f64)
}

#[cfg(test)]
mod tests {
    use super::map_plain_lines_to_coarse;
    use crate::segmenter::InputLine;
    use crate::whisper_align::CoarseLine;

    #[test]
    fn plain_lyrics_anchor_across_instrumental_gap() {
        let plain = vec![
            InputLine { timestamp: 0.0, text: "High dive into frozen waves".into() },
            InputLine { timestamp: 0.0, text: "Where the past comes back to life".into() },
            InputLine { timestamp: 0.0, text: "Walk on through a red parade".into() },
        ];
        let coarse = vec![
            CoarseLine { start: 10.0, end: 13.0, text: "High dive into frozen waves".into() },
            CoarseLine { start: 13.5, end: 16.5, text: "Where the past comes back to life".into() },
            // 60-second instrumental break before verse 2
            CoarseLine { start: 76.0, end: 79.0, text: "Walk on through a red parade".into() },
        ];
        let mapped = map_plain_lines_to_coarse(&plain, &coarse, &[], 120.0);
        assert!((mapped[0].timestamp - 10.0).abs() < 0.5);
        assert!((mapped[1].timestamp - 13.5).abs() < 0.5);
        assert!((mapped[2].timestamp - 76.0).abs() < 0.5);
    }
}
