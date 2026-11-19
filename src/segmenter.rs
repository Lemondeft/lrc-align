use anyhow::{Result, anyhow};
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter as LinderaSegmenter;
use regex::Regex;
use std::borrow::Cow;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct InputLine {
    pub timestamp: f64,
    pub text: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WordSpan {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AlignedLine {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub words: Vec<WordSpan>,
}

static LINDERA: OnceLock<LinderaSegmenter> = OnceLock::new();

fn get_lindera() -> Result<&'static LinderaSegmenter> {
    if let Some(seg) = LINDERA.get() {
        return Ok(seg);
    }
    let dict = load_dictionary("embedded://ipadic")
        .map_err(|e| anyhow!("Failed to load embedded IPADIC dictionary: {e}"))?;
    let seg = LinderaSegmenter::new(Mode::Normal, dict, None);
    let _ = LINDERA.set(seg);
    Ok(LINDERA.get().unwrap())
}

pub fn has_japanese(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{3040}'..='\u{309F}' // Hiragana
            | '\u{30A0}'..='\u{30FF}' // Katakana
            | '\u{4E00}'..='\u{9FFF}' // CJK Unified Ideographs
            | '\u{3400}'..='\u{4DBF}'
        )
    })
}

pub fn is_ignorable_punct_or_space(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '、' | '。' | '！' | '？' | '…' | '・' | ',' | '.' | '!' | '?' | '~' | '〜' | '「' | '」' | '(' | ')' | '（' | '）' | '"' | '\'' | '♪'
        )
}

pub fn strip_punct_and_space(s: &str) -> String {
    s.chars().filter(|&c| !is_ignorable_punct_or_space(c)).collect()
}

/// Parse LRC (stripping inline `<mm:ss.xx>` tags if present). Returns `(is_synced, lines)`.
pub fn parse_lrc(raw: &str) -> (bool, Vec<InputLine>) {
    let line_re = Regex::new(r"^\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\](.*)$").unwrap();
    let inline_re = Regex::new(r"<\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?>").unwrap();
    let mut lines = Vec::new();
    let mut plain = Vec::new();

    for raw_line in raw.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(caps) = line_re.captures(trimmed) {
            let mins: f64 = caps[1].parse().unwrap_or(0.0);
            let secs_str = caps[2].replace(':', ".");
            let secs: f64 = secs_str.parse().unwrap_or(0.0);
            let clean = inline_re.replace_all(&caps[3], "").trim().to_string();
            if !clean.is_empty() {
                lines.push(InputLine {
                    timestamp: mins * 60.0 + secs,
                    text: clean,
                });
            }
        } else if !trimmed.starts_with('[') {
            let clean = inline_re.replace_all(trimmed, "").trim().to_string();
            if !clean.is_empty() {
                plain.push(InputLine {
                    timestamp: -1.0,
                    text: clean,
                });
            }
        }
    }

    if !lines.is_empty() {
        lines.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        (true, lines)
    } else {
        (false, plain)
    }
}

/// Segment a lyric line into natural karaoke words, preserving every character, punctuation, and space.
pub fn segment_line_words(line: &str) -> Vec<String> {
    if line.is_empty() {
        return Vec::new();
    }
    if !has_japanese(line) {
        return segment_whitespace_words(line);
    }

    let Ok(segmenter) = get_lindera() else {
        return segment_whitespace_words(line);
    };
    let Ok(mut tokens) = segmenter.segment(Cow::Borrowed(line)) else {
        return segment_whitespace_words(line);
    };

    // Extract `(surface, pos0, pos1)` for every Lindera token.
    // Note: Lindera skips ASCII whitespace in some modes, so track byte positions to preserve spaces!
    let mut raw_parts: Vec<(String, String, String)> = Vec::new();
    let mut cursor = 0usize;
    for tok in tokens.iter_mut() {
        let surf = tok.surface.as_ref().to_string();
        let details = tok.details();
        let pos0 = details.first().map(|s| s.to_string()).unwrap_or_default();
        let pos1 = details.get(1).map(|s| s.to_string()).unwrap_or_default();

        if let Some(found_rel) = line[cursor..].find(&surf) {
            if found_rel > 0 {
                let gap = &line[cursor..cursor + found_rel];
                raw_parts.push((gap.to_string(), "空白".into(), "*".into()));
            }
            cursor += found_rel + surf.len();
        }
        raw_parts.push((surf, pos0, pos1));
    }
    if cursor < line.len() {
        raw_parts.push((line[cursor..].to_string(), "空白".into(), "*".into()));
    }

    let mut words: Vec<String> = Vec::new();
    let mut prev_pos0 = String::new();

    for (surf, pos0, pos1) in raw_parts {
        let is_punct_or_space = pos0 == "記号" || pos0 == "空白" || surf.chars().all(is_ignorable_punct_or_space);
        if is_punct_or_space {
            if let Some(last) = words.last_mut() {
                last.push_str(&surf);
            } else {
                words.push(surf);
            }
            continue;
        }

        // Merge suffixes, auxiliary verbs, conjunctive particles (て/で), non-independent verbs (てる/いる),
        // and adjectival/verb endings onto the preceding word for natural Japanese karaoke units.
        let attach_to_prev = !words.is_empty()
            && (pos0 == "助動詞"
                || (pos0 == "名詞" && pos1 == "接尾")
                || (pos0 == "名詞" && pos1 == "非自立" && matches!(surf.as_str(), "ん" | "の"))
                || (pos0 == "動詞" && pos1 == "接尾")
                || (pos0 == "動詞" && pos1 == "非自立" && matches!(surf.as_str(), "てる" | "て" | "てた" | "いる" | "い"))
                || (pos0 == "助詞" && pos1 == "接続助詞" && matches!(surf.as_str(), "て" | "で" | "ちゃ" | "じゃ"))
                || (pos0 == "助詞" && surf == "ね" && prev_pos0 == "感動詞")
                || (prev_pos0 == "接頭詞"));

        if attach_to_prev {
            words.last_mut().unwrap().push_str(&surf);
        } else {
            words.push(surf);
        }
        prev_pos0 = pos0;
    }

    // Merge stray leading punctuation onto first word
    if words.len() >= 2 && words[0].chars().all(is_ignorable_punct_or_space) {
        let lead = words.remove(0);
        words[0] = format!("{lead}{}", words[0]);
    }

    // Fall back to whole line if tokenizer dropped or altered characters
    if words.concat() != line {
        return vec![line.to_string()];
    }
    words
}

fn segment_whitespace_words(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut seen_non_space = false;
    for ch in line.chars() {
        if ch.is_whitespace() {
            cur.push(ch);
            if seen_non_space {
                out.push(std::mem::take(&mut cur));
                seen_non_space = false;
            }
        } else {
            if !seen_non_space && !cur.is_empty() {
            }
            cur.push(ch);
            seen_non_space = true;
        }
    }
    if !cur.is_empty() {
        if out.is_empty() || seen_non_space {
            out.push(cur);
        } else if let Some(last) = out.last_mut() {
            last.push_str(&cur);
        }
    }
    out
}

pub fn format_timestamp(sec: f64) -> String {
    let clamped = sec.max(0.0);
    let total_cs = (clamped * 100.0).round() as u64;
    let mins = total_cs / 6000;
    let rem_cs = total_cs % 6000;
    let secs = rem_cs / 100;
    let cs = rem_cs % 100;
    format!("{mins:02}:{secs:02}.{cs:02}")
}

pub fn format_enhanced_lrc(
    lines: &[AlignedLine],
    title: Option<&str>,
    artist: Option<&str>,
    album: Option<&str>,
) -> String {
    let mut out = Vec::new();
    if let Some(t) = title.filter(|s| !s.is_empty()) {
        out.push(format!("[ti:{t}]"));
    }
    if let Some(a) = artist.filter(|s| !s.is_empty()) {
        out.push(format!("[ar:{a}]"));
    }
    if let Some(al) = album.filter(|s| !s.is_empty()) {
        out.push(format!("[al:{al}]"));
    }
    if !out.is_empty() {
        out.push(String::new());
    }

    for line in lines {
        if line.words.is_empty() {
            continue;
        }
        let mut s = format!("[{}]", format_timestamp(line.words[0].start));
        for (idx, w) in line.words.iter().enumerate() {
            s.push_str(&format!("<{}>{}", format_timestamp(w.start), w.text));
            if let Some(next_w) = line.words.get(idx + 1) {
                if next_w.start - w.end >= 0.14 {
                    s.push_str(&format!("<{}>", format_timestamp(w.end)));
                }
            } else {
                s.push_str(&format!("<{}>", format_timestamp(w.end)));
            }
        }
        out.push(s);
    }
    out.join("\n") + "\n"
}

pub fn format_wlrc(
    lines: &[AlignedLine],
    title: Option<&str>,
    artist: Option<&str>,
    album: Option<&str>,
) -> String {
    let mut out = Vec::new();
    if let Some(t) = title.filter(|s| !s.is_empty()) {
        out.push(format!("[ti:{t}]"));
    }
    if let Some(a) = artist.filter(|s| !s.is_empty()) {
        out.push(format!("[ar:{a}]"));
    }
    if let Some(al) = album.filter(|s| !s.is_empty()) {
        out.push(format!("[al:{al}]"));
    }
    if !out.is_empty() {
        out.push(String::new());
    }

    for line in lines {
        for w in &line.words {
            let trimmed = w.text.trim();
            if !trimmed.is_empty() {
                out.push(format!("[{}]{trimmed}", format_timestamp(w.start)));
            }
        }
    }
    out.join("\n") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lrc_strips_inline_tags() {
        let raw = "[ti:言って。]\n[00:02.13]<00:02.13>言って<00:02.77>\n[00:23.84]<00:23.84>あのね、<00:26.45>私";
        let (synced, lines) = parse_lrc(raw);
        assert!(synced);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "言って");
        assert_eq!(lines[1].text, "あのね、私");
    }

    #[test]
    fn japanese_segmentation_preserves_exact_text_and_spaces() {
        let samples = [
            "言って",
            "あのね、私実は気付いてるの",
            "盲目的に盲動的に妄想的に生きて",
            "口に出して 声に出して",
            "もっと、ちゃんと 言って",
            "And then we run away, fast!",
        ];
        for s in samples {
            let words = segment_line_words(s);
            assert!(!words.is_empty());
            assert_eq!(words.concat(), s);
        }
    }

    #[test]
    fn formats_enhanced_lrc_and_wlrc() {
        let lines = vec![AlignedLine {
            start: 2.13,
            end: 2.77,
            text: "言って".into(),
            words: vec![WordSpan {
                text: "言って".into(),
                start: 2.13,
                end: 2.77,
            }],
        }];
        let elrc = format_enhanced_lrc(&lines, Some("言って。"), Some("Yorushika"), None);
        assert!(elrc.contains("[ti:言って。]"));
        assert!(elrc.contains("[00:02.13]<00:02.13>言って<00:02.77>"));
        let wlrc = format_wlrc(&lines, None, None, None);
        assert_eq!(wlrc.trim(), "[00:02.13]言って");
    }
}
