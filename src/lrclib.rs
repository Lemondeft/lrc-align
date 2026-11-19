use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LrcLibRecord {
    synced_lyrics: Option<String>,
    plain_lyrics: Option<String>,
    track_name: Option<String>,
    artist_name: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct LrcLibResult {
    pub lyrics: String,
    pub synced: bool,
    pub source: String,
}

pub async fn fetch_from_lrclib(
    title: &str,
    artist: Option<&str>,
    album: Option<&str>,
    duration_secs: Option<f64>,
) -> Result<LrcLibResult> {
    let client = reqwest::Client::builder()
        .user_agent("lrc-align/0.1.0")
        .build()?;

    if let Some(artist_name) = artist.filter(|s| !s.trim().is_empty()) {
        let mut req = client
            .get("https://lrclib.net/api/get")
            .query(&[("track_name", title), ("artist_name", artist_name)]);
        if let Some(al) = album.filter(|s| !s.trim().is_empty()) {
            req = req.query(&[("album_name", al)]);
        }
        if let Some(dur) = duration_secs.filter(|&d| d > 1.0) {
            req = req.query(&[("duration", &format!("{}", dur.round() as u64))]);
        }
        if let Ok(resp) = req.send().await {
            if resp.status().is_success() {
                if let Ok(rec) = resp.json::<LrcLibRecord>().await {
                    if let Some(synced) = rec.synced_lyrics.filter(|s| !s.trim().is_empty()) {
                        return Ok(LrcLibResult {
                            lyrics: synced,
                            synced: true,
                            source: "LRCLIB exact match".into(),
                        });
                    }
                    if let Some(plain) = rec.plain_lyrics.filter(|s| !s.trim().is_empty()) {
                        return Ok(LrcLibResult {
                            lyrics: plain,
                            synced: false,
                            source: "LRCLIB exact match (plain)".into(),
                        });
                    }
                }
            }
        }
    }

    let query = match artist.filter(|s| !s.trim().is_empty()) {
        Some(a) => format!("{a} {title}"),
        None => title.to_string(),
    };
    let records: Vec<LrcLibRecord> = client
        .get("https://lrclib.net/api/search")
        .query(&[("q", query.as_str())])
        .send()
        .await
        .context("Failed to contact LRCLIB")?
        .error_for_status()
        .context("LRCLIB search returned HTTP error")?
        .json()
        .await
        .context("Failed to parse LRCLIB response")?;

    for rec in &records {
        if let Some(synced) = rec.synced_lyrics.as_ref().filter(|s| !s.trim().is_empty()) {
            let label = format!(
                "{} - {}",
                rec.artist_name.as_deref().unwrap_or(""),
                rec.track_name.as_deref().unwrap_or("")
            );
            return Ok(LrcLibResult {
                lyrics: synced.clone(),
                synced: true,
                source: format!("LRCLIB search ({label})"),
            });
        }
    }
    for rec in &records {
        if let Some(plain) = rec.plain_lyrics.as_ref().filter(|s| !s.trim().is_empty()) {
            return Ok(LrcLibResult {
                lyrics: plain.clone(),
                synced: false,
                source: "LRCLIB search (plain)".into(),
            });
        }
    }

    Err(anyhow!("No lyrics found on LRCLIB for \"{title}\""))
}
