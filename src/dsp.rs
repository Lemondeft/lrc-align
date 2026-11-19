use anyhow::{Context, Result};
use realfft::RealFftPlanner;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use std::f32::consts::PI;

/// Isolate center-panned vocals via STFT mid-side Wiener masking, keeping 120Hz..7000Hz crisp.
pub fn isolate_center_vocals(left: &[f32], right: &[f32], is_stereo: bool, sample_rate: u32) -> Result<Vec<f32>> {
    if !is_stereo || left.len() != right.len() {
        return Ok(left.to_vec());
    }

    let n = left.len();
    let nperseg = 2048usize;
    let hop = 512usize; // 75% overlap for Hann COLA
    if n < nperseg {
        return Ok(left.iter().zip(right.iter()).map(|(l, r)| 0.5 * (l + r)).collect());
    }

    let mut mid = Vec::with_capacity(n);
    let mut side = Vec::with_capacity(n);
    for i in 0..n {
        mid.push(0.5 * (left[i] + right[i]));
        side.push(0.5 * (left[i] - right[i]));
    }

    let hann: Vec<f32> = (0..nperseg)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / nperseg as f32).cos()))
        .collect();

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(nperseg);
    let c2r = planner.plan_fft_inverse(nperseg);

    let n_bins = nperseg / 2 + 1;
    let freq_weights: Vec<f32> = (0..n_bins)
        .map(|k| {
            let hz = k as f32 * sample_rate as f32 / nperseg as f32;
            if hz < 120.0 {
                0.05
            } else if hz > 7000.0 {
                0.2
            } else {
                1.0
            }
        })
        .collect();

    let mut in_mid = vec![0.0f32; nperseg];
    let mut in_side = vec![0.0f32; nperseg];
    let mut spec_mid = r2c.make_output_vec();
    let mut spec_side = r2c.make_output_vec();
    let mut out_frame = vec![0.0f32; nperseg];
    let mut scratch_fwd = r2c.make_scratch_vec();
    let mut scratch_inv = c2r.make_scratch_vec();

    let mut out = vec![0.0f32; n];
    let mut norm = vec![0.0f32; n];

    let n_frames = (n - nperseg) / hop + 1;
    for frame_idx in 0..n_frames {
        let off = frame_idx * hop;
        for j in 0..nperseg {
            in_mid[j] = mid[off + j] * hann[j];
            in_side[j] = side[off + j] * hann[j];
        }
        r2c.process_with_scratch(&mut in_mid, &mut spec_mid, &mut scratch_fwd)
            .context("FFT forward mid failed")?;
        r2c.process_with_scratch(&mut in_side, &mut spec_side, &mut scratch_fwd)
            .context("FFT forward side failed")?;

        for k in 0..n_bins {
            let mag_m2 = spec_mid[k].norm_sqr();
            let mag_s2 = spec_side[k].norm_sqr();
            let mask = ((mag_m2 - 1.5 * mag_s2).max(0.0)) / (mag_m2 + 1e-10);
            spec_mid[k] *= mask * freq_weights[k];
        }
        // RealFFT expects DC & Nyquist imaginary parts to be 0
        spec_mid[0].im = 0.0;
        spec_mid[n_bins - 1].im = 0.0;

        c2r.process_with_scratch(&mut spec_mid, &mut out_frame, &mut scratch_inv)
            .context("FFT inverse vocal failed")?;

        let inv_scale = 1.0 / nperseg as f32;
        for j in 0..nperseg {
            let w = hann[j];
            out[off + j] += out_frame[j] * inv_scale * w;
            norm[off + j] += w * w;
        }
    }

    for i in 0..n {
        if norm[i] > 1e-6 {
            out[i] /= norm[i];
        }
    }

    Ok(out)
}

/// Resample mono audio to 16,000 Hz using `rubato` FFT resampler, trimming resampler delay.
pub fn resample_to_16k(samples: &[f32], src_sr: u32) -> Result<Vec<f32>> {
    if src_sr == 16000 {
        return Ok(samples.to_vec());
    }
    let nbr_in = samples.len();
    let mut resampler = Fft::<f32>::new(src_sr as usize, 16000, 1024, 1, FixedSync::Input)
        .context("Failed to construct FFT resampler")?;

    let delay = resampler.output_delay();
    let expected_out = ((nbr_in as f64) * 16000.0 / (src_sr as f64)).round() as usize;
    let out_cap = expected_out + delay + resampler.output_frames_max() + 2048;
    let mut outdata = vec![0.0f32; out_cap];

    let in_adapter = InterleavedSlice::new(samples, 1, nbr_in)
        .context("Failed to create input slice adapter")?;
    let mut out_adapter = InterleavedSlice::new_mut(&mut outdata, 1, out_cap)
        .context("Failed to create output slice adapter")?;

    let mut indexing = Indexing::default();
    let mut left = nbr_in;
    let mut next_in = resampler.input_frames_next();

    while left >= next_in {
        let (nin, nout) = resampler
            .process_into_buffer(&in_adapter, &mut out_adapter, Some(&indexing))
            .context("Resampling chunk failed")?;
        indexing.input_offset += nin;
        indexing.output_offset += nout;
        left -= nin;
        next_in = resampler.input_frames_next();
    }

    if left > 0 {
        indexing.partial_len = Some(left);
        let (_nin, nout) = resampler
            .process_into_buffer(&in_adapter, &mut out_adapter, Some(&indexing))
            .context("Resampling tail chunk failed")?;
        indexing.output_offset += nout;
    }

    // Flush resampler delay with zeros
    indexing.partial_len = Some(0);
    while indexing.output_offset < delay + expected_out {
        let (_nin, nout) = resampler
            .process_into_buffer(&in_adapter, &mut out_adapter, Some(&indexing))
            .context("Flushing resampler delay failed")?;
        if nout == 0 {
            break;
        }
        indexing.output_offset += nout;
    }

    let first = delay.min(outdata.len());
    let last = (first + expected_out).min(outdata.len());
    Ok(outdata[first..last].to_vec())
}

/// Second-order IIR biquad section (Direct Form II Transposed).
#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    fn highpass(fs: f64, f0: f64, q: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 + cos_w0) / 2.0) / a0,
            b1: (-(1.0 + cos_w0)) / a0,
            b2: ((1.0 + cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    fn lowpass(fs: f64, f0: f64, q: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * f0 / fs;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);
        let a0 = 1.0 + alpha;
        Self {
            b0: ((1.0 - cos_w0) / 2.0) / a0,
            b1: (1.0 - cos_w0) / a0,
            b2: ((1.0 - cos_w0) / 2.0) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
        }
    }

    fn filter_in_place(&self, buf: &mut [f64]) {
        let mut z1 = 0.0;
        let mut z2 = 0.0;
        for x in buf.iter_mut() {
            let x0 = *x;
            let y = self.b0 * x0 + z1;
            z1 = self.b1 * x0 - self.a1 * y + z2;
            z2 = self.b2 * x0 - self.a2 * y;
            *x = y;
        }
    }
}

/// Zero-phase bandpass (350..3200 Hz) + 10ms RMS vocal-core energy envelope (`env10`).
pub fn compute_vocal_envelope_10ms(vocal_16k: &[f32]) -> Vec<f32> {
    let mut x: Vec<f64> = vocal_16k.iter().map(|&v| v as f64).collect();
    let sections = [
        Biquad::highpass(16000.0, 350.0, 0.7071),
        Biquad::lowpass(16000.0, 3200.0, 0.7071),
    ];
    // Forward + backward pass for zero-phase (equivalent to 4th-order sosfiltfilt)
    for bq in &sections {
        bq.filter_in_place(&mut x);
    }
    x.reverse();
    for bq in &sections {
        bq.filter_in_place(&mut x);
    }
    x.reverse();

    let hop = 160usize; // 10ms at 16 kHz
    let n_hops = x.len() / hop;
    let mut raw_env = Vec::with_capacity(n_hops);
    for h in 0..n_hops {
        let slice = &x[h * hop..(h + 1) * hop];
        let ms: f64 = slice.iter().map(|v| v * v).sum::<f64>() / hop as f64;
        raw_env.push((ms + 1e-12).sqrt() as f32);
    }

    // 5-frame (50ms) centered moving average
    let mut smoothed = vec![0.0f32; n_hops];
    for i in 0..n_hops {
        let a = i.saturating_sub(2);
        let b = (i + 3).min(n_hops);
        let sum: f32 = raw_env[a..b].iter().sum();
        smoothed[i] = sum / (b - a) as f32;
    }
    smoothed
}

/// Find active vocal end inside `[t_start, t_max_end]` so alignment doesn't bleed into silence
/// or across a multi-second instrumental break (> 1.8s gap after vocals have already been sung).
pub fn find_active_end(env10: &[f32], t_start: f64, t_max_end: f64) -> f64 {
    let i0 = ((t_start * 100.0).floor() as usize).min(env10.len());
    let i1 = ((t_max_end * 100.0).ceil() as usize).min(env10.len());
    if i1 <= i0 {
        return t_max_end;
    }
    let seg = &env10[i0..i1];
    let peak_win = 350.min(seg.len());
    let local_peak = seg[..peak_win].iter().copied().fold(0.0f32, f32::max);
    let thresh = (0.14 * local_peak).max(0.014);

    let Some(first_act) = seg.iter().position(|&v| v > thresh) else {
        return (t_start + 3.0).min(t_max_end);
    };

    let mut last_act = first_act;
    let mut quiet_run = 0usize;
    for (k, &v) in seg.iter().enumerate().skip(first_act) {
        if v > thresh {
            last_act = k;
            quiet_run = 0;
        } else {
            quiet_run += 1;
            // A silence >= 1.8s after at least 1.2s of singing marks an instrumental break.
            if last_act >= first_act + 120 && quiet_run >= 180 {
                break;
            }
        }
    }

    let t_end = (i0 + last_act + 1) as f64 / 100.0 + 0.08;
    t_end.min(t_max_end)
}

/// Detect clean intra-line silence valleys (>= `min_gap_ms` below threshold) to snap pauses.
/// Returns active vocal islands `(start_sec, end_sec)` inside `[t_start, t_end]`.
pub fn detect_vocal_islands(
    env10: &[f32],
    t_start: f64,
    t_end: f64,
    min_gap_ms: usize,
) -> Vec<(f64, f64)> {
    let i0 = ((t_start * 100.0).floor() as usize).min(env10.len());
    let i1 = ((t_end * 100.0).ceil() as usize).min(env10.len());
    if i1 <= i0 + 10 {
        return vec![(t_start, t_end)];
    }
    let seg = &env10[i0..i1];
    let peak_win = 350.min(seg.len());
    let local_peak = seg[..peak_win].iter().copied().fold(0.0f32, f32::max);
    // Only split on clear low-energy pauses (quiet verses / dramatic breaks)
    let thresh = (0.12 * local_peak).clamp(0.014, 0.028);
    let min_gap_frames = min_gap_ms / 10;

    let active_indices: Vec<usize> = seg
        .iter()
        .enumerate()
        .filter_map(|(k, &v)| if v > thresh { Some(k) } else { None })
        .collect();
    if active_indices.is_empty() {
        return vec![(t_start, t_end)];
    }

    let mut gaps = Vec::new();
    let mut in_gap = false;
    let mut gap_start = 0usize;
    for (k, &v) in seg.iter().enumerate() {
        if v <= thresh && !in_gap {
            in_gap = true;
            gap_start = k;
        } else if v > thresh && in_gap {
            in_gap = false;
            if k - gap_start >= min_gap_frames {
                gaps.push((gap_start, k));
            }
        }
    }

    let mut islands = Vec::new();
    let mut cur_start = active_indices[0];
    for (ga, gb) in gaps {
        if ga > cur_start + 5 {
            if let Some(last_act) = seg[cur_start..ga].iter().rposition(|&v| v > thresh) {
                let s = (t_start + cur_start as f64 / 100.0).max(t_start);
                let e = (t_start + (cur_start + last_act + 1) as f64 / 100.0 + 0.04).min(t_end);
                if e > s + 0.05 {
                    islands.push((s, e));
                }
            }
        }
        if let Some(next_act) = seg[gb..].iter().position(|&v| v > thresh) {
            cur_start = gb + next_act;
        } else {
            cur_start = seg.len();
            break;
        }
    }

    if cur_start < seg.len() {
        if let Some(last_act) = seg[cur_start..].iter().rposition(|&v| v > thresh) {
            let s = (t_start + cur_start as f64 / 100.0).max(t_start);
            let e = (t_start + (cur_start + last_act + 1) as f64 / 100.0 + 0.06).min(t_end);
            if e > s + 0.05 {
                islands.push((s, e));
            }
        }
    }

    if islands.is_empty() {
        vec![(t_start, t_end)]
    } else {
        islands[0].0 = (islands[0].0 - 0.04).max(t_start);
        islands
    }
}
