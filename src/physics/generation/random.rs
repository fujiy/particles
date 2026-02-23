use noise::{NoiseFn, OpenSimplex};
use std::sync::OnceLock;

use super::params::{
    FBM_ATTENUATION_SCALE, TERRAIN_FBM_GAIN, TERRAIN_FBM_LACUNARITY, TERRAIN_FBM_OCTAVES,
    TERRAIN_GENERATOR_VERSION, WORLD_SEED,
};

pub fn terrain_noise_sample(input: [f64; 2]) -> f64 {
    terrain_noise().get(input)
}

pub fn generator_seed() -> u32 {
    WORLD_SEED ^ TERRAIN_GENERATOR_VERSION.wrapping_mul(0x9E37_79B9)
}

pub fn fbm_1d_filtered(
    x: f64,
    base_freq: f64,
    octaves: u32,
    lacunarity: f64,
    gain: f64,
    footprint_cells: f64,
    channel_offset: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = base_freq;
    let mut amp_sum = 0.0;

    for octave in 0..octaves {
        let low_pass = octave_low_pass_weight(freq, footprint_cells);
        if low_pass < 1e-3 && octave > 1 {
            break;
        }
        let value = terrain_noise_sample([x * freq, channel_offset + octave as f64 * 13.37]);
        let weighted_amp = amp * low_pass;
        sum += value * weighted_amp;
        amp_sum += weighted_amp;
        amp *= gain;
        freq *= lacunarity;
    }

    if amp_sum <= 1e-6 { 0.0 } else { sum / amp_sum }
}

pub fn default_surface_fbm(x_cells: f64, footprint_cells: f64, base_freq: f64) -> f64 {
    fbm_1d_filtered(
        x_cells,
        base_freq,
        TERRAIN_FBM_OCTAVES,
        TERRAIN_FBM_LACUNARITY,
        TERRAIN_FBM_GAIN,
        footprint_cells,
        TERRAIN_GENERATOR_VERSION as f64,
    )
}

fn terrain_noise() -> &'static OpenSimplex {
    static NOISE: OnceLock<OpenSimplex> = OnceLock::new();
    NOISE.get_or_init(|| OpenSimplex::new(generator_seed()))
}

fn octave_low_pass_weight(freq: f64, footprint_cells: f64) -> f64 {
    let footprint_freq = (freq * footprint_cells).max(0.0);
    1.0 / (1.0 + (footprint_freq * FBM_ATTENUATION_SCALE).powi(2))
}
