// Terrain generation schema version stored with save data.
pub const TERRAIN_GENERATOR_VERSION: u32 = 1;
// Global world seed used as the base entropy source.
pub const WORLD_SEED: u32 = 13_370;

// Base frequency of the surface height noise in world-cell units.
pub const HEIGHT_NOISE_FREQ: f64 = 0.01;
// Height amplitude of the terrain surface variation in cells.
pub const HEIGHT_NOISE_AMP_CELLS: i32 = 80;
// Extra high-frequency surface detail frequency multiplier over HEIGHT_NOISE_FREQ.
pub const HEIGHT_NOISE_DETAIL_FREQ_MULT: f64 = 6.0;
// Height amplitude of high-frequency surface detail in cells.
pub const HEIGHT_NOISE_DETAIL_AMP_CELLS: f32 = 2.0;
// Octaves used for high-frequency detail fBm.
pub const HEIGHT_NOISE_DETAIL_OCTAVES: u32 = 3;
// Baseline surface Y level (in cells) before adding noise.
pub const BASE_SURFACE_Y: i32 = 0;
// Mean soil thickness measured downward from the surface (in cells).
pub const SOIL_DEPTH_CELLS: i32 = 4;

// Number of fBm octaves for surface height sampling.
pub const TERRAIN_FBM_OCTAVES: u32 = 4;
// Frequency multiplier per octave for fBm.
pub const TERRAIN_FBM_LACUNARITY: f64 = 2.0;
// Amplitude multiplier per octave for fBm.
pub const TERRAIN_FBM_GAIN: f64 = 0.5;
// Additional attenuation strength for footprint low-pass filtering.
pub const FBM_ATTENUATION_SCALE: f64 = 2.2;

// Base transition band half-width around the surface (in cells).
pub const SURFACE_BLEND_BAND_BASE_CELLS: f32 = 0.35;
// Extra transition width added per footprint cell.
pub const SURFACE_BLEND_BAND_PER_FOOTPRINT: f32 = 0.75;

// Soil depth variation amplitude around SOIL_DEPTH_CELLS (in cells).
pub const SOIL_DEPTH_VARIATION_CELLS: f32 = 3.0;
// Frequency used to modulate soil depth variation.
pub const SOIL_DEPTH_NOISE_FREQ: f64 = HEIGHT_NOISE_FREQ * 0.65;
