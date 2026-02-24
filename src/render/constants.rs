use crate::physics::world::constants::{CELL_SIZE_M, CHUNK_SIZE};

// Terrain cell size in screen pixels for near-field chunk textures.
pub const CELL_PIXEL_SIZE: u32 = 8;
// Per-chunk texture size in pixels (= CHUNK_SIZE cells * CELL_PIXEL_SIZE).
pub const CHUNK_PIXEL_SIZE: u32 = (CHUNK_SIZE as u32) * CELL_PIXEL_SIZE;

// Density-to-alpha threshold ratio against rest density for grid-mass rendering.
pub const WATER_GRID_DENSITY_THRESHOLD_REST_DENSITY_RATIO: f32 = 0.4;
// Soft transition width around the density threshold.
pub const WATER_DOT_SMOOTH_WIDTH: f32 = 1.0;
// Blur kernel radius in pixels for the grid density field.
pub const WATER_BLUR_RADIUS_DOTS: i32 = 2;
// World-meter to water-texture-pixel scale.
pub const WATER_DOT_SCALE: f32 = CELL_PIXEL_SIZE as f32 / CELL_SIZE_M;
// Upper bound for equivalent particle count to render as explicit splash.
pub const WATER_SPLASH_MAX_EQUIV_PARTICLE_COUNT: f32 = 2.5;
// Equivalent-count threshold that uses the base splash radius.
pub const WATER_SPLASH_SINGLE_EQUIV_PARTICLE_COUNT: f32 = 1.5;
// Splash point radius in pixels for isolated-cell rendering.
pub const WATER_ISOLATED_SPLASH_RADIUS_DOTS: i32 = 2;
// Z layer for water overlay sprites.
pub const WATER_RENDER_Z: f32 = 6.0;
// Z layer for non-object free particles.
pub const FREE_PARTICLE_RENDER_Z: f32 = 6.5;

// World-meter to object-texture-pixel scale.
pub const OBJECT_DOT_SCALE: f32 = CELL_PIXEL_SIZE as f32 / CELL_SIZE_M;
// Z layer for object sprites (above water).
pub const OBJECT_RENDER_Z: f32 = 7.0;
// Padding pixels around object local textures.
pub const OBJECT_PADDING_PX: u32 = 2;

// Z layer for coarse LOD terrain tiles (behind near-field sprites).
pub const LOD_RENDER_Z: f32 = -0.5;
// Texture resolution divisor per LOD level.
pub const LOD_LEVEL_RESOLUTION_DIV: i32 = 2;
// Number of precomputed LOD palettes.
pub const LOD_PRECOMPUTED_LEVELS: u32 = 8;
// Frames to keep tiles alive outside required set to avoid churn.
pub const RENDER_TILE_EVICT_GRACE_FRAMES: u16 = 12;
// Blend to expected color when normalized standard error of mean falls below this.
pub const LOD_EXPECTED_STDERR_BLEND_LOW: f32 = 0.020;
// Keep stochastic sampled color when normalized standard error is above this.
pub const LOD_EXPECTED_STDERR_BLEND_HIGH: f32 = 0.070;
// Correlation length (in cells) used for effective sample count: N_eff = N / corr^2.
pub const LOD_EXPECTED_CORRELATION_CELLS: f32 = 1.6;
