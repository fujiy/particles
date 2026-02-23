use crate::physics::world::constants::{CELL_SIZE_M, CHUNK_SIZE};

pub const CELL_PIXEL_SIZE: u32 = 8;
pub const CHUNK_PIXEL_SIZE: u32 = (CHUNK_SIZE as u32) * CELL_PIXEL_SIZE;

pub const WATER_DOT_THRESHOLD_REST_DENSITY_RATIO: f32 = 0.4;
pub const WATER_DOT_SMOOTH_WIDTH: f32 = 1.0;
pub const WATER_BLUR_RADIUS_DOTS: i32 = 2;
pub const WATER_DOT_SCALE: f32 = CELL_PIXEL_SIZE as f32 / CELL_SIZE_M;
pub const WATER_RENDER_Z: f32 = 6.0;
pub const FREE_PARTICLE_RENDER_Z: f32 = 6.5;

pub const OBJECT_DOT_SCALE: f32 = CELL_PIXEL_SIZE as f32 / CELL_SIZE_M;
pub const OBJECT_RENDER_Z: f32 = 7.0;
pub const OBJECT_PADDING_PX: u32 = 2;

pub const LOD_RENDER_Z: f32 = -0.5;
pub const LOD_LEVEL_RESOLUTION_DIV: i32 = 2;
pub const LOD_PRECOMPUTED_LEVELS: u32 = 8;
pub const RENDER_TILE_EVICT_GRACE_FRAMES: u16 = 12;
// Blend to expected color when normalized standard error of mean falls below this.
pub const LOD_EXPECTED_STDERR_BLEND_LOW: f32 = 0.020;
// Keep stochastic sampled color when normalized standard error is above this.
pub const LOD_EXPECTED_STDERR_BLEND_HIGH: f32 = 0.070;
// Correlation length (in cells) used for effective sample count: N_eff = N / corr^2.
pub const LOD_EXPECTED_CORRELATION_CELLS: f32 = 1.6;
