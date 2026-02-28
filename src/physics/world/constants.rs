pub const CELL_SIZE_M: f32 = 0.25;
pub const CHUNK_SIZE: usize = 16;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CHUNK_WORLD_SIZE_M: f32 = (CHUNK_SIZE as f32) * CELL_SIZE_M;
pub const DEFAULT_SOLID_HP: u16 = 1_000;
pub const WORLD_MIN_CHUNK_X: i32 = -2;
pub const WORLD_MAX_CHUNK_X: i32 = 1;
pub const WORLD_MIN_CHUNK_Y: i32 = -2;
pub const WORLD_MAX_CHUNK_Y: i32 = 1;
pub const TERRAIN_SDF_SAMPLES_PER_CELL: i32 = 2;
// Default MPM block edge length in grid cells.
// Node count is (cell span + 1) per axis because boundary nodes are shared.
pub const DEFAULT_MPM_BLOCK_NODE_SPAN: i32 = 16;
