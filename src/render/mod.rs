use std::collections::HashMap;

use bevy::prelude::*;

mod terrain_gpu;
mod water_dot_gpu;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderVisibleTile {
    pub origin_chunk: IVec2,
    pub span_chunks: i32,
    pub lod_level: u32,
}

#[derive(Resource, Default)]
pub struct TerrainRenderDiagnostics {
    pub terrain_updated_chunk_highlight_frames: HashMap<IVec2, u8>,
    pub particle_updated_chunk_highlight_frames: HashMap<IVec2, u8>,
    pub particle_updated_sub_block_highlight_frames: HashMap<IVec2, u8>,
    pub last_particle_dirty_frame: u64,
    pub visible_tiles: Vec<RenderVisibleTile>,
    pub terrain_generation_eval_count_frame: u32,
    pub terrain_generation_origin_delta_x_frame: i32,
    pub terrain_generation_origin_delta_y_frame: i32,
    pub terrain_generation_full_refresh_frame: bool,
    pub terrain_generation_full_refresh_reason_bits: u32,
    pub terrain_override_runs_frame: u32,
    pub terrain_override_cells_frame: u32,
    pub terrain_override_pending_runs: u32,
    pub terrain_override_budget_completion_frame: f32,
    pub terrain_override_runs_total: u64,
    pub terrain_override_cells_total: u64,
}

pub use terrain_gpu::{TerrainGeneratedChunkCache, TerrainGpuPlugin};
pub use water_dot_gpu::{WaterDotGpuLabel, WaterDotGpuPlugin};
