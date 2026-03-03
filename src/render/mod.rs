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
}

pub use terrain_gpu::TerrainGpuPlugin;
pub use water_dot_gpu::{WaterDotGpuLabel, WaterDotGpuPlugin};
