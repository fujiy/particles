/// TerrainChunkGenerate compute pass.
///
/// Fills a linear material-ID buffer for one chunk so the CPU can read it back.
/// Material encoding: 0 = Empty, 1 = Stone, 2 = Soil, 3 = Sand, 4 = Grass.
#import particles::terrain_gen::{material_for_cell}

struct ChunkGenerateParams {
    chunk_x: i32,
    chunk_y: i32,
    chunk_size_i32: i32,
    generation_enabled: u32,
}

@group(0) @binding(0) var<uniform> params: ChunkGenerateParams;
@group(0) @binding(1) var<storage, read_write> generated_materials: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chunk_size = u32(max(params.chunk_size_i32, 1));
    let total_cells = chunk_size * chunk_size;
    let index = gid.x;
    if index >= total_cells {
        return;
    }

    let local_x = i32(index % chunk_size);
    let local_y = i32(index / chunk_size);
    let world_cell = vec2<i32>(
        params.chunk_x * params.chunk_size_i32 + local_x,
        params.chunk_y * params.chunk_size_i32 + local_y,
    );
    generated_materials[index] = material_for_cell(world_cell, params.generation_enabled != 0u);
}
