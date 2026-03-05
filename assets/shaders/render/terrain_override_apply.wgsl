/// TerrainOverrideApply compute pass.
///
/// Applies RLE-compressed per-chunk override runs to near_override_tex.
/// One run encodes a contiguous linear range in a CHUNK_SIZE x CHUNK_SIZE chunk.

struct OverrideParams {
    cache_origin_x: i32,
    cache_origin_y: i32,
    cache_width: u32,
    cache_height: u32,
    ring_offset_x: i32,
    ring_offset_y: i32,
    override_none: u32,
    run_count: u32,
    chunk_size_i32: i32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

struct OverrideRun {
    chunk_x: i32,
    chunk_y: i32,
    start_index: u32,
    run_length: u32,
    material: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: OverrideParams;
@group(0) @binding(1) var<storage, read> runs: array<OverrideRun>;
@group(0) @binding(2) var near_override_tex: texture_storage_2d<r16uint, write>;

fn positive_mod(x: i32, m: i32) -> i32 {
    let r = x % m;
    return select(r + m, r, r >= 0);
}

fn ring_index(logical: i32, ring_offset: i32, extent: u32) -> i32 {
    return positive_mod(logical + ring_offset, i32(extent));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let run_idx = gid.x;
    if run_idx >= params.run_count {
        return;
    }
    let run = runs[run_idx];
    let chunk_size = max(params.chunk_size_i32, 1);
    let run_end = run.start_index + run.run_length;
    for (var linear = run.start_index; linear < run_end; linear = linear + 1u) {
        let local_x = i32(linear % u32(chunk_size));
        let local_y = i32(linear / u32(chunk_size));
        let world_cell = vec2<i32>(
            run.chunk_x * chunk_size + local_x,
            run.chunk_y * chunk_size + local_y,
        );
        let logical = world_cell - vec2<i32>(params.cache_origin_x, params.cache_origin_y);
        if logical.x < 0 || logical.y < 0
            || logical.x >= i32(params.cache_width)
            || logical.y >= i32(params.cache_height) {
            continue;
        }
        let tx = ring_index(logical.x, params.ring_offset_x, params.cache_width);
        let ty = ring_index(logical.y, params.ring_offset_y, params.cache_height);
        textureStore(
            near_override_tex,
            vec2<i32>(tx, ty),
            vec4<u32>(run.material, 0u, 0u, 0u),
        );
    }
}
