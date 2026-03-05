/// TerrainNearUpdate compute pass.
///
/// Evaluates procedural terrain material IDs on GPU and writes them into
/// `near_tex` (R16Uint). CPU-provided per-cell overrides replace generated
/// values when present.
///
/// Texture layout: one texel == one world cell. `texel (tx, ty)` corresponds to
/// world cell `(cache_origin_x + tx, cache_origin_y + ty)`.
///
/// Material encoding: 0 = Empty, 1 = Stone, 2 = Soil, 3 = Sand.
#import particles::terrain_gen::{material_for_cell}

struct NearParams {
    cache_origin_x:    i32,
    cache_origin_y:    i32,
    cache_width:       u32,
    cache_height:      u32,
    ring_offset_x:     i32,
    ring_offset_y:     i32,
    generation_enabled: u32,
    override_none:     u32,
    dirty_count:       u32,
    _pad0:             u32,
    _pad1:             u32,
    _pad2:             u32,
}

struct DirtyCell {
    world_x:          i32,
    world_y:          i32,
    override_material: u32,
    _pad0:            u32,
}

@group(0) @binding(0) var<uniform> params: NearParams;
@group(0) @binding(1) var<storage, read> dirty_cells: array<DirtyCell>;
@group(0) @binding(2) var near_base_tex: texture_storage_2d<r16uint, write>;
@group(0) @binding(3) var near_override_tex: texture_storage_2d<r16uint, write>;

fn positive_mod(x: i32, m: i32) -> i32 {
    let r = x % m;
    return select(r + m, r, r >= 0);
}

fn ring_index(logical: i32, ring_offset: i32, extent: u32) -> i32 {
    return positive_mod(logical + ring_offset, i32(extent));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.dirty_count {
        return;
    }
    let dirty = dirty_cells[idx];
    let world_cell = vec2<i32>(dirty.world_x, dirty.world_y);

    let logical = world_cell - vec2<i32>(params.cache_origin_x, params.cache_origin_y);
    if logical.x < 0 || logical.y < 0
        || logical.x >= i32(params.cache_width)
        || logical.y >= i32(params.cache_height) {
        return;
    }

    let tx = ring_index(logical.x, params.ring_offset_x, params.cache_width);
    let ty = ring_index(logical.y, params.ring_offset_y, params.cache_height);
    let generated = material_for_cell(world_cell, params.generation_enabled != 0u);

    var material = generated;
    if dirty.override_material != params.override_none {
        material = dirty.override_material;
    }

    textureStore(near_base_tex, vec2<i32>(tx, ty), vec4<u32>(material, 0u, 0u, 0u));
    textureStore(
        near_override_tex,
        vec2<i32>(tx, ty),
        vec4<u32>(dirty.override_material, 0u, 0u, 0u),
    );
}
