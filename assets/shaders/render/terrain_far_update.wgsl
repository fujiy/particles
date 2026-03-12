/// TerrainFarUpdate compute pass.
///
/// Evaluates aggregated terrain materials directly from procedural world generation.
/// Output texel layout (RGBA8Uint):
///   R = top1 material id (0..4)
///   G = top1 weight in solid samples (0..255)
///   B = solid fraction (0..255)
///   A = reserved

#import particles::terrain_gen::{material_for_cell}

struct FarParams {
    far_origin_x:      i32,
    far_origin_y:      i32,
    far_width:         u32,
    far_height:        u32,
    far_downsample:    u32,
    generation_enabled: u32,
    near_origin_x:     i32,
    near_origin_y:     i32,
    near_width:        u32,
    near_height:       u32,
    ring_offset_x:     i32,
    ring_offset_y:     i32,
    override_none:     u32,
    override_hash_mask: u32,
    override_hash_len:  u32,
    override_chunk_count: u32,
    override_chunk_size_i32: i32,
    near_cache_enabled: u32,
    _pad0:             u32,
    _pad1:             u32,
}

struct OverrideHashEntry {
    chunk_x: i32,
    chunk_y: i32,
    chunk_index: i32,
    _pad0: i32,
}

@group(0) @binding(0) var<uniform> params: FarParams;
@group(0) @binding(1) var far_tex: texture_storage_2d<rgba8uint, write>;
@group(0) @binding(2) var near_base_tex: texture_2d<u32>;
@group(0) @binding(3) var near_override_tex: texture_2d<u32>;
@group(0) @binding(4) var<storage, read> override_hash_entries: array<OverrideHashEntry>;
@group(0) @binding(5) var<storage, read> override_cells: array<u32>;

const FAR_AGGREGATE_SAMPLES_AXIS: i32 = 2;

fn positive_mod(x: i32, m: i32) -> i32 {
    let r = x % m;
    return select(r + m, r, r >= 0);
}

fn ring_index(logical: i32, ring_offset: i32, extent: u32) -> i32 {
    return positive_mod(logical + ring_offset, i32(extent));
}

fn div_euclid(x: i32, d: i32) -> i32 {
    let q = x / d;
    let r = x % d;
    return select(q - 1, q, r >= 0);
}

fn rem_euclid(x: i32, d: i32) -> i32 {
    let r = x % d;
    return select(r + d, r, r >= 0);
}

fn hash_chunk(chunk_coord: vec2<i32>) -> u32 {
    let x = bitcast<u32>(chunk_coord.x);
    let y = bitcast<u32>(chunk_coord.y);
    let y_mix = y * 0x85EBCA77u;
    var h = (x * 0x9E3779B1u) ^ ((y_mix << 16u) | (y_mix >> 16u));
    h = h ^ (h >> 16u);
    return h;
}

fn lookup_sparse_override_material(world_cell: vec2<i32>) -> u32 {
    if params.override_chunk_count == 0u || params.override_hash_len == 0u {
        return params.override_none;
    }
    let chunk_size = max(params.override_chunk_size_i32, 1);
    let chunk_coord = vec2<i32>(
        div_euclid(world_cell.x, chunk_size),
        div_euclid(world_cell.y, chunk_size),
    );
    let local = vec2<i32>(
        rem_euclid(world_cell.x, chunk_size),
        rem_euclid(world_cell.y, chunk_size),
    );
    let chunk_area = u32(chunk_size * chunk_size);
    let local_index = u32(local.y * chunk_size + local.x);
    var slot = hash_chunk(chunk_coord) & params.override_hash_mask;
    var probe: u32 = 0u;
    loop {
        if probe >= params.override_hash_len {
            break;
        }
        let entry = override_hash_entries[slot];
        if entry.chunk_index < 0 {
            break;
        }
        if entry.chunk_x == chunk_coord.x && entry.chunk_y == chunk_coord.y {
            let chunk_index = u32(entry.chunk_index);
            return override_cells[chunk_index * chunk_area + local_index];
        }
        slot = (slot + 1u) & params.override_hash_mask;
        probe = probe + 1u;
    }
    return params.override_none;
}

fn material_for_far_sample(world_cell: vec2<i32>) -> u32 {
    let sparse_override = lookup_sparse_override_material(world_cell);
    if sparse_override != params.override_none {
        return sparse_override;
    }

    if params.near_cache_enabled != 0u {
        let near_logical =
            world_cell - vec2<i32>(params.near_origin_x, params.near_origin_y);
        if near_logical.x >= 0
            && near_logical.y >= 0
            && near_logical.x < i32(params.near_width)
            && near_logical.y < i32(params.near_height)
        {
            let tx = ring_index(near_logical.x, params.ring_offset_x, params.near_width);
            let ty = ring_index(near_logical.y, params.ring_offset_y, params.near_height);
            let tc = vec2<i32>(tx, ty);
            let near_override = textureLoad(near_override_tex, tc, 0).r;
            if near_override != params.override_none {
                return near_override;
            }
            return textureLoad(near_base_tex, tc, 0).r;
        }
    }
    return material_for_cell(world_cell, params.generation_enabled != 0u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fx = gid.x;
    let fy = gid.y;
    if fx >= params.far_width || fy >= params.far_height {
        return;
    }

    let downsample = i32(max(params.far_downsample, 1u));
    let world_base = vec2<i32>(
        params.far_origin_x + i32(fx) * downsample,
        params.far_origin_y + i32(fy) * downsample,
    );
    var count_stone: u32 = 0u;
    var count_soil: u32 = 0u;
    var count_sand: u32 = 0u;
    var count_grass: u32 = 0u;
    var solid_count: u32 = 0u;

    let sample_span = f32(downsample) / f32(FAR_AGGREGATE_SAMPLES_AXIS);
    for (var sy = 0; sy < FAR_AGGREGATE_SAMPLES_AXIS; sy = sy + 1) {
        for (var sx = 0; sx < FAR_AGGREGATE_SAMPLES_AXIS; sx = sx + 1) {
            let sample_x =
                min(i32(floor((f32(sx) + 0.5) * sample_span)), downsample - 1);
            let sample_y =
                min(i32(floor((f32(sy) + 0.5) * sample_span)), downsample - 1);
            let world_cell = world_base + vec2<i32>(sample_x, sample_y);
            let material = material_for_far_sample(world_cell);
            if material == 0u {
                continue;
            }
            solid_count = solid_count + 1u;
            if material == 1u {
                count_stone = count_stone + 1u;
            } else if material == 2u {
                count_soil = count_soil + 1u;
            } else if material == 3u {
                count_sand = count_sand + 1u;
            } else if material == 4u {
                count_grass = count_grass + 1u;
            }
        }
    }

    var top1_id: u32 = 0u;
    var top1_count: u32 = 0u;
    if count_stone > top1_count {
        top1_count = count_stone;
        top1_id = 1u;
    }
    if count_soil > top1_count {
        top1_count = count_soil;
        top1_id = 2u;
    }
    if count_sand > top1_count {
        top1_count = count_sand;
        top1_id = 3u;
    }
    if count_grass > top1_count {
        top1_count = count_grass;
        top1_id = 4u;
    }

    var top1_weight: u32 = 0u;
    if solid_count > 0u && top1_count > 0u {
        top1_weight = min((top1_count * 255u + (solid_count / 2u)) / solid_count, 255u);
    }
    let sample_count: u32 =
        u32(FAR_AGGREGATE_SAMPLES_AXIS * FAR_AGGREGATE_SAMPLES_AXIS);
    let solid_fraction = min((solid_count * 255u + (sample_count / 2u)) / sample_count, 255u);

    textureStore(
        far_tex,
        vec2<i32>(i32(fx), i32(fy)),
        vec4<u32>(top1_id, top1_weight, solid_fraction, 0u),
    );
}
