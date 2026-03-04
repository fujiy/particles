/// TerrainFarUpdate compute pass.
///
/// Evaluates aggregated terrain materials directly from procedural world generation.
/// Output texel layout (RGBA8Uint):
///   R = top1 material id (0..3)
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
    _pad0:             u32,
    _pad1:             u32,
}

@group(0) @binding(0) var<uniform> params: FarParams;
@group(0) @binding(1) var far_tex: texture_storage_2d<rgba8uint, write>;

const FAR_AGGREGATE_SAMPLES_AXIS: i32 = 2;

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
    var solid_count: u32 = 0u;

    let sample_span = f32(downsample) / f32(FAR_AGGREGATE_SAMPLES_AXIS);
    for (var sy = 0; sy < FAR_AGGREGATE_SAMPLES_AXIS; sy = sy + 1) {
        for (var sx = 0; sx < FAR_AGGREGATE_SAMPLES_AXIS; sx = sx + 1) {
            let sample_x =
                min(i32(floor((f32(sx) + 0.5) * sample_span)), downsample - 1);
            let sample_y =
                min(i32(floor((f32(sy) + 0.5) * sample_span)), downsample - 1);
            let world_cell = world_base + vec2<i32>(sample_x, sample_y);
            let material = material_for_cell(world_cell, params.generation_enabled != 0u);
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
