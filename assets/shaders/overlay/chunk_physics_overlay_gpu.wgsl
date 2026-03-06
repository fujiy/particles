#import bevy_render::view::View
#import particles::mpm_types::MpmParams

struct GpuChunkMeta {
    chunk_coord_x: i32,
    chunk_coord_y: i32,
    neighbor_slot_id: array<u32, 8>,
    active_tile_mask: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> params: MpmParams;
@group(0) @binding(2) var<storage, read> chunk_meta: array<GpuChunkMeta>;

struct VertexOut {
    @builtin(position)
    clip_position: vec4<f32>,
    @location(0)
    local_uv: vec2<f32>,
    @location(1)
    edge_color: vec4<f32>,
};

const INVALID_SLOT: u32 = 0xffffffffu;

fn quad_corner(vertex_index: u32) -> vec2<f32> {
    if vertex_index == 0u { return vec2<f32>(0.0, 0.0); }
    if vertex_index == 1u { return vec2<f32>(1.0, 0.0); }
    if vertex_index == 2u { return vec2<f32>(1.0, 1.0); }
    if vertex_index == 3u { return vec2<f32>(0.0, 0.0); }
    if vertex_index == 4u { return vec2<f32>(1.0, 1.0); }
    return vec2<f32>(0.0, 1.0);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOut {
    let uv = quad_corner(vertex_index);
    let chunk_entry = chunk_meta[instance_index];

    let chunk_world_size = max(f32(params.chunk_node_dim) * params.h, params.h);
    let chunk_min = vec2<f32>(
        f32(chunk_entry.chunk_coord_x) * chunk_world_size,
        f32(chunk_entry.chunk_coord_y) * chunk_world_size,
    );
    let world_xy = chunk_min + uv * chunk_world_size;

    var invalid_neighbor_count = 0u;
    for (var i = 0u; i < 8u; i++) {
        if chunk_entry.neighbor_slot_id[i] == INVALID_SLOT {
            invalid_neighbor_count += 1u;
        }
    }
    var edge_color = vec4<f32>(0.98, 0.55, 0.20, 0.92);
    if invalid_neighbor_count == 0u {
        edge_color = vec4<f32>(0.20, 0.88, 0.66, 0.88);
    }

    var out: VertexOut;
    out.clip_position = view.clip_from_world * vec4<f32>(world_xy.x, world_xy.y, 0.0, 1.0);
    out.local_uv = uv;
    out.edge_color = edge_color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let du = min(in.local_uv.x, 1.0 - in.local_uv.x);
    let dv = min(in.local_uv.y, 1.0 - in.local_uv.y);
    let d_edge = min(du, dv);

    let thickness = 0.03;
    let alpha = 1.0 - smoothstep(thickness, thickness + 0.01, d_edge);
    if alpha <= 1.0e-4 {
        discard;
    }
    return vec4<f32>(in.edge_color.rgb, in.edge_color.a * alpha);
}
