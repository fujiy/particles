// Extract chunk-level occupancy/residency transition events.

#import particles::mpm_types::MpmParams

struct GpuChunkMeta {
    chunk_coord_x: i32,
    chunk_coord_y: i32,
    neighbor_slot_id: array<u32, 8>,
    active_tile_mask: u32,
    particle_count_curr: u32, // previous frame occupancy count
    particle_count_next: u32, // current frame occupancy count
    occupied_bit_curr: u32,
    occupied_bit_next: u32,
}

struct GpuChunkEventRecord {
    slot_id: u32,
    event_kind: u32,
    _pad_a: u32,
    _pad_b: u32,
}

@group(0) @binding(0) var<uniform> params: MpmParams;
@group(0) @binding(1) var<storage, read> chunk_meta: array<GpuChunkMeta>;
@group(0) @binding(2) var<storage, read_write> event_count: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> events: array<GpuChunkEventRecord>;

const EVENT_NEWLY_OCCUPIED: u32 = 1u;
const EVENT_NEWLY_EMPTY: u32 = 2u;
const EVENT_FRONTIER_REQUEST: u32 = 3u;
const EVENT_SLOT_SNAPSHOT: u32 = 4u;
const INVALID_SLOT: u32 = 0xffffffffu;
const MAX_EVENTS: u32 = 1024u; // MAX_RESIDENT_CHUNK_SLOTS(256) * 4

fn append_event(slot_id: u32, kind: u32, payload_a: u32, payload_b: u32) {
    let idx = atomicAdd(&event_count, 1u);
    if idx >= MAX_EVENTS {
        return;
    }
    events[idx] = GpuChunkEventRecord(slot_id, kind, payload_a, payload_b);
}

@compute @workgroup_size(64)
fn extract_chunk_events(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot_id = gid.x;
    if slot_id >= params.resident_chunk_count {
        return;
    }
    let chunk = chunk_meta[slot_id];
    let prev_occupied = chunk.particle_count_curr > 0u;
    let curr_occupied = chunk.particle_count_next > 0u;

    var has_invalid_neighbor = false;
    if curr_occupied {
        for (var i = 0u; i < 8u; i++) {
            if chunk.neighbor_slot_id[i] == INVALID_SLOT {
                has_invalid_neighbor = true;
                break;
            }
        }
    }
    // Absolute occupancy/frontier snapshot (robust against readback frame drops).
    append_event(
        slot_id,
        EVENT_SLOT_SNAPSHOT,
        select(0u, 1u, curr_occupied),
        select(0u, 1u, curr_occupied && has_invalid_neighbor),
    );

    // Keep transition/frontier events for diagnostics compatibility.
    if !prev_occupied && curr_occupied {
        append_event(slot_id, EVENT_NEWLY_OCCUPIED, 0u, 0u);
    } else if prev_occupied && !curr_occupied {
        append_event(slot_id, EVENT_NEWLY_EMPTY, 0u, 0u);
    }
    if curr_occupied && has_invalid_neighbor {
        append_event(slot_id, EVENT_FRONTIER_REQUEST, 0u, 0u);
    }
}
