use bevy::prelude::*;

use super::*;

pub(super) fn draw_particle_overlay(
    mut gizmos: Gizmos,
    overlay_state: Res<ParticleOverlayState>,
    active_region: Res<PhysicsActiveRegion>,
    region_settings: Res<PhysicsRegionSettings>,
    terrain_world: Res<TerrainWorld>,
    particle_world: Res<ParticleWorld>,
    object_world: Res<ObjectWorld>,
) {
    if !overlay_state.enabled {
        return;
    }

    let water_overlay_radius = nominal_particle_draw_radius_m();
    let terrain_overlay_chunk_bounds =
        active_region
            .chunk_min
            .zip(active_region.chunk_max)
            .map(|(min_chunk, max_chunk)| {
                let halo_chunks = region_settings.active_halo_chunks.max(0);
                (
                    min_chunk - IVec2::splat(halo_chunks),
                    max_chunk + IVec2::splat(halo_chunks),
                )
            });

    for pos in terrain_world.static_particle_positions() {
        let Some((overlay_min_chunk, overlay_max_chunk)) = terrain_overlay_chunk_bounds else {
            continue;
        };
        let cell = world_to_cell(*pos);
        let chunk = IVec2::new(
            cell.x.div_euclid(CHUNK_SIZE_I32),
            cell.y.div_euclid(CHUNK_SIZE_I32),
        );
        if chunk.x < overlay_min_chunk.x
            || chunk.x > overlay_max_chunk.x
            || chunk.y < overlay_min_chunk.y
            || chunk.y > overlay_max_chunk.y
        {
            continue;
        }
        gizmos
            .circle_2d(
                *pos,
                TERRAIN_PARTICLE_RADIUS_M,
                Color::srgba(0.84, 0.54, 0.28, 0.65),
            )
            .resolution(PARTICLE_OVERLAY_CIRCLE_RESOLUTION);
    }
    for (index, pos) in particle_world.positions().iter().enumerate() {
        let is_object_particle = object_world.object_of_particle(index).is_some();
        let is_active = particle_world.is_particle_active_in_region(index);
        let is_halo = particle_world.is_particle_in_halo_region(index);
        if !is_object_particle && !is_active && !is_halo {
            continue;
        }
        let color = if is_object_particle {
            let alpha = if is_active { 0.95 } else { 0.55 };
            Color::srgba(0.98, 0.90, 0.38, alpha)
        } else if is_halo {
            Color::srgba(0.22, 0.92, 0.62, 0.82)
        } else {
            match particle_world.materials()[index] {
                ParticleMaterial::WaterLiquid => Color::srgba(0.10, 0.80, 0.95, 0.85),
                ParticleMaterial::StoneSolid
                | ParticleMaterial::StoneGranular
                | ParticleMaterial::SoilSolid
                | ParticleMaterial::SoilGranular
                | ParticleMaterial::SandSolid
                | ParticleMaterial::SandGranular => Color::srgba(0.63, 0.50, 0.34, 0.85),
            }
        };
        gizmos
            .circle_2d(*pos, water_overlay_radius, color)
            .resolution(PARTICLE_OVERLAY_CIRCLE_RESOLUTION);
    }
}
