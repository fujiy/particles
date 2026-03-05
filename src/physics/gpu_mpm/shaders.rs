// Shader asset handles for GPU MPM compute passes.

use bevy::prelude::*;

#[derive(Resource)]
pub struct MpmShaders {
    pub types: Handle<Shader>,
    pub clear: Handle<Shader>,
    pub p2g: Handle<Shader>,
    pub grid_update: Handle<Shader>,
    pub g2p: Handle<Shader>,
    pub stats_clear: Handle<Shader>,
    pub stats_total_particles: Handle<Shader>,
    pub stats_phase_counts: Handle<Shader>,
    pub stats_max_speed: Handle<Shader>,
    pub stats_penetration_tracking: Handle<Shader>,
    pub stats_water_surface_histogram: Handle<Shader>,
    pub stats_water_surface_finalize: Handle<Shader>,
    pub stats_granular_repose_bounds: Handle<Shader>,
    pub stats_granular_repose_finalize: Handle<Shader>,
    pub stats_interaction_clear_cells: Handle<Shader>,
    pub stats_interaction_mark_secondary: Handle<Shader>,
    pub stats_interaction_primary_contact: Handle<Shader>,
    pub stats_interaction_finalize: Handle<Shader>,
    pub stats_grid_density: Handle<Shader>,
    pub stats_grid_density_finalize: Handle<Shader>,
}

impl FromWorld for MpmShaders {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        Self {
            types: asset_server.load("shaders/physics/mpm_types.wgsl"),
            clear: asset_server.load("shaders/physics/mpm_clear.wgsl"),
            p2g: asset_server.load("shaders/physics/mpm_p2g.wgsl"),
            grid_update: asset_server.load("shaders/physics/mpm_grid_update.wgsl"),
            g2p: asset_server.load("shaders/physics/mpm_g2p.wgsl"),
            stats_clear: asset_server.load("shaders/stats/clear_scalars.wgsl"),
            stats_total_particles: asset_server.load("shaders/stats/total_particles.wgsl"),
            stats_phase_counts: asset_server.load("shaders/stats/phase_counts.wgsl"),
            stats_max_speed: asset_server.load("shaders/stats/max_speed.wgsl"),
            stats_penetration_tracking: asset_server.load(
                "shaders/stats/penetration_tracking.wgsl",
            ),
            stats_water_surface_histogram: asset_server.load(
                "shaders/stats/water_surface_histogram.wgsl",
            ),
            stats_water_surface_finalize: asset_server.load(
                "shaders/stats/water_surface_finalize.wgsl",
            ),
            stats_granular_repose_bounds: asset_server.load(
                "shaders/stats/granular_repose_bounds.wgsl",
            ),
            stats_granular_repose_finalize: asset_server.load(
                "shaders/stats/granular_repose_finalize.wgsl",
            ),
            stats_interaction_clear_cells: asset_server.load(
                "shaders/stats/interaction_clear_cells.wgsl",
            ),
            stats_interaction_mark_secondary: asset_server.load(
                "shaders/stats/interaction_mark_secondary.wgsl",
            ),
            stats_interaction_primary_contact: asset_server.load(
                "shaders/stats/interaction_primary_contact.wgsl",
            ),
            stats_interaction_finalize: asset_server.load(
                "shaders/stats/interaction_finalize.wgsl",
            ),
            stats_grid_density: asset_server.load("shaders/stats/grid_density.wgsl"),
            stats_grid_density_finalize: asset_server.load(
                "shaders/stats/grid_density_finalize.wgsl",
            ),
        }
    }
}
