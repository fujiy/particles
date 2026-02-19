mod render;
mod terrain;

use bevy::prelude::*;
use render::{TerrainRenderState, bootstrap_terrain_chunks, sync_dirty_terrain_chunks_to_render};
use terrain::TerrainWorld;
pub use terrain::cell_to_world_center;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainWorld>()
            .init_resource::<ParticleWorld>()
            .init_resource::<TerrainRenderState>()
            .add_systems(Startup, bootstrap_terrain_chunks)
            .add_systems(FixedUpdate, terrain_fixed_step)
            .add_systems(Update, sync_dirty_terrain_chunks_to_render);
    }
}

#[derive(Resource, Debug, Default)]
pub struct ParticleWorld {
    fixed_steps: u64,
}

fn terrain_fixed_step(mut particle_world: ResMut<ParticleWorld>) {
    particle_world.fixed_steps = particle_world.fixed_steps.saturating_add(1);
}
