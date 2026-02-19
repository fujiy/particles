mod particle;
mod render;
mod terrain;

use bevy::prelude::*;
use particle::{ParticleWorld, draw_water_particles};
use render::{TerrainRenderState, bootstrap_terrain_chunks, sync_dirty_terrain_chunks_to_render};
use terrain::TerrainWorld;
pub use terrain::cell_to_world_center;

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainWorld>()
            .init_resource::<ParticleWorld>()
            .init_resource::<SimulationState>()
            .init_resource::<TerrainRenderState>()
            .add_message::<ResetSimulationRequest>()
            .add_systems(Startup, bootstrap_terrain_chunks)
            .add_systems(
                FixedUpdate,
                step_water_particles.run_if(resource_exists::<TerrainWorld>),
            )
            .add_systems(
                Update,
                (
                    handle_sim_controls,
                    apply_sim_reset,
                    sync_dirty_terrain_chunks_to_render,
                    draw_water_particles,
                )
                    .chain(),
            );
    }
}

#[derive(Resource, Debug, Default)]
pub struct SimulationState {
    running: bool,
}

#[derive(Message)]
struct ResetSimulationRequest;

fn step_water_particles(
    sim_state: Res<SimulationState>,
    terrain_world: Res<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
) {
    particle_world.step_if_running(&terrain_world, sim_state.running);
}

fn handle_sim_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut sim_state: ResMut<SimulationState>,
    mut reset_writer: MessageWriter<ResetSimulationRequest>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.running = !sim_state.running;
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        reset_writer.write(ResetSimulationRequest);
    }
}

fn apply_sim_reset(
    mut reset_reader: MessageReader<ResetSimulationRequest>,
    mut sim_state: ResMut<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
) {
    if reset_reader.read().next().is_none() {
        return;
    }

    terrain_world.reset_fixed_world();
    particle_world.reset_to_initial();
    sim_state.running = false;
}
