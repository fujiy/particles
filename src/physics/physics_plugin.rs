use std::time::Instant;

use bevy::prelude::*;

use super::object::{ObjectPhysicsField, ObjectWorld};
use super::particle::{ParticleWorld, TERRAIN_BOUNDARY_RADIUS_M};
use super::state::{
    ResetSimulationRequest, SimFixedSet, SimUpdateSet, SimulationPerfMetrics, SimulationState,
};
use super::terrain::TerrainWorld;

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(FixedUpdate, SimFixedSet::Physics)
            .configure_sets(
                Update,
                (
                    SimUpdateSet::Controls,
                    SimUpdateSet::Interaction,
                    SimUpdateSet::Rendering,
                    SimUpdateSet::Ui,
                    SimUpdateSet::Overlay,
                    SimUpdateSet::Finalize,
                )
                    .chain(),
            )
            .init_resource::<TerrainWorld>()
            .init_resource::<ParticleWorld>()
            .init_resource::<ObjectWorld>()
            .init_resource::<ObjectPhysicsField>()
            .init_resource::<SimulationState>()
            .init_resource::<SimulationPerfMetrics>()
            .add_message::<ResetSimulationRequest>()
            .add_systems(
                FixedUpdate,
                step_water_particles.in_set(SimFixedSet::Physics),
            )
            .add_systems(
                Update,
                (handle_sim_controls, apply_sim_reset)
                    .chain()
                    .in_set(SimUpdateSet::Controls),
            )
            .add_systems(
                Update,
                finalize_frame_metrics.in_set(SimUpdateSet::Finalize),
            );
    }
}

fn step_water_particles(
    sim_state: Res<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
    mut perf_metrics: ResMut<SimulationPerfMetrics>,
) {
    let running = sim_state.running;
    object_world.update_physics_field(
        particle_world.positions(),
        particle_world.masses(),
        &mut object_field,
    );
    if running {
        terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let start = Instant::now();
        particle_world.step_if_running(&terrain_world, &object_field, &mut object_world, running);
        perf_metrics.physics_time_this_frame_secs += start.elapsed().as_secs_f64();
    } else {
        particle_world.step_if_running(&terrain_world, &object_field, &mut object_world, running);
    }
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
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
) {
    if reset_reader.read().next().is_none() {
        return;
    }

    terrain_world.reset_fixed_world();
    terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
    particle_world.reset_to_initial();
    object_world.clear();
    object_field.clear();
    sim_state.running = false;
}

fn finalize_frame_metrics(mut perf_metrics: ResMut<SimulationPerfMetrics>) {
    perf_metrics.physics_time_this_frame_secs = 0.0;
}
