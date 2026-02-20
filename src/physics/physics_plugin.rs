use std::time::Instant;

use bevy::log::tracing;
use bevy::prelude::*;

use super::object::{ObjectPhysicsField, ObjectWorld};
use super::particle::{ParticleWorld, TERRAIN_BOUNDARY_RADIUS_M};
use super::save_load;
use super::state::{
    LoadMapRequest, ResetSimulationRequest, SaveMapRequest, SimFixedSet, SimUpdateSet,
    SimulationPerfMetrics, SimulationState,
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
            .add_message::<SaveMapRequest>()
            .add_message::<LoadMapRequest>()
            .add_systems(
                FixedUpdate,
                step_water_particles.in_set(SimFixedSet::Physics),
            )
            .add_systems(
                Update,
                (
                    handle_sim_controls,
                    apply_sim_reset,
                    apply_save_load_requests,
                )
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
    let _step_span = tracing::info_span!("physics::fixed_step").entered();
    let running = sim_state.running;
    {
        let _span = tracing::info_span!("physics::object_field_update").entered();
        object_world.update_physics_field(
            particle_world.positions(),
            particle_world.masses(),
            &mut object_field,
        );
    }
    if running {
        {
            let _span = tracing::info_span!("physics::terrain_rebuild_if_dirty").entered();
            terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        }
        let start = Instant::now();
        {
            let _span = tracing::info_span!("physics::particle_step").entered();
            particle_world.step_if_running(
                &terrain_world,
                &object_field,
                &mut object_world,
                running,
            );
        }
        {
            let _span = tracing::info_span!("physics::terrain_fracture_commit").entered();
            if particle_world.apply_pending_terrain_fractures(&mut terrain_world) {
                terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
            }
        }
        perf_metrics.physics_time_this_frame_secs += start.elapsed().as_secs_f64();
    } else {
        let _span = tracing::info_span!("physics::particle_step").entered();
        particle_world.step_if_running(&terrain_world, &object_field, &mut object_world, running);
    }
}

fn handle_sim_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut sim_state: ResMut<SimulationState>,
    mut reset_writer: MessageWriter<ResetSimulationRequest>,
    mut save_writer: MessageWriter<SaveMapRequest>,
    mut load_writer: MessageWriter<LoadMapRequest>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        sim_state.running = !sim_state.running;
    }

    if keyboard.just_pressed(KeyCode::KeyR) {
        reset_writer.write(ResetSimulationRequest);
    }

    if keyboard.just_pressed(KeyCode::F5) {
        save_writer.write(SaveMapRequest);
    }

    if keyboard.just_pressed(KeyCode::F9) {
        load_writer.write(LoadMapRequest);
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

fn apply_save_load_requests(
    mut save_reader: MessageReader<SaveMapRequest>,
    mut load_reader: MessageReader<LoadMapRequest>,
    mut sim_state: ResMut<SimulationState>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut particle_world: ResMut<ParticleWorld>,
    mut object_world: ResMut<ObjectWorld>,
    mut object_field: ResMut<ObjectPhysicsField>,
) {
    if save_reader.read().next().is_some() {
        match save_load::save_to_default_path(
            &terrain_world,
            &particle_world,
            &object_world,
            &sim_state,
        ) {
            Ok(()) => tracing::info!("saved map to {}", save_load::DEFAULT_SAVE_PATH),
            Err(error) => tracing::error!("failed to save map: {error}"),
        }
    }

    if load_reader.read().next().is_some() {
        match save_load::load_from_default_path(
            &mut terrain_world,
            &mut particle_world,
            &mut object_world,
            &mut sim_state,
        ) {
            Ok(()) => {
                object_field.clear();
                tracing::info!("loaded map from {}", save_load::DEFAULT_SAVE_PATH);
            }
            Err(error) => tracing::error!("failed to load map: {error}"),
        }
    }
}

fn finalize_frame_metrics(mut perf_metrics: ResMut<SimulationPerfMetrics>) {
    perf_metrics.physics_time_this_frame_secs = 0.0;
}
