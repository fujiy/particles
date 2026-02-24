use std::time::Instant;

use bevy::log::tracing;

use crate::physics::profiler::process_cpu_time_seconds;
use crate::physics::solver::granular;
use crate::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::{ParticleStepBreakdown, ParticleWorld};
use crate::physics::world::terrain::TerrainWorld;

pub(super) fn step_substeps(
    particles: &mut ParticleWorld,
    terrain: &TerrainWorld,
    object_field: &ObjectPhysicsField,
    object_world: &mut ObjectWorld,
) -> ParticleStepBreakdown {
    let _span = tracing::info_span!("physics::step_substeps").entered();
    particles.release_far_field_particles_for_active_chunks();
    let dt_sub = particles.solver_params.fixed_dt / particles.solver_params.substeps as f32;
    let mut breakdown = ParticleStepBreakdown::default();
    for substep in 0..particles.solver_params.substeps {
        let _substep_span = tracing::info_span!("physics::substep", substep).entered();
        step_single_substep(
            particles,
            terrain,
            object_field,
            object_world,
            dt_sub,
            &mut breakdown,
        );
    }
    breakdown
}

pub(super) fn step_single_substep(
    particles: &mut ParticleWorld,
    terrain: &TerrainWorld,
    object_field: &ObjectPhysicsField,
    object_world: &mut ObjectWorld,
    dt_sub: f32,
    breakdown: &mut ParticleStepBreakdown,
) {
    particles.terrain_load_substep_counter = particles.terrain_load_substep_counter.wrapping_add(1);
    particles.object_peak_strain.clear();
    particles.object_peak_strain_particle.clear();
    let mut phase_wall_start = Instant::now();
    let mut phase_cpu_start = process_cpu_time_seconds().unwrap_or(0.0);
    {
        let _span = tracing::info_span!("physics::clear_reaction_impulses").entered();
        object_world.clear_reaction_impulses();
    }
    breakdown.clear_reaction_impulses_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.clear_reaction_impulses_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::predict_positions").entered();
        for i in 0..particles.particle_count() {
            particles.particle_execution_dt_substep[i] = 0.0;
            particles.prev_pos[i] = particles.pos[i];
            if !particles.is_active_particle(i) {
                if particles.is_halo_particle(i) {
                    // Halo particles stay frozen but keep their velocity state so that
                    // reactivation can resume with preserved momentum.
                    continue;
                }
                // Keep ballistic motion outside active region to avoid artificial "walls"
                // at the region boundary. Heavy constraints/collisions remain skipped.
                particles.pos[i] += particles.vel[i] * dt_sub;
                continue;
            }
            let effective_dt = particles.particle_execution_dt_before_scheduler(i, dt_sub);
            particles.particle_execution_dt_substep[i] = effective_dt;
            if effective_dt <= 0.0 {
                continue;
            }
            particles.vel[i] += particles.solver_params.gravity_mps2 * effective_dt;
            particles.pos[i] += particles.vel[i] * effective_dt;
        }
    }
    breakdown.predict_positions_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.predict_positions_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::cull_escaped_particles").entered();
        particles.capture_far_field_intrusions(object_world);
        particles.cull_escaped_particles(terrain, object_world);
    }
    breakdown.cull_escaped_particles_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.cull_escaped_particles_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::sub_block_rate_scheduler").entered();
        particles.prepare_sub_block_rate_scheduler(terrain);
    }
    breakdown.sub_block_rate_scheduler_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.sub_block_rate_scheduler_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);

    {
        let _span = tracing::info_span!("physics::sub_block_debt_apply").entered();
        particles.apply_sub_block_debt_before_constraints(dt_sub);
    }
    breakdown.sub_block_debt_apply_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.sub_block_debt_apply_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);

    {
        let _span = tracing::info_span!("physics::rebuild_neighbor_grid").entered();
        particles.neighbor_grid.rebuild(&particles.pos);
    }
    breakdown.rebuild_neighbor_grid_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.rebuild_neighbor_grid_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    let mut refresh_density_neighbors = true;
    for iter in 0..particles.solver_params.solver_iters {
        let _iter_span = tracing::info_span!("physics::solve_density_constraints", iter).entered();
        let solve_result = particles.solve_density_constraints(
            terrain,
            object_field,
            object_world,
            dt_sub,
            refresh_density_neighbors,
        );
        breakdown.sub_block_debt_accumulate_secs += solve_result.debt_accumulate_wall_secs;
        breakdown.sub_block_debt_accumulate_cpu_secs += solve_result.debt_accumulate_cpu_secs;
        refresh_density_neighbors = particles.neighbor_cache_requires_rebuild();
        if iter + 1 >= particles.solver_params.solver_min_iters
            && solve_result.max_density_error <= particles.solver_params.solver_error_tolerance
        {
            break;
        }
    }
    breakdown.solve_density_constraints_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.solve_density_constraints_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::granular_solver").entered();
        let reaction_impulses =
            granular::solve_contacts(particles, terrain, object_field, object_world, dt_sub);
        for (object_id, impulse) in reaction_impulses {
            object_world.accumulate_reaction_impulse(object_id, impulse);
        }
    }
    breakdown.granular_solver_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.granular_solver_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::shape_matching").entered();
        particles.solve_shape_matching_constraints(object_world);
    }
    breakdown.shape_matching_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.shape_matching_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::shape_contact_projection").entered();
        particles.project_solids_out_of_terrain(terrain);
    }
    breakdown.shape_contact_projection_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.shape_contact_projection_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);

    {
        let _span = tracing::info_span!("physics::update_velocity").entered();
        for i in 0..particles.particle_count() {
            if !particles.is_active_particle(i) {
                continue;
            }
            let effective_dt = particles.particle_execution_dt_substep[i];
            if effective_dt <= 0.0 {
                continue;
            }
            particles.vel[i] = (particles.pos[i] - particles.prev_pos[i]) / effective_dt;
            particles.vel[i] = particles.vel[i]
                .clamp_length_max(particles.material_params.particle_speed_limit_mps);
        }
    }
    breakdown.update_velocity_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.update_velocity_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::granular_restitution").entered();
        granular::apply_restitution(particles, dt_sub);
    }
    breakdown.granular_restitution_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.granular_restitution_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::contact_velocity_response").entered();
        particles.apply_contact_velocity_response(terrain, object_field, object_world);
    }
    breakdown.contact_velocity_response_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.contact_velocity_response_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);

    {
        let _span = tracing::info_span!("physics::xsph_viscosity").entered();
        particles.apply_xsph_viscosity();
    }
    breakdown.xsph_viscosity_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.xsph_viscosity_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::apply_object_reaction").entered();
        particles.apply_object_reaction_impulses(object_world);
    }
    breakdown.apply_object_reaction_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.apply_object_reaction_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::final_velocity_clamp").entered();
        for i in 0..particles.particle_count() {
            particles.vel[i] = particles.vel[i]
                .clamp_length_max(particles.material_params.particle_speed_limit_mps);
        }
    }
    breakdown.final_velocity_clamp_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.final_velocity_clamp_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    let phase_cpu_end = if particles.solver_params.enable_sleep_wake {
        {
            let _span = tracing::info_span!("physics::wake_detection").entered();
            particles.neighbor_grid.rebuild(&particles.pos);
            particles.detect_wake_events(terrain, object_field, object_world);
            particles.propagate_and_apply_wake_requests();
        }
        breakdown.wake_detection_secs += phase_wall_start.elapsed().as_secs_f64();
        process_cpu_time_seconds().unwrap_or(phase_cpu_start)
    } else {
        phase_cpu_start
    };
    breakdown.wake_detection_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    {
        let _span = tracing::info_span!("physics::fracture_detection").entered();
        particles.detect_fracture_candidates(terrain, object_field, object_world);
        particles.apply_object_fractures(object_world);
    }
    breakdown.fracture_detection_secs += phase_wall_start.elapsed().as_secs_f64();
    let phase_cpu_end = process_cpu_time_seconds().unwrap_or(phase_cpu_start);
    breakdown.fracture_detection_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
    phase_wall_start = Instant::now();
    phase_cpu_start = process_cpu_time_seconds().unwrap_or(phase_cpu_end);
    let phase_cpu_end = if particles.solver_params.enable_sleep_wake {
        {
            let _span = tracing::info_span!("physics::sleep_update").entered();
            particles.update_sleep_states(object_world);
        }
        breakdown.sleep_update_secs += phase_wall_start.elapsed().as_secs_f64();
        process_cpu_time_seconds().unwrap_or(phase_cpu_start)
    } else {
        phase_cpu_start
    };
    breakdown.sleep_update_cpu_secs += (phase_cpu_end - phase_cpu_start).max(0.0);
}
