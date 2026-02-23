use std::collections::{HashMap, HashSet};

use bevy::log::tracing;
use bevy::prelude::{IVec2, Vec2};

use super::TerrainFractureSeed;
use crate::physics::material::{
    particle_properties, solid_break_properties, terrain_break_collision_impulse_threshold,
    terrain_fracture_particle,
};
use crate::physics::world::object::{ObjectId, ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::ParticleWorld;
use crate::physics::world::particle::helpers::{
    is_granular_particle, resolve_terrain_contact_cell,
};
use crate::physics::world::terrain::{CELL_SIZE_M, TerrainCell, TerrainWorld};

pub(super) fn apply_object_reaction_impulses(
    particles: &mut ParticleWorld,
    object_world: &ObjectWorld,
) {
    let _span = tracing::info_span!("physics::object_reaction_pass").entered();
    for object in object_world.objects() {
        let impulse = object_world.reaction_impulse_of(object.id);
        if impulse.length_squared() <= 1e-12 {
            continue;
        }

        let mass_sum = object
            .particle_indices
            .iter()
            .filter_map(|&index| particles.mass.get(index).copied())
            .sum::<f32>()
            .max(1e-6);
        let dv = (impulse / mass_sum).clamp_length_max(
            particles
                .solver_params
                .object_reaction_max_dv_per_substep_mps,
        );
        for &index in &object.particle_indices {
            if index >= particles.particle_count() {
                continue;
            }
            particles.vel[index] += dv;
        }
    }
}

pub(super) fn detect_fracture_candidates(
    particles: &mut ParticleWorld,
    terrain: &TerrainWorld,
    object_field: &ObjectPhysicsField,
    object_world: &ObjectWorld,
) {
    debug_assert!(
        !particles
            .material_params
            .enable_granular_to_solid_reconversion,
        "v4 keeps granular->solid reconversion disabled"
    );
    particles.pending_object_fractures.clear();
    particles.pending_object_fracture_particles.clear();
    particles.pending_terrain_fractures.clear();
    particles.pending_terrain_fracture_seeds.clear();

    let mut terrain_impulse = HashMap::<IVec2, f32>::new();
    let mut terrain_load_metric = HashMap::<IVec2, f32>::new();
    let mut terrain_seed = HashMap::<IVec2, TerrainFractureSeed>::new();
    let mut object_terrain_impulse = HashMap::<ObjectId, f32>::new();
    for i in 0..particles.particle_count() {
        if !particles.is_active_particle(i) {
            continue;
        }
        let particle_material = particles.material[i];
        let props = particle_properties(particle_material);
        let Some((signed_distance, normal)) =
            terrain.sample_signed_distance_and_normal(particles.pos[i])
        else {
            continue;
        };
        let penetration = props.terrain_push_radius_m - signed_distance;
        if penetration <= 0.0 {
            continue;
        }
        let dt_sub = particles.solver_params.fixed_dt / particles.solver_params.substeps as f32;
        let step_normal_velocity =
            (particles.pos[i] - particles.prev_pos[i]).dot(normal) / dt_sub.max(1e-6);
        let approach_speed = (-particles.vel[i].dot(normal))
            .max(-step_normal_velocity)
            .max(0.0);
        let contact_speed = particles.vel[i]
            .dot(normal)
            .abs()
            .max(step_normal_velocity.abs());
        let impact_speed = if approach_speed > 0.0 {
            approach_speed
        } else if contact_speed >= particles.solver_params.fracture_min_impact_speed_mps {
            contact_speed
        } else {
            0.0
        };
        let impulse = particles.mass[i] * impact_speed;
        if impulse <= 1e-6 {
            continue;
        }
        if let Some(object_id) = object_world.object_of_particle(i) {
            object_terrain_impulse
                .entry(object_id)
                .and_modify(|value| *value = value.max(impulse))
                .or_insert(impulse);
            particles
                .pending_object_fracture_particles
                .entry(object_id)
                .or_default()
                .insert(i);
        }
        let Some(cell) =
            resolve_terrain_contact_cell(particles.pos[i], signed_distance, normal, terrain)
        else {
            continue;
        };
        let strain_metric = (penetration / props.terrain_push_radius_m.max(1e-6)).max(0.0);
        terrain_load_metric
            .entry(cell)
            .and_modify(|metric| *metric = metric.max(strain_metric))
            .or_insert(strain_metric);
        let entry = terrain_impulse.entry(cell).or_insert(0.0);
        *entry = (*entry).max(impulse);
        terrain_seed.entry(cell).or_default().accumulate(
            impulse,
            particles.vel[i],
            particles.pos[i],
        );
    }

    particles.evaluate_terrain_persistent_load(terrain, &terrain_load_metric);

    let mut object_contacts = Vec::new();
    for i in 0..particles.particle_count() {
        if !particles.is_active_particle(i) {
            continue;
        }
        let Some(object_id) = object_world.object_of_particle(i) else {
            continue;
        };
        let props = particle_properties(particles.material[i]);
        object_field.gather_candidate_object_ids(particles.pos[i], &mut object_contacts);
        for &contact_id in &object_contacts {
            if contact_id == object_id {
                continue;
            }
            let Some(sample) = object_world.evaluate_object_sdf(contact_id, particles.pos[i])
            else {
                continue;
            };
            let penetration = props.object_push_radius_m - sample.distance_m;
            if penetration <= 0.0 {
                continue;
            }
            particles
                .pending_object_fracture_particles
                .entry(object_id)
                .or_default()
                .insert(i);
            break;
        }
    }

    for object in object_world.objects() {
        let Some(&particle_index) = object.particle_indices.first() else {
            continue;
        };
        if particle_index >= particles.particle_count() {
            continue;
        }
        let source_material = particles.material[particle_index];
        let Some(break_props) = solid_break_properties(source_material) else {
            continue;
        };

        let external_collision_impulse = object_world.reaction_impulse_of(object.id).length();
        let terrain_collision_impulse = object_terrain_impulse
            .get(&object.id)
            .copied()
            .unwrap_or(0.0);
        let collision_impulse = external_collision_impulse + terrain_collision_impulse;
        let strain = particles
            .object_peak_strain
            .get(&object.id)
            .copied()
            .unwrap_or(0.0);
        let should_fracture = collision_impulse >= break_props.break_collision_impulse_threshold
            || strain >= break_props.break_strain_threshold;
        if should_fracture {
            particles.pending_object_fractures.insert(object.id);
            let fracture_particles = particles
                .pending_object_fracture_particles
                .entry(object.id)
                .or_default();
            if fracture_particles.is_empty() {
                if let Some(&peak_particle) = particles.object_peak_strain_particle.get(&object.id)
                {
                    fracture_particles.insert(peak_particle);
                } else {
                    fracture_particles.insert(particle_index);
                }
            }
        }
    }

    for (cell, impulse) in terrain_impulse {
        let TerrainCell::Solid { material, .. } = terrain.get_loaded_cell_or_empty(cell) else {
            continue;
        };
        if terrain_fracture_particle(material).is_none() {
            continue;
        }
        let threshold = terrain_break_collision_impulse_threshold(material);
        if impulse >= threshold {
            particles.pending_terrain_fractures.insert(cell);
            if let Some(seed) = terrain_seed.get(&cell).copied() {
                particles.pending_terrain_fracture_seeds.insert(cell, seed);
            }
        }
    }
}

pub(super) fn apply_object_fractures(
    particles: &mut ParticleWorld,
    object_world: &mut ObjectWorld,
) {
    if particles.pending_object_fractures.is_empty() {
        return;
    }

    let mut fracture_plans = Vec::new();
    for object in object_world.objects() {
        if !particles.pending_object_fractures.contains(&object.id) {
            continue;
        }
        let Some(&seed_index) = object.particle_indices.first() else {
            continue;
        };
        if seed_index >= particles.particle_count() {
            continue;
        }
        let source_material = particles.material[seed_index];
        let Some(target_material) =
            solid_break_properties(source_material).and_then(|props| props.fracture_to)
        else {
            continue;
        };
        let mut detached = HashSet::new();
        if let Some(fracture_particles) =
            particles.pending_object_fracture_particles.get(&object.id)
        {
            for &index in &object.particle_indices {
                if fracture_particles.contains(&index) {
                    detached.insert(index);
                }
            }
        }
        if detached.is_empty() {
            detached.insert(seed_index);
        }
        fracture_plans.push((object.id, target_material, detached));
    }

    let mut spawned_particles = Vec::new();
    for (object_id, target_material, detached_particles) in fracture_plans {
        for &index in &detached_particles {
            particles.fracture_particle_to_target_material(
                index,
                target_material,
                &mut spawned_particles,
            );
        }

        object_world.split_object_after_detach(
            object_id,
            &detached_particles,
            particles.positions(),
            particles.masses(),
        );
    }

    for (position, velocity, mass, material) in spawned_particles {
        let new_index = particles.pos.len();
        particles.pos.push(position);
        particles.prev_pos.push(position);
        particles.vel.push(velocity);
        particles.mass.push(mass);
        particles.material.push(material);
        if is_granular_particle(material) {
            particles.request_wake(new_index);
            particles.sleep_lock_frames[new_index] =
                particles.solver_params.fracture_wake_lock_frames;
        }
    }
    particles.resize_work_buffers();

    particles.pending_object_fractures.clear();
    particles.pending_object_fracture_particles.clear();
    particles.auto_fracture_single_cell_objects(object_world);
}

pub(super) fn solve_shape_matching_constraints(
    particles: &mut ParticleWorld,
    object_world: &mut ObjectWorld,
) {
    let _span = tracing::info_span!("physics::shape_matching_pass").entered();
    for object in object_world.objects_mut() {
        if object.particle_indices.is_empty() || object.rest_local.is_empty() {
            continue;
        }
        let alpha = object.shape_stiffness_alpha.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            continue;
        }

        let mut object_peak_strain = 0.0f32;
        let mut object_peak_particle = None;
        let shape_iters = object.shape_iters.max(1);
        for _ in 0..shape_iters {
            let mut mass_sum = 0.0;
            let mut com = Vec2::ZERO;
            for &index in &object.particle_indices {
                if index >= particles.particle_count() || !particles.is_active_particle(index) {
                    continue;
                }
                let mass = particles.mass[index];
                mass_sum += mass;
                com += particles.pos[index] * mass;
            }
            if mass_sum <= 1e-6 {
                continue;
            }
            com /= mass_sum;
            object.mass_sum = mass_sum;

            let mut a00 = 0.0;
            let mut a01 = 0.0;
            let mut a10 = 0.0;
            let mut a11 = 0.0;
            for (slot, &index) in object.particle_indices.iter().enumerate() {
                if index >= particles.particle_count() || slot >= object.rest_local.len() {
                    continue;
                }
                if !particles.is_active_particle(index) {
                    continue;
                }
                let mass = particles.mass[index];
                let p = particles.pos[index] - com;
                let q = object.rest_local[slot];
                a00 += mass * p.x * q.x;
                a01 += mass * p.x * q.y;
                a10 += mass * p.y * q.x;
                a11 += mass * p.y * q.y;
            }

            let theta = (a10 - a01).atan2(a00 + a11);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            for (slot, &index) in object.particle_indices.iter().enumerate() {
                if index >= particles.particle_count() || slot >= object.rest_local.len() {
                    continue;
                }
                if !particles.is_active_particle(index) {
                    continue;
                }
                let q = object.rest_local[slot];
                let rotated = Vec2::new(cos_t * q.x - sin_t * q.y, sin_t * q.x + cos_t * q.y);
                let goal = com + rotated;
                let current = particles.pos[index];
                let strain = (goal - current).length() / CELL_SIZE_M.max(1e-6);
                if strain > object_peak_strain {
                    object_peak_strain = strain;
                    object_peak_particle = Some(index);
                }
                particles.pos[index] = current + (goal - current) * alpha;
            }
        }
        particles
            .object_peak_strain
            .insert(object.id, object_peak_strain);
        if let Some(particle_index) = object_peak_particle {
            particles
                .object_peak_strain_particle
                .insert(object.id, particle_index);
        } else {
            particles.object_peak_strain_particle.remove(&object.id);
        }
    }
}
