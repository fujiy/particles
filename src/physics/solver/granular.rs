use std::collections::HashMap;

use bevy::log::tracing;
use bevy::prelude::Vec2;
use rayon::prelude::*;

use crate::physics::material::particle_properties;
use crate::physics::world::object::{ObjectId, ObjectPhysicsField, ObjectWorld};
use crate::physics::world::particle::ParticleWorld;
use crate::physics::world::particle::helpers::{
    granular_contact_friction_pair_scale, is_granular_particle, is_water_particle, pack_pair_key,
};
use crate::physics::world::terrain::TerrainWorld;

#[derive(Default)]
pub(crate) struct ComputeDeltaThreadScratch {
    pub(crate) object_contacts: Vec<ObjectId>,
    pub(crate) reaction_impulses: HashMap<ObjectId, Vec2>,
}

impl ComputeDeltaThreadScratch {
    pub(crate) fn merge_from(&mut self, other: Self) {
        for (object_id, impulse) in other.reaction_impulses {
            *self
                .reaction_impulses
                .entry(object_id)
                .or_insert(Vec2::ZERO) += impulse;
        }
    }

    pub(crate) fn accumulate_impulse(&mut self, object_id: ObjectId, impulse: Vec2) {
        *self
            .reaction_impulses
            .entry(object_id)
            .or_insert(Vec2::ZERO) += impulse;
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GranularObjectContactKey {
    particle: usize,
    object_id: ObjectId,
}

#[derive(Default)]
struct GranularContactThreadScratch {
    object_contacts: Vec<ObjectId>,
    delta_pos: Vec<(usize, Vec2)>,
    wake_impulses: Vec<(usize, f32)>,
    reaction_impulses: HashMap<ObjectId, Vec2>,
    terrain_lambda_n_updates: Vec<(usize, f32)>,
    terrain_lambda_t_updates: Vec<(usize, f32)>,
    object_lambda_n_updates: Vec<(GranularObjectContactKey, f32)>,
    object_lambda_t_updates: Vec<(GranularObjectContactKey, f32)>,
}

impl GranularContactThreadScratch {
    fn merge_from(&mut self, mut other: Self) {
        self.delta_pos.append(&mut other.delta_pos);
        self.wake_impulses.append(&mut other.wake_impulses);
        self.terrain_lambda_n_updates
            .append(&mut other.terrain_lambda_n_updates);
        self.terrain_lambda_t_updates
            .append(&mut other.terrain_lambda_t_updates);
        self.object_lambda_n_updates
            .append(&mut other.object_lambda_n_updates);
        self.object_lambda_t_updates
            .append(&mut other.object_lambda_t_updates);
        for (object_id, impulse) in other.reaction_impulses {
            *self
                .reaction_impulses
                .entry(object_id)
                .or_insert(Vec2::ZERO) += impulse;
        }
    }
}

pub(crate) fn solve_contacts(
    particles: &mut ParticleWorld,
    terrain: &TerrainWorld,
    object_field: &ObjectPhysicsField,
    object_world: &ObjectWorld,
    dt_sub: f32,
) -> HashMap<ObjectId, Vec2> {
    if particles.solver_params.granular_iters == 0
        || particles.solver_params.granular_substeps == 0
        || particles.particle_count() == 0
    {
        return HashMap::new();
    }

    let mut reaction_impulses = HashMap::<ObjectId, Vec2>::new();
    let n = particles.particle_count();
    let use_parallel_contacts =
        particles.parallel_enabled && n >= particles.solver_params.parallel_particle_threshold;
    let granular_substeps = particles.solver_params.granular_substeps.max(1);
    let dt_granular = dt_sub / granular_substeps as f32;
    let inv_dt_granular = 1.0 / dt_granular.max(1e-6);
    let alpha_n = particles.solver_params.xpbd_contact_compliance_n / dt_granular.max(1e-6).powi(2);
    let alpha_t = particles.solver_params.xpbd_contact_compliance_t / dt_granular.max(1e-6).powi(2);

    let mut lambda_n_pair = HashMap::<u64, f32>::new();
    let mut lambda_t_pair = HashMap::<u64, f32>::new();
    let mut lambda_n_terrain = vec![0.0f32; n];
    let mut lambda_t_terrain = vec![0.0f32; n];
    let mut lambda_n_object = HashMap::<GranularObjectContactKey, f32>::new();
    let mut lambda_t_object = HashMap::<GranularObjectContactKey, f32>::new();
    let mut wake_impulses = vec![0.0f32; n];

    for substep in 0..granular_substeps {
        {
            let _span = tracing::info_span!("physics::granular_substep_rebuild", substep).entered();
            particles.neighbor_grid.rebuild(&particles.pos);
            particles.rebuild_neighbor_cache(use_parallel_contacts);
            particles.mark_neighbor_cache_anchor();
        }

        for iter in 0..particles.solver_params.granular_iters {
            let mut delta_pos = vec![Vec2::ZERO; n];
            wake_impulses.fill(0.0);

            {
                let _span =
                    tracing::info_span!("physics::granular_pair_contacts", substep, iter).entered();
                for i in 0..n {
                    let material_i = particles.material[i];
                    if !particles.is_active_particle(i) || is_water_particle(material_i) {
                        continue;
                    }
                    if is_granular_particle(material_i)
                        && !particles.is_particle_scheduled_in_sub_block(i)
                    {
                        continue;
                    }

                    let props_i = particle_properties(material_i);
                    let inv_mass_i = 1.0 / particles.mass[i].max(1e-6);

                    for &j in &particles.neighbor_cache[i] {
                        let material_j = particles.material[j];
                        if j <= i
                            || !particles.is_active_particle(j)
                            || is_water_particle(material_j)
                        {
                            continue;
                        }
                        if is_granular_particle(material_j)
                            && !particles.is_particle_scheduled_in_sub_block(j)
                        {
                            continue;
                        }

                        let props_j = particle_properties(material_j);
                        let r = particles.pos[i] - particles.pos[j];
                        let dist2 = r.length_squared();
                        if dist2 <= 1e-12 {
                            continue;
                        }
                        let contact_radius = props_i.radius_m + props_j.radius_m;
                        if dist2 >= contact_radius * contact_radius {
                            continue;
                        }

                        let dist = dist2.sqrt();
                        let n_ij = r / dist;
                        let c_n = dist - contact_radius;
                        let inv_mass_j = 1.0 / particles.mass[j].max(1e-6);
                        let w_sum = inv_mass_i + inv_mass_j;
                        if w_sum <= 1e-8 {
                            continue;
                        }

                        let pair_key = pack_pair_key(i, j);
                        let lambda_n = lambda_n_pair.entry(pair_key).or_insert(0.0);
                        let delta_lambda_n = (-c_n - alpha_n * *lambda_n) / (w_sum + alpha_n);
                        *lambda_n += delta_lambda_n;
                        let normal_correction = n_ij * delta_lambda_n;
                        let corr_i_n = normal_correction * inv_mass_i;
                        let corr_j_n = -normal_correction * inv_mass_j;
                        delta_pos[i] += corr_i_n;
                        delta_pos[j] += corr_j_n;

                        if let Some(scale) = granular_contact_friction_pair_scale(
                            material_i,
                            material_j,
                        ) {
                            let mu_k =
                                0.5 * (props_i.friction_dynamic + props_j.friction_dynamic) * scale;
                            let rel_vel = particles.vel[i] - particles.vel[j];
                            let tangent = rel_vel - n_ij * rel_vel.dot(n_ij);
                            let tangent_len = tangent.length();
                            if tangent_len > 1e-6 {
                                let t_hat = tangent / tangent_len;
                                let c_t = rel_vel.dot(t_hat) * dt_granular;
                                let lambda_t = lambda_t_pair.entry(pair_key).or_insert(0.0);
                                let delta_lambda_t =
                                    (-c_t - alpha_t * *lambda_t) / (w_sum + alpha_t);
                                let max_tangent = mu_k.max(0.0) * lambda_n.abs();
                                let next_lambda_t =
                                    (*lambda_t + delta_lambda_t).clamp(-max_tangent, max_tangent);
                                let applied_t = next_lambda_t - *lambda_t;
                                *lambda_t = next_lambda_t;
                                let tangent_correction = t_hat * applied_t;
                                delta_pos[i] += tangent_correction * inv_mass_i;
                                delta_pos[j] += -tangent_correction * inv_mass_j;
                            }
                        }

                        let impulse_i = particles.mass[i] * corr_i_n.length() * inv_dt_granular;
                        let impulse_j = particles.mass[j] * corr_j_n.length() * inv_dt_granular;
                        wake_impulses[i] += impulse_i;
                        wake_impulses[j] += impulse_j;
                    }
                }
            }

            let terrain_object_reduced = {
                let _span =
                    tracing::info_span!("physics::granular_terrain_object_contacts", substep, iter)
                        .entered();
                if use_parallel_contacts {
                    (0..n)
                        .into_par_iter()
                        .fold(GranularContactThreadScratch::default, |mut scratch, i| {
                            accumulate_terrain_object_contacts_for_particle(
                                particles,
                                terrain,
                                object_field,
                                object_world,
                                i,
                                dt_granular,
                                inv_dt_granular,
                                alpha_n,
                                alpha_t,
                                &lambda_n_terrain,
                                &lambda_t_terrain,
                                &lambda_n_object,
                                &lambda_t_object,
                                &mut scratch,
                            );
                            scratch
                        })
                        .reduce(GranularContactThreadScratch::default, |mut a, b| {
                            a.merge_from(b);
                            a
                        })
                } else {
                    let mut scratch = GranularContactThreadScratch::default();
                    for i in 0..n {
                        accumulate_terrain_object_contacts_for_particle(
                            particles,
                            terrain,
                            object_field,
                            object_world,
                            i,
                            dt_granular,
                            inv_dt_granular,
                            alpha_n,
                            alpha_t,
                            &lambda_n_terrain,
                            &lambda_t_terrain,
                            &lambda_n_object,
                            &lambda_t_object,
                            &mut scratch,
                        );
                    }
                    scratch
                }
            };

            {
                let _span =
                    tracing::info_span!("physics::granular_reduce_apply", substep, iter).entered();
                for (i, next_lambda) in terrain_object_reduced.terrain_lambda_n_updates {
                    lambda_n_terrain[i] = next_lambda;
                }
                for (i, next_lambda) in terrain_object_reduced.terrain_lambda_t_updates {
                    lambda_t_terrain[i] = next_lambda;
                }
                for (key, next_lambda) in terrain_object_reduced.object_lambda_n_updates {
                    lambda_n_object.insert(key, next_lambda);
                }
                for (key, next_lambda) in terrain_object_reduced.object_lambda_t_updates {
                    lambda_t_object.insert(key, next_lambda);
                }
                for (i, delta) in terrain_object_reduced.delta_pos {
                    delta_pos[i] += delta;
                }
                for (i, impulse) in terrain_object_reduced.wake_impulses {
                    wake_impulses[i] += impulse;
                }
                for (object_id, impulse) in terrain_object_reduced.reaction_impulses {
                    *reaction_impulses.entry(object_id).or_insert(Vec2::ZERO) += impulse;
                }
            }

            {
                let _span = tracing::info_span!("physics::granular_apply_positions", substep, iter)
                    .entered();
                for i in 0..n {
                    let material = particles.material[i];
                    if !particles.is_active_particle(i) || is_water_particle(material) {
                        continue;
                    }
                    if is_granular_particle(material)
                        && !particles.is_particle_scheduled_in_sub_block(i)
                    {
                        continue;
                    }
                    particles.pos[i] += delta_pos[i];
                    if wake_impulses[i] > particles.solver_params.wake_impulse_threshold {
                        particles.request_wake(i);
                    }
                }
            }

            if iter + 1 < particles.solver_params.granular_iters
                && particles.neighbor_cache_requires_rebuild()
            {
                let _span =
                    tracing::info_span!("physics::granular_iter_rebuild", substep, iter).entered();
                particles.neighbor_grid.rebuild(&particles.pos);
                particles.rebuild_neighbor_cache(use_parallel_contacts);
                particles.mark_neighbor_cache_anchor();
            }
        }
    }

    reaction_impulses
}

#[allow(clippy::too_many_arguments)]
fn accumulate_terrain_object_contacts_for_particle(
    particles: &ParticleWorld,
    terrain: &TerrainWorld,
    object_field: &ObjectPhysicsField,
    object_world: &ObjectWorld,
    i: usize,
    dt_granular: f32,
    inv_dt_granular: f32,
    alpha_n: f32,
    alpha_t: f32,
    lambda_n_terrain: &[f32],
    lambda_t_terrain: &[f32],
    lambda_n_object: &HashMap<GranularObjectContactKey, f32>,
    lambda_t_object: &HashMap<GranularObjectContactKey, f32>,
    scratch: &mut GranularContactThreadScratch,
) {
    let material_i = particles.material[i];
    if !particles.is_active_particle(i) || is_water_particle(material_i) {
        return;
    }
    if is_granular_particle(material_i) && !particles.is_particle_scheduled_in_sub_block(i) {
        return;
    }
    let props_i = particle_properties(material_i);
    let inv_mass_i = 1.0 / particles.mass[i].max(1e-6);

    if let Some((signed_distance, normal)) =
        terrain.sample_signed_distance_and_normal(particles.pos[i])
    {
        let c_n = signed_distance - props_i.terrain_push_radius_m;
        if c_n < 0.0 {
            let key = i;
            let prev_lambda_n = *lambda_n_terrain.get(key).unwrap_or(&0.0);
            let delta_lambda_n = (-c_n - alpha_n * prev_lambda_n) / (inv_mass_i + alpha_n);
            let next_lambda_n = prev_lambda_n + delta_lambda_n;
            scratch.terrain_lambda_n_updates.push((i, next_lambda_n));
            let corr_n = normal * (delta_lambda_n * inv_mass_i);
            scratch.delta_pos.push((i, corr_n));

            let rel_vel = particles.vel[i];
            let tangent = rel_vel - normal * rel_vel.dot(normal);
            let tangent_len = tangent.length();
            if tangent_len > 1e-6 {
                let t_hat = tangent / tangent_len;
                let c_t = rel_vel.dot(t_hat) * dt_granular;
                let prev_lambda_t = *lambda_t_terrain.get(key).unwrap_or(&0.0);
                let delta_lambda_t = (-c_t - alpha_t * prev_lambda_t) / (inv_mass_i + alpha_t);
                let max_tangent = (props_i.friction_dynamic
                    * particles.solver_params.terrain_contact_friction_scale)
                    .max(0.0)
                    * next_lambda_n.abs();
                let next_lambda_t =
                    (prev_lambda_t + delta_lambda_t).clamp(-max_tangent, max_tangent);
                let applied_t = next_lambda_t - prev_lambda_t;
                scratch.terrain_lambda_t_updates.push((i, next_lambda_t));
                scratch
                    .delta_pos
                    .push((i, t_hat * (applied_t * inv_mass_i)));
            }
            scratch
                .wake_impulses
                .push((i, particles.mass[i] * corr_n.length() * inv_dt_granular));
        }
    }

    object_field.gather_candidate_object_ids(particles.pos[i], &mut scratch.object_contacts);
    let mut best_contact: Option<(ObjectId, Vec2, f32)> = None;
    for &object_id in &scratch.object_contacts {
        if object_world.object_of_particle(i) == Some(object_id) {
            continue;
        }
        let Some(sample) = object_world.evaluate_object_sdf(object_id, particles.pos[i]) else {
            continue;
        };
        let c_n = sample.distance_m - props_i.object_push_radius_m;
        if c_n >= 0.0 {
            continue;
        }
        let replace = match best_contact {
            Some((_, _, best_c_n)) => c_n < best_c_n,
            None => true,
        };
        if replace {
            best_contact = Some((object_id, sample.normal_world, c_n));
        }
    }
    scratch.object_contacts.clear();

    if let Some((object_id, normal, c_n)) = best_contact {
        let key = GranularObjectContactKey {
            particle: i,
            object_id,
        };
        let prev_lambda_n = *lambda_n_object.get(&key).unwrap_or(&0.0);
        let delta_lambda_n = (-c_n - alpha_n * prev_lambda_n) / (inv_mass_i + alpha_n);
        let next_lambda_n = prev_lambda_n + delta_lambda_n;
        scratch.object_lambda_n_updates.push((key, next_lambda_n));
        let corr_n = normal * (delta_lambda_n * inv_mass_i);
        scratch.delta_pos.push((i, corr_n));

        let rel_vel = particles.vel[i];
        let tangent = rel_vel - normal * rel_vel.dot(normal);
        let tangent_len = tangent.length();
        if tangent_len > 1e-6 {
            let t_hat = tangent / tangent_len;
            let c_t = rel_vel.dot(t_hat) * dt_granular;
            let prev_lambda_t = *lambda_t_object.get(&key).unwrap_or(&0.0);
            let delta_lambda_t = (-c_t - alpha_t * prev_lambda_t) / (inv_mass_i + alpha_t);
            let max_tangent = props_i.friction_dynamic.max(0.0) * next_lambda_n.abs();
            let next_lambda_t = (prev_lambda_t + delta_lambda_t).clamp(-max_tangent, max_tangent);
            let applied_t = next_lambda_t - prev_lambda_t;
            scratch.object_lambda_t_updates.push((key, next_lambda_t));
            scratch
                .delta_pos
                .push((i, t_hat * (applied_t * inv_mass_i)));
        }

        let reaction_impulse = -(particles.mass[i] * corr_n) * inv_dt_granular;
        *scratch
            .reaction_impulses
            .entry(object_id)
            .or_insert(Vec2::ZERO) += reaction_impulse;
        scratch
            .wake_impulses
            .push((i, particles.mass[i] * corr_n.length() * inv_dt_granular));
    }
}

pub(crate) fn apply_restitution(particles: &mut ParticleWorld, _dt_sub: f32) {
    let mut neighbors = Vec::new();
    let _span = tracing::info_span!("physics::granular_restitution_pairs").entered();
    for i in 0..particles.particle_count() {
        let material_i = particles.material[i];
        if !particles.is_active_particle(i) || is_water_particle(material_i) {
            continue;
        }
        if is_granular_particle(material_i) && !particles.is_particle_scheduled_in_sub_block(i) {
            continue;
        }
        particles
            .neighbor_grid
            .gather(particles.pos[i], &mut neighbors);
        for &j in &neighbors {
            let material_j = particles.material[j];
            if j <= i
                || !particles.is_active_particle(j)
                || is_water_particle(material_j)
            {
                continue;
            }
            if is_granular_particle(material_j) && !particles.is_particle_scheduled_in_sub_block(j) {
                continue;
            }
            let r = particles.pos[i] - particles.pos[j];
            let dist2 = r.length_squared();
            if dist2 <= 1e-12 {
                continue;
            }
            let props_i = particle_properties(material_i);
            let props_j = particle_properties(material_j);
            let contact_radius = props_i.radius_m + props_j.radius_m;
            if dist2 >= contact_radius * contact_radius {
                continue;
            }

            let n_ij = r.normalize_or_zero();
            let rel_normal_speed = (particles.vel[i] - particles.vel[j]).dot(n_ij);
            if rel_normal_speed >= 0.0 {
                continue;
            }
            let restitution = 0.5 * (props_i.contact_restitution + props_j.contact_restitution);
            let inv_mass_i = 1.0 / particles.mass[i].max(1e-6);
            let inv_mass_j = 1.0 / particles.mass[j].max(1e-6);
            let denom = (inv_mass_i + inv_mass_j).max(1e-6);
            let impulse = -(1.0 + restitution.clamp(0.0, 1.0)) * rel_normal_speed / denom;
            let impulse_vec = n_ij * impulse;
            particles.vel[i] += impulse_vec * inv_mass_i;
            particles.vel[j] -= impulse_vec * inv_mass_j;

            if impulse.abs() > particles.solver_params.wake_impulse_threshold {
                particles.request_wake(i);
                particles.request_wake(j);
            }
        }
    }
}
