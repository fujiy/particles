use bevy::prelude::*;

use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::grid::{GridBlock, GridHierarchy};
use crate::physics::world::kernel::evaluate_quadratic_bspline_stencil_2d;
use crate::physics::world::particle::{ParticleMaterial, ParticleWorld};

const GRID_MASS_EPSILON: f32 = 1e-8;
const DET_EPSILON: f32 = 1e-6;

#[derive(Clone, Copy, Debug)]
pub struct MpmWaterParams {
    pub dt: f32,
    pub gravity: Vec2,
    pub rho0: f32,
    pub bulk_modulus: f32,
    pub viscosity: f32,
    pub j_min: f32,
    pub j_max: f32,
    pub active_mass_threshold: f32,
    pub cfl_limit: f32,
}

impl Default for MpmWaterParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            gravity: Vec2::new(0.0, -9.81),
            rho0: 1_000.0,
            bulk_modulus: 2.0e5,
            viscosity: 2.0,
            j_min: 0.6,
            j_max: 1.4,
            active_mass_threshold: 1e-6,
            cfl_limit: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MpmWaterStepMetrics {
    pub particle_mass_sum: f32,
    pub grid_mass_sum: f32,
    pub max_particle_speed_mps: f32,
    pub max_cfl_ratio: f32,
    pub clamped_particle_count: usize,
}

pub fn sound_speed_mps(params: &MpmWaterParams) -> f32 {
    (params.bulk_modulus.max(0.0) / params.rho0.max(DET_EPSILON)).sqrt()
}

pub fn cfl_ratio(params: &MpmWaterParams, h_b: f32, max_particle_speed_mps: f32) -> f32 {
    let h_b = h_b.max(DET_EPSILON);
    let c = sound_speed_mps(params);
    params.dt.max(0.0) * (max_particle_speed_mps.max(0.0) + c) / h_b
}

pub fn cfl_violated(params: &MpmWaterParams, h_b: f32, max_particle_speed_mps: f32) -> bool {
    cfl_ratio(params, h_b, max_particle_speed_mps) > params.cfl_limit
}

pub fn rebuild_continuum_from_particle_world(
    particles: &ParticleWorld,
    continuum: &mut ContinuumParticleWorld,
    params: &MpmWaterParams,
) -> usize {
    continuum.clear();
    let inv_rho0 = 1.0 / params.rho0.max(DET_EPSILON);
    for ((&position, &velocity), (&mass, &material)) in particles
        .positions()
        .iter()
        .zip(particles.vel.iter())
        .zip(particles.masses().iter().zip(particles.materials().iter()))
    {
        if !matches!(material, ParticleMaterial::WaterLiquid) {
            continue;
        }
        continuum.spawn_water_particle(position, velocity, mass, mass.max(0.0) * inv_rho0);
    }
    continuum.len()
}

pub fn sync_continuum_to_particle_world(
    particles: &mut ParticleWorld,
    continuum: &ContinuumParticleWorld,
) -> bool {
    let water_indices: Vec<usize> = particles
        .materials()
        .iter()
        .enumerate()
        .filter_map(|(i, &material)| {
            if matches!(material, ParticleMaterial::WaterLiquid) {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if water_indices.len() != continuum.len() {
        return false;
    }
    for (continuum_index, &particle_index) in water_indices.iter().enumerate() {
        particles.prev_pos[particle_index] = particles.pos[particle_index];
        particles.pos[particle_index] = continuum.x[continuum_index];
        particles.vel[particle_index] = continuum.v[continuum_index];
    }
    true
}

pub fn step_single_rate(
    particles: &mut ContinuumParticleWorld,
    grid: &mut GridHierarchy,
    params: &MpmWaterParams,
) -> MpmWaterStepMetrics {
    let mut metrics = MpmWaterStepMetrics {
        particle_mass_sum: particles.m.iter().copied().sum(),
        ..Default::default()
    };
    let Some(block) = grid.blocks_mut().first_mut() else {
        return metrics;
    };
    if particles.is_empty() {
        block.clear_nodes();
        return metrics;
    }

    p2g(particles, block, params);
    grid_update(block, params);
    g2p(particles, block, params, &mut metrics);

    metrics.grid_mass_sum = block.nodes().iter().map(|n| n.m).sum();
    metrics.max_cfl_ratio = cfl_ratio(params, block.h_b, metrics.max_particle_speed_mps);
    block.rebuild_active_nodes(params.active_mass_threshold.max(0.0));
    metrics
}

fn p2g(particles: &ContinuumParticleWorld, block: &mut GridBlock, params: &MpmWaterParams) {
    block.clear_nodes();
    let dt = params.dt.max(0.0);
    let inv_h = 1.0 / block.h_b.max(DET_EPSILON);

    for i in 0..particles.len() {
        let x_p = particles.x[i];
        let v_p = particles.v[i];
        let m_p = particles.m[i].max(0.0);
        let v0_p = particles.v0[i].max(0.0);
        let c_p = particles.c[i];
        let f_p = particles.f[i];
        let j = f_p.determinant();
        let j_stable = j.clamp(params.j_min, params.j_max).max(DET_EPSILON);
        let pressure = params.bulk_modulus * (j_stable - 1.0);
        let viscous = (c_p + c_p.transpose()) * params.viscosity;
        let stress =
            Mat2::from_cols(Vec2::new(-pressure, 0.0), Vec2::new(0.0, -pressure)) + viscous;

        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        for sample in stencil.samples() {
            let node_world = sample.node.as_vec2() * block.h_b;
            let dpos = node_world - x_p;
            let Some(node) = block.node_mut_by_world(sample.node) else {
                continue;
            };

            let apic_velocity = v_p + c_p * dpos;
            let internal_force = -(stress * sample.grad) * v0_p;
            node.m += sample.weight * m_p;
            node.p += sample.weight * m_p * apic_velocity + dt * internal_force;
        }
    }
}

fn grid_update(block: &mut GridBlock, params: &MpmWaterParams) {
    let dt = params.dt.max(0.0);
    for node in block.nodes_mut().iter_mut() {
        if node.m <= GRID_MASS_EPSILON {
            node.m = 0.0;
            node.p = Vec2::ZERO;
            node.v = Vec2::ZERO;
            continue;
        }
        let inv_m = 1.0 / node.m;
        node.v = node.p * inv_m;
        node.v += dt * params.gravity;
        node.p = node.v * node.m;
    }
}

fn g2p(
    particles: &mut ContinuumParticleWorld,
    block: &GridBlock,
    params: &MpmWaterParams,
    metrics: &mut MpmWaterStepMetrics,
) {
    let dt = params.dt.max(0.0);
    let inv_h = 1.0 / block.h_b.max(DET_EPSILON);
    let apic_scale = 4.0 * inv_h * inv_h;

    for i in 0..particles.len() {
        let x_p = particles.x[i];
        let stencil = evaluate_quadratic_bspline_stencil_2d(x_p, inv_h);
        let mut next_v = Vec2::ZERO;
        let mut next_c = Mat2::ZERO;

        for sample in stencil.samples() {
            let Some(node) = block.node_by_world(sample.node) else {
                continue;
            };
            if node.m <= GRID_MASS_EPSILON {
                continue;
            }

            let node_world = sample.node.as_vec2() * block.h_b;
            let dpos = node_world - x_p;
            next_v += sample.weight * node.v;
            next_c += outer_product(node.v, dpos * (sample.weight * apic_scale));
        }

        particles.v[i] = next_v;
        particles.c[i] = next_c;
        particles.x[i] += dt * next_v;
        particles.f[i] = clamp_f_determinant(
            (Mat2::IDENTITY + next_c * dt) * particles.f[i],
            params.j_min,
            params.j_max,
            &mut metrics.clamped_particle_count,
        );
        metrics.max_particle_speed_mps = metrics.max_particle_speed_mps.max(next_v.length());
    }
}

fn clamp_f_determinant(f: Mat2, j_min: f32, j_max: f32, clamped_counter: &mut usize) -> Mat2 {
    let det = f.determinant();
    let target = det.clamp(j_min, j_max).max(DET_EPSILON);
    if (target - det).abs() <= 1e-6 {
        return f;
    }
    *clamped_counter += 1;
    let scale = (target / det.max(DET_EPSILON)).sqrt();
    f * scale
}

fn outer_product(lhs: Vec2, rhs: Vec2) -> Mat2 {
    Mat2::from_cols(lhs * rhs.x, lhs * rhs.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::world::continuum::ContinuumMaterial;
    use crate::physics::world::grid::GridBlock;
    use crate::physics::world::particle::ParticleWorld;

    fn make_world_with_single_block() -> (ContinuumParticleWorld, GridHierarchy) {
        let mut particles = ContinuumParticleWorld::default();
        let mut grid = GridHierarchy::default();
        grid.add_block(GridBlock::new(
            0,
            0.25,
            1.0 / 60.0,
            IVec2::new(-24, -24),
            UVec2::new(48, 48),
        ));
        particles.spawn_particle(
            Vec2::new(0.25, 0.25),
            Vec2::ZERO,
            1.0,
            1.0 / 1_000.0,
            ContinuumMaterial::Water,
        );
        particles.spawn_particle(
            Vec2::new(0.45, 0.25),
            Vec2::ZERO,
            1.0,
            1.0 / 1_000.0,
            ContinuumMaterial::Water,
        );
        (particles, grid)
    }

    #[test]
    fn p2g_conserves_mass_within_single_block() {
        let (mut particles, mut grid) = make_world_with_single_block();
        let params = MpmWaterParams {
            gravity: Vec2::ZERO,
            ..Default::default()
        };
        let metrics = step_single_rate(&mut particles, &mut grid, &params);
        assert!((metrics.particle_mass_sum - 2.0).abs() < 1e-6);
        assert!((metrics.grid_mass_sum - 2.0).abs() < 1e-4);
    }

    #[test]
    fn static_water_stays_nearly_static_without_external_force() {
        let (mut particles, mut grid) = make_world_with_single_block();
        let params = MpmWaterParams {
            gravity: Vec2::ZERO,
            viscosity: 0.0,
            ..Default::default()
        };
        for _ in 0..20 {
            step_single_rate(&mut particles, &mut grid, &params);
        }
        for &v in &particles.v {
            assert!(v.length() < 1e-4);
        }
    }

    #[test]
    fn g2p_applies_gravity_and_updates_particle_state() {
        let (mut particles, mut grid) = make_world_with_single_block();
        let params = MpmWaterParams {
            gravity: Vec2::new(0.0, -9.81),
            ..Default::default()
        };
        let y_before = particles.x[0].y;
        let metrics = step_single_rate(&mut particles, &mut grid, &params);
        assert!(particles.x[0].y < y_before);
        assert!(metrics.max_particle_speed_mps > 0.0);
    }

    #[test]
    fn deformation_gradient_is_clamped_by_j_bounds() {
        let (mut particles, mut grid) = make_world_with_single_block();
        particles.f[0] = Mat2::from_cols(Vec2::new(8.0, 0.0), Vec2::new(0.0, 8.0));
        let params = MpmWaterParams {
            gravity: Vec2::ZERO,
            j_min: 0.8,
            j_max: 1.2,
            ..Default::default()
        };
        let metrics = step_single_rate(&mut particles, &mut grid, &params);
        let j = particles.f[0].determinant();
        assert!(j >= 0.8 - 1e-4 && j <= 1.2 + 1e-4);
        assert!(metrics.clamped_particle_count >= 1);
    }

    #[test]
    fn cfl_detection_flags_large_dt_or_velocity() {
        let params = MpmWaterParams {
            dt: 0.05,
            bulk_modulus: 2.0e6,
            rho0: 1_000.0,
            cfl_limit: 1.0,
            ..Default::default()
        };
        let ratio = cfl_ratio(&params, 0.25, 10.0);
        assert!(ratio > 1.0);
        assert!(cfl_violated(&params, 0.25, 10.0));
    }

    #[test]
    fn rebuild_continuum_filters_water_particles_only() {
        let mut particles = ParticleWorld::default();
        particles
            .restore_from_snapshot(
                vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(2.0, 0.0),
                ],
                vec![Vec2::ZERO; 3],
                vec![
                    ParticleMaterial::WaterLiquid,
                    ParticleMaterial::StoneSolid,
                    ParticleMaterial::WaterLiquid,
                ],
            )
            .unwrap();
        let mut continuum = ContinuumParticleWorld::default();
        let count = rebuild_continuum_from_particle_world(
            &particles,
            &mut continuum,
            &MpmWaterParams::default(),
        );
        assert_eq!(count, 2);
        assert_eq!(continuum.len(), 2);
        assert_eq!(continuum.x[0], Vec2::new(0.0, 0.0));
        assert_eq!(continuum.x[1], Vec2::new(2.0, 0.0));
    }

    #[test]
    fn sync_continuum_updates_only_water_particles() {
        let mut particles = ParticleWorld::default();
        particles
            .restore_from_snapshot(
                vec![
                    Vec2::new(0.0, 0.0),
                    Vec2::new(1.0, 0.0),
                    Vec2::new(2.0, 0.0),
                ],
                vec![Vec2::ZERO; 3],
                vec![
                    ParticleMaterial::WaterLiquid,
                    ParticleMaterial::StoneSolid,
                    ParticleMaterial::WaterLiquid,
                ],
            )
            .unwrap();

        let mut continuum = ContinuumParticleWorld::default();
        continuum.spawn_water_particle(Vec2::new(5.0, 6.0), Vec2::new(1.0, 2.0), 1.0, 0.001);
        continuum.spawn_water_particle(Vec2::new(7.0, 8.0), Vec2::new(3.0, 4.0), 1.0, 0.001);

        assert!(sync_continuum_to_particle_world(&mut particles, &continuum));
        assert_eq!(particles.pos[0], Vec2::new(5.0, 6.0));
        assert_eq!(particles.vel[0], Vec2::new(1.0, 2.0));
        assert_eq!(particles.pos[1], Vec2::new(1.0, 0.0));
        assert_eq!(particles.pos[2], Vec2::new(7.0, 8.0));
        assert_eq!(particles.vel[2], Vec2::new(3.0, 4.0));
    }
}
