use std::collections::HashMap;

use bevy::prelude::*;

use super::terrain::{CELL_SIZE_M, TerrainWorld};

pub const GRAVITY_MPS2: Vec2 = Vec2::new(0.0, -9.81);
pub const FIXED_DT: f32 = 1.0 / 60.0;
pub const SUBSTEPS: usize = 2;
pub const SOLVER_ITERS: usize = 6;
pub const PARTICLE_SPACING_M: f32 = 0.125;
pub const PARTICLE_RADIUS_M: f32 = 0.06;
pub const REST_DENSITY: f32 = 1000.0;
pub const EPSILON_LAMBDA: f32 = 1e-6;
pub const XSPH_VISCOSITY: f32 = 0.01;
pub const H_WATER_OVER_DX: f32 = 2.0;
pub const WATER_KERNEL_RADIUS_M: f32 = PARTICLE_SPACING_M * H_WATER_OVER_DX;
pub const WATER_PARTICLE_MASS: f32 = REST_DENSITY * PARTICLE_SPACING_M * PARTICLE_SPACING_M;
pub const TERRAIN_BOUNDARY_RADIUS_M: f32 = PARTICLE_RADIUS_M + CELL_SIZE_M * 0.5;
pub const TERRAIN_SDF_PUSH_RADIUS_M: f32 = PARTICLE_RADIUS_M;
pub const TERRAIN_REPULSION_STIFFNESS: f32 = 0.55;

const INITIAL_WATER_ORIGIN_M: Vec2 = Vec2::new(-1.6, 2.0);
const INITIAL_WATER_COLS: usize = 50;
const INITIAL_WATER_ROWS: usize = 30;
pub const PARTICLE_SPEED_LIMIT_MPS: f32 = 20.0;

pub fn nominal_particle_draw_radius_m() -> f32 {
    (default_particle_mass() / (std::f32::consts::PI * REST_DENSITY)).sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleMaterial {
    Water,
}

#[derive(Debug, Default)]
struct NeighborGrid {
    buckets: HashMap<IVec2, Vec<usize>>,
}

impl NeighborGrid {
    fn rebuild(&mut self, positions: &[Vec2]) {
        self.buckets.clear();
        for (idx, position) in positions.iter().enumerate() {
            let cell = particle_grid_cell(*position);
            self.buckets.entry(cell).or_default().push(idx);
        }
    }

    fn gather(&self, position: Vec2, out_neighbors: &mut Vec<usize>) {
        out_neighbors.clear();
        let center = particle_grid_cell(position);
        for y in (center.y - 1)..=(center.y + 1) {
            for x in (center.x - 1)..=(center.x + 1) {
                let Some(indices) = self.buckets.get(&IVec2::new(x, y)) else {
                    continue;
                };
                out_neighbors.extend(indices.iter().copied());
            }
        }
    }
}

#[derive(Resource, Debug)]
pub struct ParticleWorld {
    pub pos: Vec<Vec2>,
    pub prev_pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    pub mass: Vec<f32>,
    pub material: Vec<ParticleMaterial>,
    pub density: Vec<f32>,
    pub lambda: Vec<f32>,
    pub delta_pos: Vec<Vec2>,
    initial_pos: Vec<Vec2>,
    initial_vel: Vec<Vec2>,
    neighbor_grid: NeighborGrid,
    viscosity_work: Vec<Vec2>,
}

impl Default for ParticleWorld {
    fn default() -> Self {
        let initial_pos = generate_initial_water_positions();
        let initial_vel = vec![Vec2::ZERO; initial_pos.len()];
        let mut world = Self {
            pos: initial_pos.clone(),
            prev_pos: initial_pos.clone(),
            vel: initial_vel.clone(),
            mass: vec![default_particle_mass(); initial_pos.len()],
            material: vec![ParticleMaterial::Water; initial_pos.len()],
            density: Vec::new(),
            lambda: Vec::new(),
            delta_pos: Vec::new(),
            initial_pos,
            initial_vel,
            neighbor_grid: NeighborGrid::default(),
            viscosity_work: Vec::new(),
        };
        world.resize_work_buffers();
        world
    }
}

impl ParticleWorld {
    pub fn reset_to_initial(&mut self) {
        self.pos.clone_from(&self.initial_pos);
        self.prev_pos.clone_from(&self.initial_pos);
        self.vel.clone_from(&self.initial_vel);
        self.mass.fill(default_particle_mass());
        self.material.fill(ParticleMaterial::Water);
        self.resize_work_buffers();
    }

    pub fn particle_count(&self) -> usize {
        self.pos.len()
    }

    pub fn positions(&self) -> &[Vec2] {
        &self.pos
    }

    pub fn masses(&self) -> &[f32] {
        &self.mass
    }

    pub fn add_velocity_in_radius(
        &mut self,
        center: Vec2,
        radius: f32,
        velocity_delta: Vec2,
        velocity_limit_mps: f32,
    ) {
        let radius2 = radius * radius;
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) > radius2 {
                continue;
            }
            self.vel[i] = (self.vel[i] + velocity_delta).clamp_length_max(velocity_limit_mps);
        }
    }

    pub fn spawn_water_particles_along_segment(
        &mut self,
        from: Vec2,
        to: Vec2,
        spacing: f32,
        initial_velocity: Vec2,
        spacing_carry_m: &mut f32,
    ) -> usize {
        if spacing <= 0.0 {
            return 0;
        }

        let segment = to - from;
        let length = segment.length();
        if length < 1e-6 {
            return 0;
        }

        let direction = segment / length;
        let carry = (*spacing_carry_m).clamp(0.0, spacing);
        let total = carry + length;
        let spawn_count = (total / spacing).floor() as usize;
        if spawn_count == 0 {
            *spacing_carry_m = total;
            return 0;
        }

        let first_offset = spacing - carry;
        for i in 0..spawn_count {
            let offset = first_offset + i as f32 * spacing;
            let pos = from + direction * offset;
            self.push_water_particle(pos, initial_velocity);
        }

        *spacing_carry_m = total - spawn_count as f32 * spacing;
        self.resize_work_buffers();
        spawn_count
    }

    pub fn remove_particles_in_radius(&mut self, center: Vec2, radius: f32) -> usize {
        let radius2 = radius * radius;
        let mut removed = 0;
        for i in (0..self.particle_count()).rev() {
            if self.pos[i].distance_squared(center) > radius2 {
                continue;
            }
            self.swap_remove_particle(i);
            removed += 1;
        }

        self.resize_work_buffers();
        removed
    }

    pub fn step_if_running(&mut self, terrain: &TerrainWorld, running: bool) {
        if running {
            self.step_substeps(terrain);
        }
    }

    pub fn step_substeps(&mut self, terrain: &TerrainWorld) {
        let dt_sub = FIXED_DT / SUBSTEPS as f32;
        for _ in 0..SUBSTEPS {
            self.step_single_substep(terrain, dt_sub);
        }
    }

    fn step_single_substep(&mut self, terrain: &TerrainWorld, dt_sub: f32) {
        for i in 0..self.particle_count() {
            self.prev_pos[i] = self.pos[i];
            self.vel[i] += GRAVITY_MPS2 * dt_sub;
            self.pos[i] += self.vel[i] * dt_sub;
        }

        self.neighbor_grid.rebuild(&self.pos);
        for _ in 0..SOLVER_ITERS {
            self.solve_density_constraints(terrain);
        }

        for i in 0..self.particle_count() {
            self.vel[i] = (self.pos[i] - self.prev_pos[i]) / dt_sub;
            self.vel[i] = self.vel[i].clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
        }

        self.apply_xsph_viscosity();
        for vel in &mut self.vel {
            *vel = vel.clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
        }
    }

    fn solve_density_constraints(&mut self, terrain: &TerrainWorld) {
        let mut neighbors = Vec::new();
        let h_ww = WATER_KERNEL_RADIUS_M;

        for i in 0..self.particle_count() {
            self.neighbor_grid.gather(self.pos[i], &mut neighbors);

            let mut rho = 0.0;
            let mut grad_i = Vec2::ZERO;
            let mut grad_sum_sq = 0.0;

            for &j in &neighbors {
                let r = self.pos[i] - self.pos[j];
                let r2 = r.length_squared();
                if r2 >= h_ww * h_ww {
                    continue;
                }

                rho += self.mass[j] * kernel_poly6(r2, h_ww);

                if i == j {
                    continue;
                }

                let grad = (self.mass[j] / REST_DENSITY) * kernel_spiky_grad(r, h_ww);
                grad_sum_sq += grad.length_squared();
                grad_i += grad;
            }

            grad_sum_sq += grad_i.length_squared();
            let c_i = rho / REST_DENSITY - 1.0;

            self.density[i] = rho;
            self.lambda[i] = -c_i / (grad_sum_sq + EPSILON_LAMBDA);
        }

        for i in 0..self.particle_count() {
            self.neighbor_grid.gather(self.pos[i], &mut neighbors);

            let mut delta = Vec2::ZERO;
            let mut boundary_push = Vec2::ZERO;
            for &j in &neighbors {
                if i == j {
                    continue;
                }
                let r = self.pos[i] - self.pos[j];
                if r.length_squared() >= h_ww * h_ww {
                    continue;
                }
                delta += (self.lambda[i] + self.lambda[j]) * kernel_spiky_grad(r, h_ww);
            }

            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            {
                let penetration = TERRAIN_SDF_PUSH_RADIUS_M - signed_distance;
                if penetration > 0.0 {
                    boundary_push += normal * penetration * TERRAIN_REPULSION_STIFFNESS;
                }
            }
            self.delta_pos[i] = delta / REST_DENSITY + boundary_push;
        }

        for i in 0..self.particle_count() {
            self.pos[i] += self.delta_pos[i];
        }
    }

    fn apply_xsph_viscosity(&mut self) {
        let mut neighbors = Vec::new();
        self.viscosity_work.clone_from(&self.vel);

        for i in 0..self.particle_count() {
            self.neighbor_grid.gather(self.pos[i], &mut neighbors);

            let mut correction = Vec2::ZERO;
            for &j in &neighbors {
                if i == j {
                    continue;
                }
                let r = self.pos[i] - self.pos[j];
                let w = kernel_poly6(r.length_squared(), WATER_KERNEL_RADIUS_M);
                correction += (self.vel[j] - self.vel[i]) * w;
            }
            self.viscosity_work[i] = self.vel[i] + XSPH_VISCOSITY * correction;
        }

        self.vel.clone_from(&self.viscosity_work);
    }

    fn push_water_particle(&mut self, position: Vec2, velocity: Vec2) {
        self.pos.push(position);
        self.prev_pos.push(position);
        self.vel.push(velocity);
        self.mass.push(default_particle_mass());
        self.material.push(ParticleMaterial::Water);
    }

    fn swap_remove_particle(&mut self, index: usize) {
        self.pos.swap_remove(index);
        self.prev_pos.swap_remove(index);
        self.vel.swap_remove(index);
        self.mass.swap_remove(index);
        self.material.swap_remove(index);
        self.density.swap_remove(index);
        self.lambda.swap_remove(index);
        self.delta_pos.swap_remove(index);
        self.viscosity_work.swap_remove(index);
    }

    fn resize_work_buffers(&mut self) {
        let count = self.particle_count();
        self.prev_pos.resize(count, Vec2::ZERO);
        self.vel.resize(count, Vec2::ZERO);
        self.mass.resize(count, default_particle_mass());
        self.material.resize(count, ParticleMaterial::Water);
        self.density.resize(count, 0.0);
        self.lambda.resize(count, 0.0);
        self.delta_pos.resize(count, Vec2::ZERO);
        self.viscosity_work.resize(count, Vec2::ZERO);
    }
}

fn generate_initial_water_positions() -> Vec<Vec2> {
    let mut positions = Vec::with_capacity(INITIAL_WATER_COLS * INITIAL_WATER_ROWS);
    for row in 0..INITIAL_WATER_ROWS {
        for col in 0..INITIAL_WATER_COLS {
            positions.push(
                INITIAL_WATER_ORIGIN_M
                    + Vec2::new(col as f32, row as f32) * PARTICLE_SPACING_M
                    + Vec2::splat(PARTICLE_RADIUS_M),
            );
        }
    }
    positions
}

fn default_particle_mass() -> f32 {
    WATER_PARTICLE_MASS
}

fn particle_grid_cell(position: Vec2) -> IVec2 {
    IVec2::new(
        (position.x / WATER_KERNEL_RADIUS_M).floor() as i32,
        (position.y / WATER_KERNEL_RADIUS_M).floor() as i32,
    )
}

fn kernel_poly6(r2: f32, support_radius: f32) -> f32 {
    let h2 = support_radius * support_radius;
    if !(0.0..h2).contains(&r2) {
        return 0.0;
    }
    let x = h2 - r2;
    let coeff = 4.0 / (std::f32::consts::PI * h2.powi(4));
    coeff * x.powi(3)
}

fn kernel_spiky_grad(r: Vec2, support_radius: f32) -> Vec2 {
    let len = r.length();
    if !(1e-6..support_radius).contains(&len) {
        return Vec2::ZERO;
    }
    let coeff = -30.0 / (std::f32::consts::PI * support_radius.powi(5));
    r * (coeff * (support_radius - len).powi(2) / len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::terrain::{CHUNK_WORLD_SIZE_M, TerrainWorld, WORLD_MIN_CHUNK_X};

    #[test]
    fn particles_fall_under_gravity() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let before_avg_y = particles.positions().iter().map(|p| p.y).sum::<f32>()
            / particles.positions().len() as f32;

        for _ in 0..10 {
            particles.step_if_running(&terrain, true);
        }
        let after_avg_y = particles.positions().iter().map(|p| p.y).sum::<f32>()
            / particles.positions().len() as f32;

        assert!(after_avg_y < before_avg_y - 0.01);
    }

    #[test]
    fn static_terrain_boundary_push_affects_motion() {
        let mut terrain_static = TerrainWorld::default();
        terrain_static.reset_fixed_world();
        terrain_static.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let terrain_empty = TerrainWorld::default();

        let mut with_terrain = ParticleWorld::default();
        with_terrain.pos = vec![Vec2::new(0.1, 0.6)];
        with_terrain.prev_pos = with_terrain.pos.clone();
        with_terrain.vel = vec![Vec2::new(0.0, -8.0)];
        with_terrain.mass = vec![default_particle_mass()];
        with_terrain.material = vec![ParticleMaterial::Water];
        with_terrain.resize_work_buffers();

        let mut without_terrain = ParticleWorld::default();
        without_terrain.pos = with_terrain.pos.clone();
        without_terrain.prev_pos = without_terrain.pos.clone();
        without_terrain.vel = with_terrain.vel.clone();
        without_terrain.mass = with_terrain.mass.clone();
        without_terrain.material = with_terrain.material.clone();
        without_terrain.resize_work_buffers();

        for _ in 0..12 {
            with_terrain.step_if_running(&terrain_static, true);
            without_terrain.step_if_running(&terrain_empty, true);
        }

        let y_with = with_terrain.pos[0].y;
        let y_without = without_terrain.pos[0].y;
        assert!(
            y_with > y_without + 0.2,
            "boundary push did not keep particle away from terrain (with={y_with}, without={y_without})"
        );
    }

    #[test]
    fn particles_do_not_penetrate_frozen_terrain() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        particles.pos = vec![Vec2::new(0.1, 0.6)];
        particles.prev_pos = particles.pos.clone();
        particles.vel = vec![Vec2::new(0.0, -8.0)];
        particles.mass = vec![default_particle_mass()];
        particles.material = vec![ParticleMaterial::Water];
        particles.resize_work_buffers();

        for _ in 0..12 {
            particles.step_if_running(&terrain, true);
        }

        let p = particles.pos[0];
        assert!(
            p.y > -0.25,
            "particle tunneled too deep into terrain: y={}",
            p.y
        );
    }

    #[test]
    fn paused_simulation_keeps_particle_positions() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let before = particles.pos.clone();

        particles.step_if_running(&terrain, false);

        assert_eq!(before, particles.pos);
    }

    #[test]
    fn reset_restores_initial_configuration() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let initial_pos = particles.pos.clone();
        let initial_vel = particles.vel.clone();

        particles.step_if_running(&terrain, true);
        particles.reset_to_initial();

        assert_eq!(particles.pos, initial_pos);
        assert_eq!(particles.vel, initial_vel);
    }

    #[test]
    fn boundary_overcrowding_remains_stable() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let left_wall_x = WORLD_MIN_CHUNK_X as f32 * CHUNK_WORLD_SIZE_M;

        for (i, pos) in particles.pos.iter_mut().enumerate() {
            let row = (i / 20) as f32;
            let col = (i % 20) as f32;
            *pos = Vec2::new(left_wall_x + 0.25 + col * 0.03, 0.4 + row * 0.03);
        }
        particles.prev_pos.clone_from(&particles.pos);
        particles.vel.fill(Vec2::ZERO);

        for _ in 0..30 {
            particles.step_if_running(&terrain, true);
        }

        for pos in &particles.pos {
            assert!(pos.is_finite());
        }
        for vel in &particles.vel {
            assert!(vel.is_finite());
            assert!(vel.length() <= PARTICLE_SPEED_LIMIT_MPS + 1e-4);
        }
    }
}
