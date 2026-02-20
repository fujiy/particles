use std::collections::HashMap;

use bevy::log::tracing;
use bevy::prelude::*;

use super::object::{
    OBJECT_BOUNDARY_PUSH_RADIUS_M, OBJECT_REPULSION_STIFFNESS, ObjectId, ObjectPhysicsField,
    ObjectWorld,
};
use super::terrain::{CELL_SIZE_M, TerrainWorld, cell_to_world_center};

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
pub const STONE_PARTICLE_MASS: f32 = REST_DENSITY * CELL_SIZE_M * CELL_SIZE_M;
pub const STONE_PARTICLE_RADIUS_M: f32 = CELL_SIZE_M * 0.5;
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
    Stone,
}

#[derive(Debug, Clone)]
pub struct ParticleRemovalResult {
    pub removed_count: usize,
    pub old_to_new: Vec<Option<usize>>,
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
    object_contacts_work: Vec<ObjectId>,
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
            object_contacts_work: Vec::new(),
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

    pub fn materials(&self) -> &[ParticleMaterial] {
        &self.material
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

    pub fn spawn_stone_particles_from_cells(
        &mut self,
        cells: &[IVec2],
        initial_velocity: Vec2,
    ) -> Vec<usize> {
        let mut indices = Vec::with_capacity(cells.len());
        for &cell in cells {
            let position = cell_to_world_center(cell);
            let index = self.pos.len();
            self.pos.push(position);
            self.prev_pos.push(position);
            self.vel.push(initial_velocity);
            self.mass.push(STONE_PARTICLE_MASS);
            self.material.push(ParticleMaterial::Stone);
            indices.push(index);
        }
        self.resize_work_buffers();
        indices
    }

    pub fn remove_particles_in_radius_with_map(
        &mut self,
        center: Vec2,
        radius: f32,
    ) -> ParticleRemovalResult {
        let old_count = self.particle_count();
        let radius2 = radius * radius;
        let mut keep = vec![true; old_count];
        let mut removed_count = 0;
        for (index, position) in self.pos.iter().enumerate() {
            if position.distance_squared(center) <= radius2 {
                keep[index] = false;
                removed_count += 1;
            }
        }

        if removed_count == 0 {
            return ParticleRemovalResult {
                removed_count,
                old_to_new: (0..old_count).map(Some).collect(),
            };
        }

        let new_count = old_count - removed_count;
        let mut old_to_new = vec![None; old_count];
        let mut next = 0usize;
        for old_index in 0..old_count {
            if !keep[old_index] {
                continue;
            }
            old_to_new[old_index] = Some(next);
            next += 1;
        }

        let mut new_pos = Vec::with_capacity(new_count);
        let mut new_prev_pos = Vec::with_capacity(new_count);
        let mut new_vel = Vec::with_capacity(new_count);
        let mut new_mass = Vec::with_capacity(new_count);
        let mut new_material = Vec::with_capacity(new_count);
        for old_index in 0..old_count {
            if !keep[old_index] {
                continue;
            }
            new_pos.push(self.pos[old_index]);
            new_prev_pos.push(self.prev_pos[old_index]);
            new_vel.push(self.vel[old_index]);
            new_mass.push(self.mass[old_index]);
            new_material.push(self.material[old_index]);
        }

        self.pos = new_pos;
        self.prev_pos = new_prev_pos;
        self.vel = new_vel;
        self.mass = new_mass;
        self.material = new_material;
        self.resize_work_buffers();

        ParticleRemovalResult {
            removed_count,
            old_to_new,
        }
    }

    pub fn step_if_running(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
        running: bool,
    ) {
        if running {
            self.step_substeps(terrain, object_field, object_world);
        }
    }

    pub fn step_substeps(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
    ) {
        let _span = tracing::info_span!("physics::step_substeps").entered();
        let dt_sub = FIXED_DT / SUBSTEPS as f32;
        for substep in 0..SUBSTEPS {
            let _substep_span = tracing::info_span!("physics::substep", substep).entered();
            self.step_single_substep(terrain, object_field, object_world, dt_sub);
        }
    }

    fn step_single_substep(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
        dt_sub: f32,
    ) {
        {
            let _span = tracing::info_span!("physics::clear_reaction_impulses").entered();
            object_world.clear_reaction_impulses();
        }
        {
            let _span = tracing::info_span!("physics::predict_positions").entered();
            for i in 0..self.particle_count() {
                self.prev_pos[i] = self.pos[i];
                self.vel[i] += GRAVITY_MPS2 * dt_sub;
                self.pos[i] += self.vel[i] * dt_sub;
            }
        }

        {
            let _span = tracing::info_span!("physics::rebuild_neighbor_grid").entered();
            self.neighbor_grid.rebuild(&self.pos);
        }
        for iter in 0..SOLVER_ITERS {
            let _iter_span =
                tracing::info_span!("physics::solve_density_constraints", iter).entered();
            self.solve_density_constraints(terrain, object_field, object_world, dt_sub);
        }
        {
            let _span = tracing::info_span!("physics::shape_matching").entered();
            self.solve_shape_matching_constraints(object_world);
        }

        {
            let _span = tracing::info_span!("physics::update_velocity").entered();
            for i in 0..self.particle_count() {
                self.vel[i] = (self.pos[i] - self.prev_pos[i]) / dt_sub;
                self.vel[i] = self.vel[i].clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
            }
        }

        {
            let _span = tracing::info_span!("physics::xsph_viscosity").entered();
            self.apply_xsph_viscosity();
        }
        {
            let _span = tracing::info_span!("physics::apply_object_reaction").entered();
            self.apply_object_reaction_impulses(object_world);
        }
        {
            let _span = tracing::info_span!("physics::final_velocity_clamp").entered();
            for vel in &mut self.vel {
                *vel = vel.clamp_length_max(PARTICLE_SPEED_LIMIT_MPS);
            }
        }
    }

    fn solve_density_constraints(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
        dt_sub: f32,
    ) {
        let _span = tracing::info_span!("physics::density_constraint_pass").entered();
        let mut neighbors = Vec::new();
        let h_ww = WATER_KERNEL_RADIUS_M;
        let inv_dt = 1.0 / dt_sub.max(1e-6);

        {
            let _span = tracing::info_span!("physics::density_lambda").entered();
            for i in 0..self.particle_count() {
                if !is_water_particle(self.material[i]) {
                    self.density[i] = REST_DENSITY;
                    self.lambda[i] = 0.0;
                    continue;
                }
                self.neighbor_grid.gather(self.pos[i], &mut neighbors);

                let mut rho = 0.0;
                let mut grad_i = Vec2::ZERO;
                let mut grad_sum_sq = 0.0;

                for &j in &neighbors {
                    if !is_water_particle(self.material[j]) {
                        continue;
                    }
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
        }

        {
            let _span = tracing::info_span!("physics::compute_delta").entered();
            for i in 0..self.particle_count() {
                let is_water = is_water_particle(self.material[i]);
                if is_water {
                    self.neighbor_grid.gather(self.pos[i], &mut neighbors);
                } else {
                    neighbors.clear();
                }

                let mut delta = Vec2::ZERO;
                let mut boundary_push = Vec2::ZERO;
                if is_water {
                    for &j in &neighbors {
                        if i == j || !is_water_particle(self.material[j]) {
                            continue;
                        }
                        let r = self.pos[i] - self.pos[j];
                        if r.length_squared() >= h_ww * h_ww {
                            continue;
                        }
                        delta += (self.lambda[i] + self.lambda[j]) * kernel_spiky_grad(r, h_ww);
                    }
                }

                if let Some((signed_distance, normal)) =
                    terrain.sample_signed_distance_and_normal(self.pos[i])
                {
                    let push_radius = if is_water {
                        TERRAIN_SDF_PUSH_RADIUS_M
                    } else {
                        STONE_PARTICLE_RADIUS_M
                    };
                    let penetration = push_radius - signed_distance;
                    if penetration > 0.0 {
                        boundary_push += normal * penetration * TERRAIN_REPULSION_STIFFNESS;
                    }
                }
                let mut object_push = Vec2::ZERO;
                let self_object = object_world.object_of_particle(i);
                if is_water || self_object.is_some() {
                    object_field
                        .gather_candidate_object_ids(self.pos[i], &mut self.object_contacts_work);
                    let push_radius = if is_water {
                        OBJECT_BOUNDARY_PUSH_RADIUS_M
                    } else {
                        STONE_PARTICLE_RADIUS_M
                    };

                    for &object_id in &self.object_contacts_work {
                        if self_object == Some(object_id) {
                            continue;
                        }
                        let Some(sample) = object_world.evaluate_object_sdf(object_id, self.pos[i])
                        else {
                            continue;
                        };
                        let penetration = push_radius - sample.distance_m;
                        if penetration <= 0.0 {
                            continue;
                        }
                        let push = sample.normal_world * penetration * OBJECT_REPULSION_STIFFNESS;
                        object_push += push;
                        let reaction_impulse = -(self.mass[i] * push) * inv_dt;
                        object_world.accumulate_reaction_impulse(object_id, reaction_impulse);
                    }
                }
                let max_push = if is_water {
                    OBJECT_BOUNDARY_PUSH_RADIUS_M
                } else {
                    STONE_PARTICLE_RADIUS_M
                };
                object_push = object_push.clamp_length_max(max_push);

                self.delta_pos[i] = if is_water {
                    delta / REST_DENSITY + boundary_push + object_push
                } else {
                    boundary_push + object_push
                };
            }
        }

        {
            let _span = tracing::info_span!("physics::apply_delta").entered();
            for i in 0..self.particle_count() {
                self.pos[i] += self.delta_pos[i];
            }
        }
    }

    fn apply_xsph_viscosity(&mut self) {
        let _span = tracing::info_span!("physics::xsph_pass").entered();
        let mut neighbors = Vec::new();
        self.viscosity_work.clone_from(&self.vel);

        for i in 0..self.particle_count() {
            if !is_water_particle(self.material[i]) {
                self.viscosity_work[i] = self.vel[i];
                continue;
            }
            self.neighbor_grid.gather(self.pos[i], &mut neighbors);

            let mut correction = Vec2::ZERO;
            for &j in &neighbors {
                if i == j || !is_water_particle(self.material[j]) {
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

    fn apply_object_reaction_impulses(&mut self, object_world: &ObjectWorld) {
        let _span = tracing::info_span!("physics::object_reaction_pass").entered();
        for object in object_world.objects() {
            let impulse = object_world.reaction_impulse_of(object.id);
            if impulse.length_squared() <= 1e-12 {
                continue;
            }

            let mass_sum = object
                .particle_indices
                .iter()
                .filter_map(|&index| self.mass.get(index).copied())
                .sum::<f32>()
                .max(1e-6);
            let dv = impulse / mass_sum;
            for &index in &object.particle_indices {
                if index >= self.particle_count() {
                    continue;
                }
                self.vel[index] += dv;
            }
        }
    }

    fn solve_shape_matching_constraints(&mut self, object_world: &mut ObjectWorld) {
        let _span = tracing::info_span!("physics::shape_matching_pass").entered();
        for object in object_world.objects_mut() {
            if object.particle_indices.is_empty() || object.rest_local.is_empty() {
                continue;
            }
            let alpha = object.shape_stiffness_alpha.clamp(0.0, 1.0);
            if alpha <= 0.0 {
                continue;
            }

            let shape_iters = object.shape_iters.max(1);
            for _ in 0..shape_iters {
                let mut mass_sum = 0.0;
                let mut com = Vec2::ZERO;
                for &index in &object.particle_indices {
                    if index >= self.particle_count() {
                        continue;
                    }
                    let mass = self.mass[index];
                    mass_sum += mass;
                    com += self.pos[index] * mass;
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
                    if index >= self.particle_count() || slot >= object.rest_local.len() {
                        continue;
                    }
                    let mass = self.mass[index];
                    let p = self.pos[index] - com;
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
                    if index >= self.particle_count() || slot >= object.rest_local.len() {
                        continue;
                    }
                    let q = object.rest_local[slot];
                    let rotated = Vec2::new(cos_t * q.x - sin_t * q.y, sin_t * q.x + cos_t * q.y);
                    let goal = com + rotated;
                    let current = self.pos[index];
                    self.pos[index] = current + (goal - current) * alpha;
                }
            }
        }
    }

    fn push_water_particle(&mut self, position: Vec2, velocity: Vec2) {
        self.pos.push(position);
        self.prev_pos.push(position);
        self.vel.push(velocity);
        self.mass.push(default_particle_mass());
        self.material.push(ParticleMaterial::Water);
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
        self.object_contacts_work.clear();
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

fn is_water_particle(material: ParticleMaterial) -> bool {
    matches!(material, ParticleMaterial::Water)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::object::{OBJECT_SHAPE_ITERS, OBJECT_SHAPE_STIFFNESS_ALPHA};
    use crate::physics::terrain::{CHUNK_WORLD_SIZE_M, TerrainWorld, WORLD_MIN_CHUNK_X};

    #[test]
    fn particles_fall_under_gravity() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();
        let before_avg_y = particles.positions().iter().map(|p| p.y).sum::<f32>()
            / particles.positions().len() as f32;

        for _ in 0..10 {
            particles.step_if_running(&terrain, &object_field, &mut object_world, true);
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
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();

        for _ in 0..12 {
            with_terrain.step_if_running(&terrain_static, &object_field, &mut object_world, true);
            without_terrain.step_if_running(&terrain_empty, &object_field, &mut object_world, true);
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
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();

        for _ in 0..12 {
            particles.step_if_running(&terrain, &object_field, &mut object_world, true);
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
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();
        let before = particles.pos.clone();

        particles.step_if_running(&terrain, &object_field, &mut object_world, false);

        assert_eq!(before, particles.pos);
    }

    #[test]
    fn reset_restores_initial_configuration() {
        let mut terrain = TerrainWorld::default();
        terrain.reset_fixed_world();
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        let mut particles = ParticleWorld::default();
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();
        let initial_pos = particles.pos.clone();
        let initial_vel = particles.vel.clone();

        particles.step_if_running(&terrain, &object_field, &mut object_world, true);
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
        let mut object_world = ObjectWorld::default();
        let object_field = ObjectPhysicsField::default();

        for _ in 0..30 {
            particles.step_if_running(&terrain, &object_field, &mut object_world, true);
        }

        for pos in &particles.pos {
            assert!(pos.is_finite());
        }
        for vel in &particles.vel {
            assert!(vel.is_finite());
            assert!(vel.length() <= PARTICLE_SPEED_LIMIT_MPS + 1e-4);
        }
    }

    #[test]
    fn shape_matching_keeps_stone_object_cohesive() {
        let terrain = TerrainWorld::default();
        let mut particles = ParticleWorld::default();
        particles.pos.clear();
        particles.prev_pos.clear();
        particles.vel.clear();
        particles.mass.clear();
        particles.material.clear();
        particles.density.clear();
        particles.lambda.clear();
        particles.delta_pos.clear();
        particles.viscosity_work.clear();
        particles.resize_work_buffers();

        let cells = [IVec2::new(0, 6), IVec2::new(1, 6), IVec2::new(2, 6)];
        let indices = particles.spawn_stone_particles_from_cells(&cells, Vec2::ZERO);
        let initial_span = particles.pos[indices[2]].x - particles.pos[indices[0]].x;

        let mut objects = ObjectWorld::default();
        let _ = objects.create_object(
            indices,
            particles.positions(),
            particles.masses(),
            OBJECT_SHAPE_STIFFNESS_ALPHA,
            OBJECT_SHAPE_ITERS,
        );
        let object_field = ObjectPhysicsField::default();

        for _ in 0..30 {
            particles.step_if_running(&terrain, &object_field, &mut objects, true);
        }

        let span = particles.pos[2].x - particles.pos[0].x;
        assert!(
            (span - initial_span).abs() < 0.05,
            "shape matching drifted too much: initial={initial_span}, current={span}"
        );
    }

    #[test]
    fn water_object_contact_pushes_object_motion() {
        let terrain = TerrainWorld::default();
        let stone_cell = IVec2::new(0, 6);
        let water_pos = cell_to_world_center(stone_cell);

        let mut with_water = ParticleWorld::default();
        with_water.pos.clear();
        with_water.prev_pos.clear();
        with_water.vel.clear();
        with_water.mass.clear();
        with_water.material.clear();
        with_water.density.clear();
        with_water.lambda.clear();
        with_water.delta_pos.clear();
        with_water.viscosity_work.clear();
        with_water.resize_work_buffers();
        let stone_indices = with_water.spawn_stone_particles_from_cells(&[stone_cell], Vec2::ZERO);
        with_water.pos.push(water_pos);
        with_water.prev_pos.push(water_pos);
        with_water.vel.push(Vec2::ZERO);
        with_water.mass.push(default_particle_mass());
        with_water.material.push(ParticleMaterial::Water);
        with_water.resize_work_buffers();

        let mut with_water_objects = ObjectWorld::default();
        let _ = with_water_objects
            .create_object(
                stone_indices,
                with_water.positions(),
                with_water.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            )
            .expect("object should be created");
        let mut with_water_field = ObjectPhysicsField::default();
        with_water_objects.update_physics_field(
            with_water.positions(),
            with_water.masses(),
            &mut with_water_field,
        );
        with_water.step_if_running(&terrain, &with_water_field, &mut with_water_objects, true);

        let mut without_water = ParticleWorld::default();
        without_water.pos.clear();
        without_water.prev_pos.clear();
        without_water.vel.clear();
        without_water.mass.clear();
        without_water.material.clear();
        without_water.density.clear();
        without_water.lambda.clear();
        without_water.delta_pos.clear();
        without_water.viscosity_work.clear();
        without_water.resize_work_buffers();
        let stone_indices_no_water =
            without_water.spawn_stone_particles_from_cells(&[stone_cell], Vec2::ZERO);

        let mut without_water_objects = ObjectWorld::default();
        let _ = without_water_objects
            .create_object(
                stone_indices_no_water,
                without_water.positions(),
                without_water.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            )
            .expect("object should be created");
        let mut without_water_field = ObjectPhysicsField::default();
        without_water_objects.update_physics_field(
            without_water.positions(),
            without_water.masses(),
            &mut without_water_field,
        );
        without_water.step_if_running(
            &terrain,
            &without_water_field,
            &mut without_water_objects,
            true,
        );

        let with_vel = with_water.vel[0];
        let without_vel = without_water.vel[0];
        assert!(
            (with_vel - without_vel).length() > 1e-3,
            "expected water contact to alter object velocity (with={with_vel:?}, without={without_vel:?})"
        );
    }
}
