use bevy::prelude::*;
use std::collections::HashSet;

use crate::physics::material::{
    MaterialParams, particle_properties, particles_per_cell, solid_break_properties,
};
use crate::physics::solver::params_defaults::DEFAULT_SOLVER_PARAMS;
use crate::physics::solver::params_types::SolverParams;
use crate::physics::world::object::ObjectWorld;
use crate::physics::world::terrain::{CELL_SIZE_M, CHUNK_SIZE_I32, TerrainWorld, cell_to_world_center};

pub use crate::physics::material::ParticleMaterial;

const WAKE_RADIUS: f32 = DEFAULT_SOLVER_PARAMS.wake_radius_m;
const ACTIVE_MIN_FRAMES: u16 = DEFAULT_SOLVER_PARAMS.active_min_frames;

#[derive(Debug, Clone)]
pub struct ParticleRemovalResult {
    pub removed_count: usize,
    pub old_to_new: Vec<Option<usize>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleActivityState {
    Active,
    Sleeping,
}

#[derive(Resource, Debug)]
pub struct ParticleWorld {
    pub pos: Vec<Vec2>,
    pub prev_pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    pub mass: Vec<f32>,
    pub material: Vec<ParticleMaterial>,
    activity_state: Vec<ParticleActivityState>,
    sleep_candidate_frames: Vec<u16>,
    active_hold_frames: Vec<u16>,
    sleep_lock_frames: Vec<u16>,
    pending_wake: Vec<bool>,
    initial_pos: Vec<Vec2>,
    initial_vel: Vec<Vec2>,
    active_chunk_min: Option<IVec2>,
    active_chunk_max: Option<IVec2>,
    active_halo_chunks: i32,
    pub(crate) parallel_enabled: bool,
    pub(crate) solver_params: SolverParams,
    pub(crate) material_params: MaterialParams,
}

impl Default for ParticleWorld {
    fn default() -> Self {
        Self {
            pos: Vec::new(),
            prev_pos: Vec::new(),
            vel: Vec::new(),
            mass: Vec::new(),
            material: Vec::new(),
            activity_state: Vec::new(),
            sleep_candidate_frames: Vec::new(),
            active_hold_frames: Vec::new(),
            sleep_lock_frames: Vec::new(),
            pending_wake: Vec::new(),
            initial_pos: Vec::new(),
            initial_vel: Vec::new(),
            active_chunk_min: None,
            active_chunk_max: None,
            active_halo_chunks: 0,
            parallel_enabled: true,
            solver_params: SolverParams::default(),
            material_params: MaterialParams::default(),
        }
    }
}

impl ParticleWorld {
    pub fn particle_count(&self) -> usize {
        self.pos.len()
    }

    pub fn set_parallel_enabled(&mut self, enabled: bool) {
        self.parallel_enabled = enabled;
    }

    pub fn set_solver_params(&mut self, params: SolverParams) {
        self.solver_params = params;
    }

    pub fn set_material_params(&mut self, params: MaterialParams) {
        self.material_params = params;
    }

    pub fn set_active_chunk_region_bounds(
        &mut self,
        chunk_min: Option<IVec2>,
        chunk_max: Option<IVec2>,
    ) {
        self.active_chunk_min = chunk_min;
        self.active_chunk_max = chunk_max;
    }

    pub fn set_active_halo_chunks(&mut self, halo_chunks: i32) {
        self.active_halo_chunks = halo_chunks.max(0);
    }

    pub fn configure_far_field_queue(
        &mut self,
        _center_chunk: Option<IVec2>,
        _freeze_radius_chunks: i32,
        _release_radius_chunks: i32,
        _release_particles_per_frame: usize,
        _release_clearance_radius_m: f32,
        _release_clearance_max_wait_frames: u16,
    ) {
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

    pub fn activity_states(&self) -> &[ParticleActivityState] {
        &self.activity_state
    }

    pub fn restore_from_snapshot(
        &mut self,
        positions: Vec<Vec2>,
        velocities: Vec<Vec2>,
        materials: Vec<ParticleMaterial>,
    ) -> Result<(), String> {
        if positions.len() != velocities.len() || positions.len() != materials.len() {
            return Err("particle snapshot has mismatched array lengths".to_string());
        }

        self.pos = positions;
        self.prev_pos = self.pos.clone();
        self.vel = velocities;
        self.material = materials;
        self.mass = self
            .material
            .iter()
            .map(|&mat| particle_properties(mat).mass)
            .collect();
        self.initial_pos = self.pos.clone();
        self.initial_vel = self.vel.clone();
        self.resize_work_buffers();
        Ok(())
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
            self.request_wake(i);
        }
    }

    pub fn wake_particles_in_radius(&mut self, center: Vec2, radius: f32) {
        let radius2 = radius * radius;
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) <= radius2 {
                self.request_wake(i);
            }
        }
    }

    pub fn promote_particles_in_chunk_radius(&mut self, center_chunk: IVec2, radius_chunks: i32) {
        let radius_chunks = radius_chunks.max(0);
        for i in 0..self.particle_count() {
            let chunk = world_pos_to_chunk(self.pos[i]);
            if is_chunk_outside_radius(chunk, center_chunk, radius_chunks) {
                continue;
            }
            self.request_wake(i);
        }
    }

    pub fn spawn_material_particles_from_cells(
        &mut self,
        cells: &[IVec2],
        material: ParticleMaterial,
        initial_velocity: Vec2,
    ) -> Vec<usize> {
        let mut indices = Vec::new();
        for &cell in cells {
            self.append_material_particles_in_cell(cell, material, initial_velocity, &mut indices);
            self.request_wake_near(cell_to_world_center(cell), WAKE_RADIUS);
        }
        self.resize_work_buffers();
        indices
    }

    pub fn remove_particles_in_radius_with_map(
        &mut self,
        center: Vec2,
        radius: f32,
    ) -> ParticleRemovalResult {
        let radius2 = radius * radius;
        let count = self.particle_count();
        let mut old_to_new = vec![None; count];
        let mut next = 0usize;
        for (i, &p) in self.pos.iter().enumerate() {
            if p.distance_squared(center) <= radius2 {
                continue;
            }
            old_to_new[i] = Some(next);
            next += 1;
        }
        if next == count {
            return ParticleRemovalResult {
                removed_count: 0,
                old_to_new,
            };
        }

        let keep: Vec<bool> = old_to_new.iter().map(Option::is_some).collect();
        self.compact_with_keep(&keep);
        ParticleRemovalResult {
            removed_count: count - next,
            old_to_new,
        }
    }

    pub fn fracture_solid_particles_in_radius(&mut self, center: Vec2, radius: f32) -> HashSet<usize> {
        let radius2 = radius * radius;
        let mut fractured = HashSet::new();
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) > radius2 {
                continue;
            }
            let Some(spec) = solid_break_properties(self.material[i]) else {
                continue;
            };
            let Some(target) = spec.fracture_to else {
                continue;
            };
            self.material[i] = target;
            self.mass[i] = particle_properties(target).mass;
            self.request_wake(i);
            fractured.insert(i);
        }
        fractured
    }

    pub fn detach_terrain_components_after_cell_removal(
        &mut self,
        _terrain: &mut TerrainWorld,
        _object_world: &mut ObjectWorld,
        _removed_cells: &HashSet<IVec2>,
    ) -> bool {
        // GPU-first path: terrain detach to rigid objects is deferred.
        false
    }

    pub fn postprocess_objects_after_topology_edit(&mut self, _object_world: &mut ObjectWorld) {}

    pub fn detach_and_postprocess_objects(
        &mut self,
        object_world: &mut ObjectWorld,
        detached_particles: &HashSet<usize>,
    ) {
        object_world.split_objects_after_detach(detached_particles, self.positions(), self.masses());
    }

    pub(crate) fn resize_work_buffers(&mut self) {
        let count = self.particle_count();
        self.activity_state.resize(count, ParticleActivityState::Active);
        self.sleep_candidate_frames.resize(count, 0);
        self.active_hold_frames.resize(count, ACTIVE_MIN_FRAMES);
        self.sleep_lock_frames.resize(count, 0);
        self.pending_wake.resize(count, false);
    }

    fn append_material_particles_in_cell(
        &mut self,
        cell: IVec2,
        material: ParticleMaterial,
        velocity: Vec2,
        out_indices: &mut Vec<usize>,
    ) {
        let props = particle_properties(material);
        let count = particles_per_cell(material);
        let axis = particle_grid_axis(count);
        let axis_f = axis as f32;
        let spacing = CELL_SIZE_M / axis_f.max(1.0);
        let cell_min = cell_to_world_center(cell) - Vec2::splat(CELL_SIZE_M * 0.5);

        let mut spawned = 0u32;
        'grid: for y in 0..axis {
            for x in 0..axis {
                if spawned >= count {
                    break 'grid;
                }
                let offset = Vec2::new((x as f32 + 0.5) * spacing, (y as f32 + 0.5) * spacing);
                let p = cell_min + offset;
                let index = self.pos.len();
                self.pos.push(p);
                self.prev_pos.push(p);
                self.vel.push(velocity);
                self.mass.push(props.mass);
                self.material.push(material);
                self.activity_state.push(ParticleActivityState::Active);
                self.sleep_candidate_frames.push(0);
                self.active_hold_frames.push(ACTIVE_MIN_FRAMES);
                self.sleep_lock_frames.push(0);
                self.pending_wake.push(false);
                out_indices.push(index);
                spawned += 1;
            }
        }
    }

    fn compact_with_keep(&mut self, keep: &[bool]) {
        fn compact_vec<T: Copy>(data: &mut Vec<T>, keep: &[bool]) {
            let mut write = 0usize;
            for read in 0..data.len() {
                if !keep[read] {
                    continue;
                }
                if write != read {
                    data[write] = data[read];
                }
                write += 1;
            }
            data.truncate(write);
        }

        compact_vec(&mut self.pos, keep);
        compact_vec(&mut self.prev_pos, keep);
        compact_vec(&mut self.vel, keep);
        compact_vec(&mut self.mass, keep);
        compact_vec(&mut self.material, keep);
        compact_vec(&mut self.activity_state, keep);
        compact_vec(&mut self.sleep_candidate_frames, keep);
        compact_vec(&mut self.active_hold_frames, keep);
        compact_vec(&mut self.sleep_lock_frames, keep);
        compact_vec(&mut self.pending_wake, keep);
    }

    fn request_wake(&mut self, index: usize) {
        if index >= self.particle_count() {
            return;
        }
        self.activity_state[index] = ParticleActivityState::Active;
        self.sleep_candidate_frames[index] = 0;
        self.active_hold_frames[index] = self.active_hold_frames[index].max(ACTIVE_MIN_FRAMES);
        self.sleep_lock_frames[index] = 0;
        self.pending_wake[index] = false;
    }

    fn request_wake_near(&mut self, center: Vec2, radius: f32) {
        self.wake_particles_in_radius(center, radius);
    }
}

fn particle_grid_axis(count: u32) -> u32 {
    let axis = (count as f32).sqrt().ceil() as u32;
    axis.max(1)
}

fn world_pos_to_chunk(world_pos: Vec2) -> IVec2 {
    let cell = IVec2::new(
        (world_pos.x / CELL_SIZE_M).floor() as i32,
        (world_pos.y / CELL_SIZE_M).floor() as i32,
    );
    IVec2::new(
        cell.x.div_euclid(CHUNK_SIZE_I32),
        cell.y.div_euclid(CHUNK_SIZE_I32),
    )
}

fn is_chunk_outside_radius(chunk: IVec2, center: IVec2, radius: i32) -> bool {
    (chunk.x - center.x).abs() > radius || (chunk.y - center.y).abs() > radius
}
