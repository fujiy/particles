use bevy::log::tracing;
use bevy::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::physics::connectivity::{FOUR_NEIGHBOR_OFFSETS, flood_fill_4_limited};
use crate::physics::material::{
    DEFAULT_MATERIAL_PARAMS, MaterialParams, particle_properties, particles_per_cell,
    solid_break_properties, terrain_fracture_particle, terrain_solid_particle,
    water_kernel_radius_m,
};
use crate::physics::solver::params_types::SolverParams;
use crate::physics::world::neighbor_grid::NeighborGrid;
use crate::physics::world::object::{
    OBJECT_SHAPE_ITERS, OBJECT_SHAPE_STIFFNESS_ALPHA, ObjectId, ObjectPhysicsField, ObjectWorld,
};
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, TerrainCell, TerrainWorld, cell_to_world_center, world_to_cell,
};

use crate::physics::solver::params_defaults::DEFAULT_SOLVER_PARAMS;

const FIXED_DT: f32 = DEFAULT_SOLVER_PARAMS.fixed_dt;
const SUBSTEPS: usize = DEFAULT_SOLVER_PARAMS.substeps;
const EPSILON_LAMBDA: f32 = DEFAULT_SOLVER_PARAMS.epsilon_lambda;
const TERRAIN_GHOST_DENSITY_SCALE: f32 = DEFAULT_SOLVER_PARAMS.terrain_ghost_density_scale;
const TERRAIN_GHOST_DELTA_SCALE: f32 = DEFAULT_SOLVER_PARAMS.terrain_ghost_delta_scale;
const PARALLEL_PARTICLE_THRESHOLD: usize = DEFAULT_SOLVER_PARAMS.parallel_particle_threshold;
const PARTICLE_CONTACT_PUSH_FACTOR: f32 = DEFAULT_SOLVER_PARAMS.particle_contact_push_factor;
const DETACH_FLOOD_FILL_MAX_CELLS: usize = DEFAULT_SOLVER_PARAMS.detach_flood_fill_max_cells;
const SLEEP_DISP_THRESHOLD: f32 = DEFAULT_SOLVER_PARAMS.sleep_disp_threshold;
const SLEEP_VEL_THRESHOLD: f32 = DEFAULT_SOLVER_PARAMS.sleep_vel_threshold;
const SLEEP_FRAMES: u16 = DEFAULT_SOLVER_PARAMS.sleep_frames;
const WAKE_DISP_THRESHOLD: f32 = DEFAULT_SOLVER_PARAMS.wake_disp_threshold;
const WAKE_RADIUS: f32 = DEFAULT_SOLVER_PARAMS.wake_radius_m;
const ACTIVE_MIN_FRAMES: u16 = DEFAULT_SOLVER_PARAMS.active_min_frames;
const FRACTURE_WAKE_LOCK_FRAMES: u16 = DEFAULT_SOLVER_PARAMS.fracture_wake_lock_frames;
const TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS: u32 =
    DEFAULT_SOLVER_PARAMS.terrain_load_sample_interval_substeps;
const TERRAIN_LOAD_STRAIN_THRESHOLD: f32 = DEFAULT_SOLVER_PARAMS.terrain_load_strain_threshold;
const TERRAIN_LOAD_BREAK_DURATION_SECONDS: f32 =
    DEFAULT_SOLVER_PARAMS.terrain_load_break_duration_seconds;
const TERRAIN_LOAD_DECAY_PER_SAMPLE: f32 = DEFAULT_SOLVER_PARAMS.terrain_load_decay_per_sample;
const GRANULAR_CONTACT_FRICTION_SCALE: f32 = DEFAULT_SOLVER_PARAMS.granular_contact_friction_scale;
const GRANULAR_GRANULAR_CONTACT_FRICTION_BOOST: f32 =
    DEFAULT_SOLVER_PARAMS.granular_granular_contact_friction_boost;
const GRANULAR_SOLID_CONTACT_FRICTION_BOOST: f32 =
    DEFAULT_SOLVER_PARAMS.granular_solid_contact_friction_boost;
const TERRAIN_CONTACT_FRICTION_SCALE: f32 = DEFAULT_SOLVER_PARAMS.terrain_contact_friction_scale;
const GRANULAR_CONTACT_NORMAL_DAMPING: f32 = DEFAULT_SOLVER_PARAMS.granular_contact_normal_damping;
const TERRAIN_CONTACT_NORMAL_DAMPING: f32 = DEFAULT_SOLVER_PARAMS.terrain_contact_normal_damping;
const GRANULAR_SPAWN_JITTER_RATIO: f32 = DEFAULT_SOLVER_PARAMS.granular_spawn_jitter_ratio;
const PARTICLE_ESCAPE_MARGIN_X_CELLS: i32 = DEFAULT_SOLVER_PARAMS.particle_escape_margin_x_cells;
const PARTICLE_ESCAPE_MARGIN_BOTTOM_CELLS: i32 =
    DEFAULT_SOLVER_PARAMS.particle_escape_margin_bottom_cells;
const PARTICLE_ESCAPE_MARGIN_TOP_CELLS: i32 =
    DEFAULT_SOLVER_PARAMS.particle_escape_margin_top_cells;
const FAR_FIELD_EDGE_INSET_M: f32 = DEFAULT_SOLVER_PARAMS.far_field_edge_inset_m;
const NEIGHBOR_LIST_SKIN_M: f32 = DEFAULT_SOLVER_PARAMS.neighbor_list_skin_m;

pub use crate::physics::material::ParticleMaterial;

const REST_DENSITY: f32 = DEFAULT_MATERIAL_PARAMS.rest_density;
const PARTICLE_SPEED_LIMIT_MPS: f32 = DEFAULT_MATERIAL_PARAMS.particle_speed_limit_mps;
const WATER_KERNEL_RADIUS_M: f32 = water_kernel_radius_m(DEFAULT_MATERIAL_PARAMS);

pub fn nominal_particle_draw_radius_m() -> f32 {
    (default_particle_mass() / (std::f32::consts::PI * REST_DENSITY)).sqrt()
}

#[derive(Clone, Copy, Debug)]
pub struct ParticleStepPhaseTiming {
    pub name: &'static str,
    pub wall_duration_secs: f64,
    pub cpu_duration_secs: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ParticleStepBreakdown {
    pub clear_reaction_impulses_secs: f64,
    pub clear_reaction_impulses_cpu_secs: f64,
    pub predict_positions_secs: f64,
    pub predict_positions_cpu_secs: f64,
    pub cull_escaped_particles_secs: f64,
    pub cull_escaped_particles_cpu_secs: f64,
    pub rebuild_neighbor_grid_secs: f64,
    pub rebuild_neighbor_grid_cpu_secs: f64,
    pub solve_density_constraints_secs: f64,
    pub solve_density_constraints_cpu_secs: f64,
    pub granular_solver_secs: f64,
    pub granular_solver_cpu_secs: f64,
    pub shape_matching_secs: f64,
    pub shape_matching_cpu_secs: f64,
    pub shape_contact_projection_secs: f64,
    pub shape_contact_projection_cpu_secs: f64,
    pub update_velocity_secs: f64,
    pub update_velocity_cpu_secs: f64,
    pub granular_restitution_secs: f64,
    pub granular_restitution_cpu_secs: f64,
    pub contact_velocity_response_secs: f64,
    pub contact_velocity_response_cpu_secs: f64,
    pub xsph_viscosity_secs: f64,
    pub xsph_viscosity_cpu_secs: f64,
    pub apply_object_reaction_secs: f64,
    pub apply_object_reaction_cpu_secs: f64,
    pub final_velocity_clamp_secs: f64,
    pub final_velocity_clamp_cpu_secs: f64,
    pub wake_detection_secs: f64,
    pub wake_detection_cpu_secs: f64,
    pub fracture_detection_secs: f64,
    pub fracture_detection_cpu_secs: f64,
    pub sleep_update_secs: f64,
    pub sleep_update_cpu_secs: f64,
}

impl ParticleStepBreakdown {
    pub fn total_wall_secs(&self) -> f64 {
        self.phases()
            .iter()
            .map(|phase| phase.wall_duration_secs)
            .sum::<f64>()
    }

    pub fn total_cpu_secs(&self) -> f64 {
        self.phases()
            .iter()
            .map(|phase| phase.cpu_duration_secs)
            .sum::<f64>()
    }

    pub fn phases(&self) -> [ParticleStepPhaseTiming; 17] {
        [
            ParticleStepPhaseTiming {
                name: "clear_reaction_impulses",
                wall_duration_secs: self.clear_reaction_impulses_secs,
                cpu_duration_secs: self.clear_reaction_impulses_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "predict_positions",
                wall_duration_secs: self.predict_positions_secs,
                cpu_duration_secs: self.predict_positions_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "cull_escaped_particles",
                wall_duration_secs: self.cull_escaped_particles_secs,
                cpu_duration_secs: self.cull_escaped_particles_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "rebuild_neighbor_grid",
                wall_duration_secs: self.rebuild_neighbor_grid_secs,
                cpu_duration_secs: self.rebuild_neighbor_grid_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "solve_density_constraints",
                wall_duration_secs: self.solve_density_constraints_secs,
                cpu_duration_secs: self.solve_density_constraints_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "granular_solver",
                wall_duration_secs: self.granular_solver_secs,
                cpu_duration_secs: self.granular_solver_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "shape_matching",
                wall_duration_secs: self.shape_matching_secs,
                cpu_duration_secs: self.shape_matching_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "shape_contact_projection",
                wall_duration_secs: self.shape_contact_projection_secs,
                cpu_duration_secs: self.shape_contact_projection_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "update_velocity",
                wall_duration_secs: self.update_velocity_secs,
                cpu_duration_secs: self.update_velocity_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "granular_restitution",
                wall_duration_secs: self.granular_restitution_secs,
                cpu_duration_secs: self.granular_restitution_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "contact_velocity_response",
                wall_duration_secs: self.contact_velocity_response_secs,
                cpu_duration_secs: self.contact_velocity_response_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "xsph_viscosity",
                wall_duration_secs: self.xsph_viscosity_secs,
                cpu_duration_secs: self.xsph_viscosity_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "apply_object_reaction",
                wall_duration_secs: self.apply_object_reaction_secs,
                cpu_duration_secs: self.apply_object_reaction_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "final_velocity_clamp",
                wall_duration_secs: self.final_velocity_clamp_secs,
                cpu_duration_secs: self.final_velocity_clamp_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "wake_detection",
                wall_duration_secs: self.wake_detection_secs,
                cpu_duration_secs: self.wake_detection_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "fracture_detection",
                wall_duration_secs: self.fracture_detection_secs,
                cpu_duration_secs: self.fracture_detection_cpu_secs,
            },
            ParticleStepPhaseTiming {
                name: "sleep_update",
                wall_duration_secs: self.sleep_update_secs,
                cpu_duration_secs: self.sleep_update_cpu_secs,
            },
        ]
    }
}

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

pub(crate) mod helpers;
#[path = "../../solver/liquid.rs"]
mod liquid_solver;
#[path = "../../solver/object.rs"]
mod object_solver;

use crate::physics::solver::granular::ComputeDeltaThreadScratch;
use helpers::*;

#[derive(Clone, Copy, Debug, Default)]
struct TerrainFractureSeed {
    impulse_sum: f32,
    velocity_weighted_sum: Vec2,
    contact_weighted_sum: Vec2,
}

impl TerrainFractureSeed {
    fn accumulate(&mut self, impulse: f32, velocity: Vec2, contact_pos: Vec2) {
        self.impulse_sum += impulse;
        self.velocity_weighted_sum += velocity * impulse;
        self.contact_weighted_sum += contact_pos * impulse;
    }

    fn velocity(self) -> Vec2 {
        if self.impulse_sum <= 1e-6 {
            Vec2::ZERO
        } else {
            self.velocity_weighted_sum / self.impulse_sum
        }
    }

    fn contact_pos(self) -> Vec2 {
        if self.impulse_sum <= 1e-6 {
            Vec2::ZERO
        } else {
            self.contact_weighted_sum / self.impulse_sum
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TerrainPersistentLoadState {
    strain_metric: f32,
    sustained_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ChunkEdge {
    North,
    South,
    East,
    West,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DeferredEdgeKey {
    chunk: IVec2,
    edge: ChunkEdge,
    grid_segment: i32,
}

#[derive(Clone, Copy, Debug)]
struct DeferredParticleState {
    position: Vec2,
    velocity: Vec2,
    mass: f32,
    material: ParticleMaterial,
}

#[derive(Clone, Copy, Debug)]
struct BoundaryReleaseLock {
    position: Vec2,
    wait_frames: u16,
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
    activity_state: Vec<ParticleActivityState>,
    sleep_candidate_frames: Vec<u16>,
    active_hold_frames: Vec<u16>,
    sleep_lock_frames: Vec<u16>,
    pending_wake: Vec<bool>,
    initial_pos: Vec<Vec2>,
    initial_vel: Vec<Vec2>,
    pub(crate) neighbor_grid: NeighborGrid,
    pub(crate) neighbor_cache: Vec<Vec<usize>>,
    neighbor_cache_anchor_pos: Vec<Vec2>,
    neighbor_cache_anchor_valid: bool,
    viscosity_work: Vec<Vec2>,
    object_peak_strain: HashMap<ObjectId, f32>,
    object_peak_strain_particle: HashMap<ObjectId, usize>,
    pending_object_fractures: HashSet<ObjectId>,
    pending_object_fracture_particles: HashMap<ObjectId, HashSet<usize>>,
    pending_terrain_fractures: HashSet<IVec2>,
    pending_terrain_fracture_seeds: HashMap<IVec2, TerrainFractureSeed>,
    terrain_persistent_load: HashMap<IVec2, TerrainPersistentLoadState>,
    terrain_load_substep_counter: u64,
    active_chunk_min: Option<IVec2>,
    active_chunk_max: Option<IVec2>,
    active_halo_chunks: i32,
    far_field_center_chunk: Option<IVec2>,
    far_field_freeze_radius_chunks: i32,
    far_field_release_radius_chunks: i32,
    far_field_release_particles_per_frame: usize,
    far_field_release_clearance_radius_m: f32,
    far_field_release_clearance_max_wait_frames: u16,
    deferred_inactive_chunk_particles: HashMap<IVec2, VecDeque<DeferredParticleState>>,
    deferred_boundary_particles: HashMap<DeferredEdgeKey, VecDeque<DeferredParticleState>>,
    deferred_boundary_release_locks: HashMap<DeferredEdgeKey, BoundaryReleaseLock>,
    pub(crate) parallel_enabled: bool,
    pub(crate) solver_params: SolverParams,
    pub(crate) material_params: MaterialParams,
}

impl Default for ParticleWorld {
    fn default() -> Self {
        let initial_pos = Vec::new();
        let initial_vel = Vec::new();
        let mut world = Self {
            pos: initial_pos.clone(),
            prev_pos: initial_pos.clone(),
            vel: initial_vel.clone(),
            mass: vec![default_particle_mass(); initial_pos.len()],
            material: vec![ParticleMaterial::WaterLiquid; initial_pos.len()],
            density: Vec::new(),
            lambda: Vec::new(),
            delta_pos: Vec::new(),
            activity_state: Vec::new(),
            sleep_candidate_frames: Vec::new(),
            active_hold_frames: Vec::new(),
            sleep_lock_frames: Vec::new(),
            pending_wake: Vec::new(),
            initial_pos,
            initial_vel,
            neighbor_grid: NeighborGrid::default(),
            neighbor_cache: Vec::new(),
            neighbor_cache_anchor_pos: Vec::new(),
            neighbor_cache_anchor_valid: false,
            viscosity_work: Vec::new(),
            object_peak_strain: HashMap::new(),
            object_peak_strain_particle: HashMap::new(),
            pending_object_fractures: HashSet::new(),
            pending_object_fracture_particles: HashMap::new(),
            pending_terrain_fractures: HashSet::new(),
            pending_terrain_fracture_seeds: HashMap::new(),
            terrain_persistent_load: HashMap::new(),
            terrain_load_substep_counter: 0,
            active_chunk_min: None,
            active_chunk_max: None,
            active_halo_chunks: 1,
            far_field_center_chunk: None,
            far_field_freeze_radius_chunks: 0,
            far_field_release_radius_chunks: 0,
            far_field_release_particles_per_frame: 0,
            far_field_release_clearance_radius_m: 0.0,
            far_field_release_clearance_max_wait_frames: 0,
            deferred_inactive_chunk_particles: HashMap::new(),
            deferred_boundary_particles: HashMap::new(),
            deferred_boundary_release_locks: HashMap::new(),
            parallel_enabled: true,
            solver_params: SolverParams::default(),
            material_params: MaterialParams::default(),
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
        self.material.fill(ParticleMaterial::WaterLiquid);
        self.object_peak_strain.clear();
        self.object_peak_strain_particle.clear();
        self.pending_object_fractures.clear();
        self.pending_object_fracture_particles.clear();
        self.pending_terrain_fractures.clear();
        self.pending_terrain_fracture_seeds.clear();
        self.terrain_persistent_load.clear();
        self.deferred_inactive_chunk_particles.clear();
        self.deferred_boundary_particles.clear();
        self.deferred_boundary_release_locks.clear();
        self.terrain_load_substep_counter = 0;
        self.activity_state.fill(ParticleActivityState::Active);
        self.sleep_candidate_frames.fill(0);
        self.active_hold_frames.fill(0);
        self.sleep_lock_frames.fill(0);
        self.pending_wake.fill(false);
        self.resize_work_buffers();
    }

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
        center_chunk: Option<IVec2>,
        freeze_radius_chunks: i32,
        release_radius_chunks: i32,
        release_particles_per_frame: usize,
        release_clearance_radius_m: f32,
        release_clearance_max_wait_frames: u16,
    ) {
        self.far_field_center_chunk = center_chunk;
        self.far_field_freeze_radius_chunks = freeze_radius_chunks.max(0);
        self.far_field_release_radius_chunks = release_radius_chunks.max(0);
        self.far_field_release_particles_per_frame = release_particles_per_frame;
        self.far_field_release_clearance_radius_m = release_clearance_radius_m.max(0.0);
        self.far_field_release_clearance_max_wait_frames = release_clearance_max_wait_frames;
        if self.far_field_center_chunk.is_none() {
            self.deferred_inactive_chunk_particles.clear();
            self.deferred_boundary_particles.clear();
            self.deferred_boundary_release_locks.clear();
        }
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

    pub fn is_particle_active_in_region(&self, index: usize) -> bool {
        self.is_active_particle(index)
    }

    pub fn is_particle_in_halo_region(&self, index: usize) -> bool {
        self.is_halo_particle(index)
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
            .map(|&material| particle_properties(material).mass)
            .collect();
        self.initial_pos = self.pos.clone();
        self.initial_vel = self.vel.clone();
        self.object_peak_strain.clear();
        self.object_peak_strain_particle.clear();
        self.pending_object_fractures.clear();
        self.pending_object_fracture_particles.clear();
        self.pending_terrain_fractures.clear();
        self.pending_terrain_fracture_seeds.clear();
        self.terrain_persistent_load.clear();
        self.deferred_inactive_chunk_particles.clear();
        self.deferred_boundary_particles.clear();
        self.deferred_boundary_release_locks.clear();
        self.terrain_load_substep_counter = 0;
        self.activity_state.fill(ParticleActivityState::Active);
        self.sleep_candidate_frames.fill(0);
        self.active_hold_frames.fill(0);
        self.sleep_lock_frames.fill(0);
        self.pending_wake.fill(false);
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
            if self.activity_state[i] == ParticleActivityState::Active {
                continue;
            }
            self.activity_state[i] = ParticleActivityState::Active;
            self.sleep_candidate_frames[i] = 0;
            self.active_hold_frames[i] = self.active_hold_frames[i].max(ACTIVE_MIN_FRAMES);
            self.sleep_lock_frames[i] = 0;
            self.pending_wake[i] = false;
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
        self.spawn_material_particles_from_cells(
            cells,
            ParticleMaterial::StoneSolid,
            initial_velocity,
        )
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

    pub fn apply_pending_terrain_fractures(
        &mut self,
        terrain: &mut TerrainWorld,
        object_world: &mut ObjectWorld,
    ) -> bool {
        if self.pending_terrain_fractures.is_empty() {
            return false;
        }

        let fracture_cells: Vec<_> = self.pending_terrain_fractures.drain().collect();
        let fracture_cell_set: HashSet<_> = fracture_cells.iter().copied().collect();
        let mut changed = false;
        let mut appended_indices = Vec::new();
        for &cell in &fracture_cells {
            let current = terrain.get_loaded_cell_or_empty(cell);
            let TerrainCell::Solid { material, .. } = current else {
                continue;
            };
            let Some(target_particle) = terrain_fracture_particle(material) else {
                continue;
            };
            if !terrain.set_cell(cell, TerrainCell::Empty) {
                continue;
            }
            changed = true;
            self.append_material_particles_in_cell(
                cell,
                target_particle,
                self.terrain_fracture_seed_velocity(cell),
                &mut appended_indices,
            );
        }

        let detached_components =
            self.collect_detached_terrain_components(terrain, &fracture_cell_set);

        for component_cells in detached_components {
            let spawn_plan =
                self.build_terrain_detach_spawn_plan(&component_cells, terrain, &fracture_cell_set);
            if spawn_plan.cells.is_empty() {
                continue;
            }
            let indices = self.spawn_terrain_detach_component(&spawn_plan);
            if indices.is_empty() {
                continue;
            }

            for cell in &spawn_plan.cells {
                changed |= terrain.set_cell(*cell, TerrainCell::Empty);
            }
            let _ = object_world.create_object(
                indices,
                self.positions(),
                self.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            );
        }

        if !appended_indices.is_empty() || changed {
            for &cell in &fracture_cells {
                self.request_wake_near(cell_to_world_center(cell), WAKE_RADIUS);
            }
            self.resize_work_buffers();
        }
        if changed {
            self.auto_fracture_single_cell_objects(object_world);
        }
        self.pending_terrain_fracture_seeds.clear();
        changed
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
        let mut new_activity_state = Vec::with_capacity(new_count);
        let mut new_sleep_candidate_frames = Vec::with_capacity(new_count);
        let mut new_active_hold_frames = Vec::with_capacity(new_count);
        let mut new_sleep_lock_frames = Vec::with_capacity(new_count);
        let mut new_pending_wake = Vec::with_capacity(new_count);
        for old_index in 0..old_count {
            if !keep[old_index] {
                continue;
            }
            new_pos.push(self.pos[old_index]);
            new_prev_pos.push(self.prev_pos[old_index]);
            new_vel.push(self.vel[old_index]);
            new_mass.push(self.mass[old_index]);
            new_material.push(self.material[old_index]);
            new_activity_state.push(self.activity_state[old_index]);
            new_sleep_candidate_frames.push(self.sleep_candidate_frames[old_index]);
            new_active_hold_frames.push(self.active_hold_frames[old_index]);
            new_sleep_lock_frames.push(self.sleep_lock_frames[old_index]);
            new_pending_wake.push(self.pending_wake[old_index]);
        }

        self.pos = new_pos;
        self.prev_pos = new_prev_pos;
        self.vel = new_vel;
        self.mass = new_mass;
        self.material = new_material;
        self.activity_state = new_activity_state;
        self.sleep_candidate_frames = new_sleep_candidate_frames;
        self.active_hold_frames = new_active_hold_frames;
        self.sleep_lock_frames = new_sleep_lock_frames;
        self.pending_wake = new_pending_wake;
        self.request_wake_near(center, WAKE_RADIUS);
        self.resize_work_buffers();

        ParticleRemovalResult {
            removed_count,
            old_to_new,
        }
    }

    pub fn fracture_solid_particles_in_radius(
        &mut self,
        center: Vec2,
        radius: f32,
    ) -> HashSet<usize> {
        let radius2 = radius * radius;
        let mut detached = HashSet::new();
        let mut spawned_particles = Vec::new();
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) > radius2 {
                continue;
            }
            let Some(target_material) =
                solid_break_properties(self.material[i]).and_then(|props| props.fracture_to)
            else {
                continue;
            };
            detached.insert(i);
            self.fracture_particle_to_target_material(i, target_material, &mut spawned_particles);
        }
        for (position, velocity, mass, material) in spawned_particles {
            let new_index = self.pos.len();
            self.pos.push(position);
            self.prev_pos.push(position);
            self.vel.push(velocity);
            self.mass.push(mass);
            self.material.push(material);
            if is_granular_particle(material) {
                self.request_wake(new_index);
                self.sleep_lock_frames[new_index] = FRACTURE_WAKE_LOCK_FRAMES;
            }
        }
        if !detached.is_empty() {
            self.request_wake_near(center, WAKE_RADIUS);
            self.resize_work_buffers();
        }
        detached
    }

    pub fn detach_terrain_components_after_cell_removal(
        &mut self,
        terrain: &mut TerrainWorld,
        object_world: &mut ObjectWorld,
        removed_cells: &HashSet<IVec2>,
    ) -> bool {
        if removed_cells.is_empty() {
            return false;
        }
        let detached_components = self.collect_detached_terrain_components(terrain, removed_cells);
        if detached_components.is_empty() {
            return false;
        }

        let mut changed = false;
        let no_seed_cells = HashSet::new();
        for component_cells in detached_components {
            let spawn_plan =
                self.build_terrain_detach_spawn_plan(&component_cells, terrain, &no_seed_cells);
            if spawn_plan.cells.is_empty() {
                continue;
            }
            let indices = self.spawn_terrain_detach_component(&spawn_plan);
            if indices.is_empty() {
                continue;
            }
            for cell in &spawn_plan.cells {
                changed |= terrain.set_cell(*cell, TerrainCell::Empty);
            }
            let _ = object_world.create_object(
                indices,
                self.positions(),
                self.masses(),
                OBJECT_SHAPE_STIFFNESS_ALPHA,
                OBJECT_SHAPE_ITERS,
            );
        }
        if changed {
            for &cell in removed_cells {
                self.request_wake_near(cell_to_world_center(cell), WAKE_RADIUS);
            }
            self.resize_work_buffers();
            self.auto_fracture_single_cell_objects(object_world);
        }
        changed
    }

    pub fn postprocess_objects_after_topology_edit(&mut self, object_world: &mut ObjectWorld) {
        object_world.split_all_disconnected_objects(self.positions(), self.masses());
        self.auto_fracture_single_cell_objects(object_world);
    }

    pub fn detach_and_postprocess_objects(
        &mut self,
        object_world: &mut ObjectWorld,
        detached_particles: &HashSet<usize>,
    ) {
        object_world.split_objects_after_detach(
            detached_particles,
            self.positions(),
            self.masses(),
        );
        self.auto_fracture_single_cell_objects(object_world);
    }

    pub fn step_if_running(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
        running: bool,
    ) -> ParticleStepBreakdown {
        if running {
            return self.step_substeps(terrain, object_field, object_world);
        }
        ParticleStepBreakdown::default()
    }

    pub fn step_substeps(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
    ) -> ParticleStepBreakdown {
        liquid_solver::step_substeps(self, terrain, object_field, object_world)
    }

    fn solve_density_constraints(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &mut ObjectWorld,
        dt_sub: f32,
        refresh_neighbors: bool,
    ) -> f32 {
        let _span = tracing::info_span!("physics::density_constraint_pass").entered();
        let h_ww = WATER_KERNEL_RADIUS_M;
        let inv_dt = 1.0 / dt_sub.max(1e-6);
        let use_parallel =
            self.parallel_enabled && self.particle_count() >= PARALLEL_PARTICLE_THRESHOLD;

        if refresh_neighbors {
            let _span = tracing::info_span!("physics::rebuild_neighbor_cache").entered();
            self.rebuild_neighbor_cache(use_parallel);
            self.mark_neighbor_cache_anchor();
        }

        let max_density_error = {
            let _span = tracing::info_span!("physics::density_lambda").entered();
            if use_parallel {
                self.solve_density_lambda_parallel(terrain, h_ww)
            } else {
                self.solve_density_lambda_sequential(terrain, h_ww)
            }
        };

        let reaction_impulses = {
            let _span = tracing::info_span!("physics::compute_delta").entered();
            if use_parallel {
                self.compute_delta_parallel(terrain, object_field, object_world, inv_dt, h_ww)
            } else {
                self.compute_delta_sequential(terrain, object_field, object_world, inv_dt, h_ww)
            }
        };
        for (object_id, impulse) in reaction_impulses {
            object_world.accumulate_reaction_impulse(object_id, impulse);
        }

        {
            let _span = tracing::info_span!("physics::apply_delta").entered();
            for i in 0..self.particle_count() {
                self.pos[i] += self.delta_pos[i];
            }
        }
        max_density_error
    }

    pub(crate) fn rebuild_neighbor_cache(&mut self, use_parallel: bool) {
        let count = self.particle_count();
        self.neighbor_cache.resize_with(count, Vec::new);
        if use_parallel {
            let positions = &self.pos;
            let grid = &self.neighbor_grid;
            self.neighbor_cache
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, neighbors)| grid.gather(positions[i], neighbors));
        } else {
            for i in 0..count {
                self.neighbor_grid
                    .gather(self.pos[i], &mut self.neighbor_cache[i]);
            }
        }
    }

    pub(crate) fn mark_neighbor_cache_anchor(&mut self) {
        self.neighbor_cache_anchor_pos.clone_from(&self.pos);
        self.neighbor_cache_anchor_valid = true;
    }

    pub(crate) fn neighbor_cache_requires_rebuild(&self) -> bool {
        if !self.neighbor_cache_anchor_valid
            || self.neighbor_cache_anchor_pos.len() != self.particle_count()
        {
            return true;
        }
        let threshold = (NEIGHBOR_LIST_SKIN_M * 0.5).max(1e-5);
        let threshold2 = threshold * threshold;
        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) {
                continue;
            }
            if self.pos[i].distance_squared(self.neighbor_cache_anchor_pos[i]) > threshold2 {
                return true;
            }
        }
        false
    }

    fn solve_density_lambda_sequential(&mut self, terrain: &TerrainWorld, h_ww: f32) -> f32 {
        let mut max_density_error = 0.0f32;
        for i in 0..self.particle_count() {
            if !self.is_active_or_halo_particle(i) || !is_water_particle(self.material[i]) {
                self.density[i] = REST_DENSITY;
                self.lambda[i] = 0.0;
                continue;
            }
            let neighbors = &self.neighbor_cache[i];
            let mut rho = 0.0;
            let mut grad_i = Vec2::ZERO;
            let mut grad_sum_sq = 0.0;

            for &j in neighbors {
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

            if let Some(ghost_r) = terrain_ghost_neighbor_vector(self.pos[i], terrain, h_ww) {
                let ghost_mass = self.mass[i];
                let ghost_r2 = ghost_r.length_squared();
                rho += ghost_mass * kernel_poly6(ghost_r2, h_ww) * TERRAIN_GHOST_DENSITY_SCALE;
                let ghost_grad = (ghost_mass / REST_DENSITY)
                    * kernel_spiky_grad(ghost_r, h_ww)
                    * TERRAIN_GHOST_DENSITY_SCALE;
                grad_sum_sq += ghost_grad.length_squared();
                grad_i += ghost_grad;
            }

            grad_sum_sq += grad_i.length_squared();
            let c_i = rho / REST_DENSITY - 1.0;

            self.density[i] = rho;
            self.lambda[i] = -c_i / (grad_sum_sq + EPSILON_LAMBDA);
            max_density_error = max_density_error.max(c_i.abs());
        }
        max_density_error
    }

    fn solve_density_lambda_parallel(&mut self, terrain: &TerrainWorld, h_ww: f32) -> f32 {
        let positions = &self.pos;
        let masses = &self.mass;
        let materials = &self.material;
        let activity_state = &self.activity_state;
        let neighbor_cache = &self.neighbor_cache;

        self.density
            .par_iter_mut()
            .zip(self.lambda.par_iter_mut())
            .enumerate()
            .map(|(i, (density_i, lambda_i))| {
                if activity_state[i] == ParticleActivityState::Sleeping
                    || !is_water_particle(materials[i])
                {
                    *density_i = REST_DENSITY;
                    *lambda_i = 0.0;
                    return 0.0;
                }

                let mut rho = 0.0;
                let mut grad_i = Vec2::ZERO;
                let mut grad_sum_sq = 0.0;
                for &j in &neighbor_cache[i] {
                    if !is_water_particle(materials[j]) {
                        continue;
                    }
                    let r = positions[i] - positions[j];
                    let r2 = r.length_squared();
                    if r2 >= h_ww * h_ww {
                        continue;
                    }

                    rho += masses[j] * kernel_poly6(r2, h_ww);
                    if i == j {
                        continue;
                    }
                    let grad = (masses[j] / REST_DENSITY) * kernel_spiky_grad(r, h_ww);
                    grad_sum_sq += grad.length_squared();
                    grad_i += grad;
                }

                if let Some(ghost_r) = terrain_ghost_neighbor_vector(positions[i], terrain, h_ww) {
                    let ghost_mass = masses[i];
                    let ghost_r2 = ghost_r.length_squared();
                    rho += ghost_mass * kernel_poly6(ghost_r2, h_ww) * TERRAIN_GHOST_DENSITY_SCALE;
                    let ghost_grad = (ghost_mass / REST_DENSITY)
                        * kernel_spiky_grad(ghost_r, h_ww)
                        * TERRAIN_GHOST_DENSITY_SCALE;
                    grad_sum_sq += ghost_grad.length_squared();
                    grad_i += ghost_grad;
                }

                grad_sum_sq += grad_i.length_squared();
                let c_i = rho / REST_DENSITY - 1.0;
                *density_i = rho;
                *lambda_i = -c_i / (grad_sum_sq + EPSILON_LAMBDA);
                c_i.abs()
            })
            .reduce(|| 0.0, f32::max)
    }

    fn compute_delta_sequential(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
        inv_dt: f32,
        h_ww: f32,
    ) -> HashMap<ObjectId, Vec2> {
        let dt_sub = 1.0 / inv_dt.max(1e-6);
        let mut reaction_impulses = HashMap::new();
        let mut object_contacts = Vec::new();
        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) {
                self.delta_pos[i] = Vec2::ZERO;
                continue;
            }
            let material = self.material[i];
            let props = particle_properties(material);
            let is_water = is_water_particle(material);
            if !is_water {
                self.delta_pos[i] = Vec2::ZERO;
                continue;
            }
            let neighbors = &self.neighbor_cache[i];

            let mut delta = Vec2::ZERO;
            let mut boundary_push = Vec2::ZERO;
            let mut particle_contact_push = Vec2::ZERO;
            if is_water {
                for &j in neighbors {
                    if i == j || !is_water_particle(self.material[j]) {
                        continue;
                    }
                    let r = self.pos[i] - self.pos[j];
                    if r.length_squared() >= h_ww * h_ww {
                        continue;
                    }
                    delta += (self.lambda[i] + self.lambda[j]) * kernel_spiky_grad(r, h_ww);
                }
                if let Some(ghost_r) = terrain_ghost_neighbor_vector(self.pos[i], terrain, h_ww) {
                    delta += self.lambda[i]
                        * kernel_spiky_grad(ghost_r, h_ww)
                        * TERRAIN_GHOST_DELTA_SCALE;
                }
            }
            for &j in neighbors {
                if i == j {
                    continue;
                }
                let neighbor_material = self.material[j];
                if is_water && is_water_particle(neighbor_material) {
                    continue;
                }
                let neighbor_props = particle_properties(neighbor_material);
                let r = self.pos[i] - self.pos[j];
                let dist2 = r.length_squared();
                let contact_radius = props.radius_m + neighbor_props.radius_m;
                if dist2 <= 1e-12 || dist2 >= contact_radius * contact_radius {
                    continue;
                }
                let dist = dist2.sqrt();
                let normal = r / dist;
                let penetration = contact_radius - dist;
                let stiffness = 0.5
                    * (props.object_repulsion_stiffness
                        + neighbor_props.object_repulsion_stiffness);
                let normal_push = normal * penetration * stiffness * PARTICLE_CONTACT_PUSH_FACTOR;
                particle_contact_push += normal_push;
                if let Some(contact_pair_scale) =
                    granular_contact_friction_pair_scale(material, neighbor_material)
                {
                    let static_friction = 0.5
                        * (props.friction_static + neighbor_props.friction_static)
                        * GRANULAR_CONTACT_FRICTION_SCALE
                        * contact_pair_scale;
                    let dynamic_friction = 0.5
                        * (props.friction_dynamic + neighbor_props.friction_dynamic)
                        * GRANULAR_CONTACT_FRICTION_SCALE
                        * contact_pair_scale;
                    let rel_velocity = self.vel[i] - self.vel[j];
                    particle_contact_push += granular_tangential_friction_push(
                        normal,
                        normal_push,
                        rel_velocity,
                        static_friction,
                        dynamic_friction,
                        dt_sub,
                    );
                    let rel_normal_speed = (self.vel[i] - self.vel[j]).dot(normal);
                    if rel_normal_speed < 0.0 {
                        let normal_damping_push = -normal
                            * rel_normal_speed
                            * dt_sub
                            * GRANULAR_CONTACT_NORMAL_DAMPING
                            * contact_pair_scale;
                        let max_damping_push = penetration * 0.35;
                        particle_contact_push +=
                            normal_damping_push.clamp_length_max(max_damping_push);
                    }
                }
            }

            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            {
                let penetration = props.terrain_push_radius_m - signed_distance;
                if penetration > 0.0 {
                    let normal_push = normal * penetration * props.terrain_repulsion_stiffness;
                    boundary_push += normal_push;
                    if !is_water {
                        boundary_push += granular_tangential_friction_push(
                            normal,
                            normal_push,
                            self.vel[i],
                            props.friction_static * TERRAIN_CONTACT_FRICTION_SCALE,
                            props.friction_dynamic * TERRAIN_CONTACT_FRICTION_SCALE,
                            dt_sub,
                        );
                        let normal_speed = self.vel[i].dot(normal);
                        if normal_speed < 0.0 {
                            let normal_damping_push =
                                -normal * normal_speed * dt_sub * TERRAIN_CONTACT_NORMAL_DAMPING;
                            boundary_push +=
                                normal_damping_push.clamp_length_max(penetration * 0.35);
                        }
                    }
                }
            }

            let mut object_push = Vec2::ZERO;
            let self_object = object_world.object_of_particle(i);
            if is_water || self_object.is_some() {
                object_field.gather_candidate_object_ids(self.pos[i], &mut object_contacts);

                for &object_id in &object_contacts {
                    if self_object == Some(object_id) {
                        continue;
                    }
                    let Some(sample) = object_world.evaluate_object_sdf(object_id, self.pos[i])
                    else {
                        continue;
                    };
                    let penetration = props.object_push_radius_m - sample.distance_m;
                    if penetration <= 0.0 {
                        continue;
                    }
                    let push = sample.normal_world * penetration * props.object_repulsion_stiffness;
                    object_push += push;
                    let reaction_impulse = -(self.mass[i] * push) * inv_dt;
                    *reaction_impulses.entry(object_id).or_insert(Vec2::ZERO) += reaction_impulse;
                }
            }

            let max_push = props.object_push_radius_m;
            object_push = object_push.clamp_length_max(max_push);
            self.delta_pos[i] = if is_water {
                delta / REST_DENSITY + boundary_push + object_push + particle_contact_push
            } else {
                boundary_push + object_push + particle_contact_push
            };
        }
        reaction_impulses
    }

    fn compute_delta_parallel(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
        inv_dt: f32,
        h_ww: f32,
    ) -> HashMap<ObjectId, Vec2> {
        let dt_sub = 1.0 / inv_dt.max(1e-6);
        let positions = &self.pos;
        let velocities = &self.vel;
        let masses = &self.mass;
        let materials = &self.material;
        let activity_state = &self.activity_state;
        let lambdas = &self.lambda;
        let neighbor_cache = &self.neighbor_cache;

        let reduced = self
            .delta_pos
            .par_iter_mut()
            .enumerate()
            .fold(
                ComputeDeltaThreadScratch::default,
                |mut scratch, (i, delta_pos_i)| {
                    if activity_state[i] == ParticleActivityState::Sleeping {
                        *delta_pos_i = Vec2::ZERO;
                        return scratch;
                    }
                    let material = materials[i];
                    let props = particle_properties(material);
                    let is_water = is_water_particle(material);
                    if !is_water {
                        *delta_pos_i = Vec2::ZERO;
                        return scratch;
                    }
                    let neighbors = &neighbor_cache[i];

                    let mut delta = Vec2::ZERO;
                    let mut boundary_push = Vec2::ZERO;
                    let mut particle_contact_push = Vec2::ZERO;
                    if is_water {
                        for &j in neighbors {
                            if i == j || !is_water_particle(materials[j]) {
                                continue;
                            }
                            let r = positions[i] - positions[j];
                            if r.length_squared() >= h_ww * h_ww {
                                continue;
                            }
                            delta += (lambdas[i] + lambdas[j]) * kernel_spiky_grad(r, h_ww);
                        }
                        if let Some(ghost_r) =
                            terrain_ghost_neighbor_vector(positions[i], terrain, h_ww)
                        {
                            delta += lambdas[i]
                                * kernel_spiky_grad(ghost_r, h_ww)
                                * TERRAIN_GHOST_DELTA_SCALE;
                        }
                    }
                    for &j in neighbors {
                        if i == j {
                            continue;
                        }
                        let neighbor_material = materials[j];
                        if is_water && is_water_particle(neighbor_material) {
                            continue;
                        }
                        let neighbor_props = particle_properties(neighbor_material);
                        let r = positions[i] - positions[j];
                        let dist2 = r.length_squared();
                        let contact_radius = props.radius_m + neighbor_props.radius_m;
                        if dist2 <= 1e-12 || dist2 >= contact_radius * contact_radius {
                            continue;
                        }
                        let dist = dist2.sqrt();
                        let normal = r / dist;
                        let penetration = contact_radius - dist;
                        let stiffness = 0.5
                            * (props.object_repulsion_stiffness
                                + neighbor_props.object_repulsion_stiffness);
                        let normal_push =
                            normal * penetration * stiffness * PARTICLE_CONTACT_PUSH_FACTOR;
                        particle_contact_push += normal_push;
                        if let Some(contact_pair_scale) =
                            granular_contact_friction_pair_scale(material, neighbor_material)
                        {
                            let static_friction = 0.5
                                * (props.friction_static + neighbor_props.friction_static)
                                * GRANULAR_CONTACT_FRICTION_SCALE
                                * contact_pair_scale;
                            let dynamic_friction = 0.5
                                * (props.friction_dynamic + neighbor_props.friction_dynamic)
                                * GRANULAR_CONTACT_FRICTION_SCALE
                                * contact_pair_scale;
                            let rel_velocity = velocities[i] - velocities[j];
                            particle_contact_push += granular_tangential_friction_push(
                                normal,
                                normal_push,
                                rel_velocity,
                                static_friction,
                                dynamic_friction,
                                dt_sub,
                            );
                            let rel_normal_speed = (velocities[i] - velocities[j]).dot(normal);
                            if rel_normal_speed < 0.0 {
                                let normal_damping_push = -normal
                                    * rel_normal_speed
                                    * dt_sub
                                    * GRANULAR_CONTACT_NORMAL_DAMPING
                                    * contact_pair_scale;
                                let max_damping_push = penetration * 0.35;
                                particle_contact_push +=
                                    normal_damping_push.clamp_length_max(max_damping_push);
                            }
                        }
                    }

                    if let Some((signed_distance, normal)) =
                        terrain.sample_signed_distance_and_normal(positions[i])
                    {
                        let penetration = props.terrain_push_radius_m - signed_distance;
                        if penetration > 0.0 {
                            let normal_push =
                                normal * penetration * props.terrain_repulsion_stiffness;
                            boundary_push += normal_push;
                            if !is_water {
                                boundary_push += granular_tangential_friction_push(
                                    normal,
                                    normal_push,
                                    velocities[i],
                                    props.friction_static * TERRAIN_CONTACT_FRICTION_SCALE,
                                    props.friction_dynamic * TERRAIN_CONTACT_FRICTION_SCALE,
                                    dt_sub,
                                );
                                let normal_speed = velocities[i].dot(normal);
                                if normal_speed < 0.0 {
                                    let normal_damping_push = -normal
                                        * normal_speed
                                        * dt_sub
                                        * TERRAIN_CONTACT_NORMAL_DAMPING;
                                    boundary_push +=
                                        normal_damping_push.clamp_length_max(penetration * 0.35);
                                }
                            }
                        }
                    }

                    let mut object_push = Vec2::ZERO;
                    let self_object = object_world.object_of_particle(i);
                    if is_water || self_object.is_some() {
                        object_field.gather_candidate_object_ids(
                            positions[i],
                            &mut scratch.object_contacts,
                        );

                        for contact_index in 0..scratch.object_contacts.len() {
                            let object_id = scratch.object_contacts[contact_index];
                            if self_object == Some(object_id) {
                                continue;
                            }
                            let Some(sample) =
                                object_world.evaluate_object_sdf(object_id, positions[i])
                            else {
                                continue;
                            };
                            let penetration = props.object_push_radius_m - sample.distance_m;
                            if penetration <= 0.0 {
                                continue;
                            }
                            let push = sample.normal_world
                                * penetration
                                * props.object_repulsion_stiffness;
                            object_push += push;
                            let reaction_impulse = -(masses[i] * push) * inv_dt;
                            scratch.accumulate_impulse(object_id, reaction_impulse);
                        }
                    }

                    let max_push = props.object_push_radius_m;
                    object_push = object_push.clamp_length_max(max_push);
                    *delta_pos_i = if is_water {
                        delta / REST_DENSITY + boundary_push + object_push + particle_contact_push
                    } else {
                        boundary_push + object_push + particle_contact_push
                    };

                    scratch
                },
            )
            .reduce(ComputeDeltaThreadScratch::default, |mut a, b| {
                a.merge_from(b);
                a
            });

        reduced.reaction_impulses
    }

    fn apply_contact_velocity_response(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
    ) {
        let mut object_contacts = Vec::new();
        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) {
                continue;
            }
            let material = self.material[i];
            let props = particle_properties(material);
            if !props.apply_contact_velocity_response {
                continue;
            }

            let mut normal_sum = Vec2::ZERO;
            let mut contact_count = 0usize;

            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            {
                let penetration = props.terrain_push_radius_m - signed_distance;
                if penetration > 0.0 {
                    normal_sum += normal;
                    contact_count += 1;
                }
            }

            let self_object = object_world.object_of_particle(i);
            object_field.gather_candidate_object_ids(self.pos[i], &mut object_contacts);
            for &object_id in &object_contacts {
                if self_object == Some(object_id) {
                    continue;
                }
                let Some(sample) = object_world.evaluate_object_sdf(object_id, self.pos[i]) else {
                    continue;
                };
                let penetration = props.object_push_radius_m - sample.distance_m;
                if penetration <= 0.0 {
                    continue;
                }
                normal_sum += sample.normal_world;
                contact_count += 1;
            }

            if contact_count == 0 {
                continue;
            }
            let contact_normal = normal_sum.normalize_or_zero();
            if contact_normal == Vec2::ZERO {
                continue;
            }

            let normal_speed = self.vel[i].dot(contact_normal);
            let tangential_velocity = self.vel[i] - contact_normal * normal_speed;
            let normal_contact_speed = normal_speed.abs();
            let tangential_speed = tangential_velocity.length();
            let adjusted_tangent = if tangential_speed <= 1e-6 {
                Vec2::ZERO
            } else if tangential_speed <= props.friction_static * normal_contact_speed {
                Vec2::ZERO
            } else {
                let reduced_speed =
                    (tangential_speed - props.friction_dynamic * normal_contact_speed).max(0.0);
                tangential_velocity * (reduced_speed / tangential_speed)
            };
            let adjusted_normal_speed = if normal_speed < 0.0 {
                -normal_speed * props.contact_restitution.clamp(0.0, 1.0)
            } else {
                normal_speed
            };
            self.vel[i] = adjusted_tangent + contact_normal * adjusted_normal_speed;
        }
    }

    fn apply_xsph_viscosity(&mut self) {
        let _span = tracing::info_span!("physics::xsph_pass").entered();
        self.viscosity_work.clone_from(&self.vel);
        let count = self.particle_count();
        let use_parallel = self.parallel_enabled && count >= PARALLEL_PARTICLE_THRESHOLD;

        if use_parallel {
            let positions = &self.pos;
            let velocities = &self.vel;
            let materials = &self.material;
            let activity = &self.activity_state;
            let neighbor_grid = &self.neighbor_grid;
            self.viscosity_work
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out)| {
                    let active = activity
                        .get(i)
                        .copied()
                        .unwrap_or(ParticleActivityState::Active)
                        == ParticleActivityState::Active;
                    if !active || !is_water_particle(materials[i]) {
                        *out = velocities[i];
                        return;
                    }
                    let mut neighbors = Vec::new();
                    neighbor_grid.gather(positions[i], &mut neighbors);

                    let mut correction = Vec2::ZERO;
                    for &j in &neighbors {
                        if i == j || !is_water_particle(materials[j]) {
                            continue;
                        }
                        let r = positions[i] - positions[j];
                        let w = kernel_poly6(r.length_squared(), WATER_KERNEL_RADIUS_M);
                        correction += (velocities[j] - velocities[i]) * w;
                    }
                    let viscosity = particle_properties(materials[i]).xsph_viscosity;
                    *out = velocities[i] + viscosity * correction;
                });
        } else {
            let mut neighbors = Vec::new();
            for i in 0..count {
                if !self.is_active_particle(i) || !is_water_particle(self.material[i]) {
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
                let viscosity = particle_properties(self.material[i]).xsph_viscosity;
                self.viscosity_work[i] = self.vel[i] + viscosity * correction;
            }
        }

        self.vel.clone_from(&self.viscosity_work);
    }

    fn apply_object_reaction_impulses(&mut self, object_world: &ObjectWorld) {
        object_solver::apply_object_reaction_impulses(self, object_world);
    }

    fn detect_fracture_candidates(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
    ) {
        object_solver::detect_fracture_candidates(self, terrain, object_field, object_world);
    }

    fn apply_object_fractures(&mut self, object_world: &mut ObjectWorld) {
        object_solver::apply_object_fractures(self, object_world);
    }

    fn solve_shape_matching_constraints(&mut self, object_world: &mut ObjectWorld) {
        object_solver::solve_shape_matching_constraints(self, object_world);
    }

    fn project_solids_out_of_terrain(&mut self, terrain: &TerrainWorld) {
        for i in 0..self.particle_count() {
            if !self.is_active_particle(i) || is_water_particle(self.material[i]) {
                continue;
            }
            let props = particle_properties(self.material[i]);
            let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            else {
                continue;
            };
            let penetration = props.terrain_push_radius_m - signed_distance;
            if penetration > 0.0 {
                self.pos[i] += normal * penetration;
            }
        }
    }

    fn push_water_particle(&mut self, position: Vec2, velocity: Vec2) {
        self.pos.push(position);
        self.prev_pos.push(position);
        self.vel.push(velocity);
        self.mass.push(default_particle_mass());
        self.material.push(ParticleMaterial::WaterLiquid);
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
        let particle_mass = props.mass;
        let jitter_scale = spacing * GRANULAR_SPAWN_JITTER_RATIO;
        let min_bound = cell_min + Vec2::splat(props.radius_m);
        let max_bound = cell_min + Vec2::splat(CELL_SIZE_M - props.radius_m);
        let granular_spawn = is_granular_particle(material) && jitter_scale > 0.0;

        for y in 0..axis {
            for x in 0..axis {
                let local = Vec2::new((x as f32 + 0.5) * spacing, (y as f32 + 0.5) * spacing);
                let mut position = cell_min + local;
                if granular_spawn {
                    position += spawn_jitter_offset(cell, material, x, y, jitter_scale);
                    position = Vec2::new(
                        position.x.clamp(min_bound.x, max_bound.x),
                        position.y.clamp(min_bound.y, max_bound.y),
                    );
                }
                let index = self.pos.len();
                self.pos.push(position);
                self.prev_pos.push(position);
                self.vel.push(velocity);
                self.mass.push(particle_mass);
                self.material.push(material);
                out_indices.push(index);
            }
        }
    }

    fn fracture_particle_to_target_material(
        &mut self,
        index: usize,
        target_material: ParticleMaterial,
        spawned_particles: &mut Vec<(Vec2, Vec2, f32, ParticleMaterial)>,
    ) {
        if index >= self.particle_count() {
            return;
        }
        let target_props = particle_properties(target_material);
        let target_mass = target_props.mass;
        let split_count = target_props.particles_per_cell.max(1) as usize;
        let base_pos = self.pos[index];
        let base_vel = self.vel[index];
        self.material[index] = target_material;
        self.mass[index] = target_mass;
        if is_granular_particle(target_material) {
            self.request_wake(index);
            self.sleep_lock_frames[index] = FRACTURE_WAKE_LOCK_FRAMES;
        }
        if split_count <= 1 {
            return;
        }

        let axis = (split_count as f32).sqrt().ceil() as usize;
        let spacing = CELL_SIZE_M / axis as f32;
        let cell_min = base_pos - Vec2::splat(CELL_SIZE_M * 0.5);
        let mut placed = 0usize;
        'grid: for y in 0..axis {
            for x in 0..axis {
                if placed >= split_count {
                    break 'grid;
                }
                let split_pos =
                    cell_min + Vec2::new((x as f32 + 0.5) * spacing, (y as f32 + 0.5) * spacing);
                if placed == 0 {
                    self.pos[index] = split_pos;
                    self.prev_pos[index] = split_pos;
                } else {
                    spawned_particles.push((split_pos, base_vel, target_mass, target_material));
                }
                placed += 1;
            }
        }
    }

    fn auto_fracture_single_cell_objects(&mut self, object_world: &mut ObjectWorld) {
        let mut objects_to_remove = HashSet::new();
        let mut targets = Vec::<(usize, ParticleMaterial)>::new();
        for object in object_world.objects() {
            if object.particle_indices.is_empty() {
                continue;
            }
            let mut occupied_cells = HashSet::new();
            for &index in &object.particle_indices {
                if index >= self.particle_count() {
                    continue;
                }
                occupied_cells.insert(world_to_cell(self.pos[index]));
                if occupied_cells.len() > 1 {
                    break;
                }
            }
            if occupied_cells.len() != 1 {
                continue;
            }
            let mut has_fracturable = false;
            for &index in &object.particle_indices {
                if index >= self.particle_count() {
                    continue;
                }
                let Some(target_material) = solid_break_properties(self.material[index])
                    .and_then(|props| props.fracture_to)
                else {
                    continue;
                };
                has_fracturable = true;
                targets.push((index, target_material));
            }
            if has_fracturable {
                objects_to_remove.insert(object.id);
            }
        }
        if targets.is_empty() {
            return;
        }

        let mut spawned_particles = Vec::new();
        for (index, target_material) in targets {
            self.fracture_particle_to_target_material(
                index,
                target_material,
                &mut spawned_particles,
            );
        }
        for (position, velocity, mass, material) in spawned_particles {
            let new_index = self.pos.len();
            self.pos.push(position);
            self.prev_pos.push(position);
            self.vel.push(velocity);
            self.mass.push(mass);
            self.material.push(material);
            if is_granular_particle(material) {
                self.request_wake(new_index);
                self.sleep_lock_frames[new_index] = FRACTURE_WAKE_LOCK_FRAMES;
            }
        }
        self.resize_work_buffers();
        object_world.remove_objects_by_ids(&objects_to_remove);
    }

    pub(crate) fn is_active_particle(&self, index: usize) -> bool {
        let active = self
            .activity_state
            .get(index)
            .copied()
            .unwrap_or(ParticleActivityState::Active)
            == ParticleActivityState::Active;
        if !active {
            return false;
        }
        let (Some(chunk_min), Some(chunk_max)) = (self.active_chunk_min, self.active_chunk_max)
        else {
            return true;
        };
        if index >= self.particle_count() {
            return false;
        }
        let cell = world_to_cell(self.pos[index]);
        let chunk = IVec2::new(
            cell.x.div_euclid(CHUNK_SIZE_I32),
            cell.y.div_euclid(CHUNK_SIZE_I32),
        );
        chunk.x >= chunk_min.x
            && chunk.x <= chunk_max.x
            && chunk.y >= chunk_min.y
            && chunk.y <= chunk_max.y
    }

    fn is_halo_particle(&self, index: usize) -> bool {
        if index >= self.particle_count() {
            return false;
        }
        let halo = self.active_halo_chunks.max(0);
        if halo == 0 {
            return false;
        }
        let (Some(chunk_min), Some(chunk_max)) = (self.active_chunk_min, self.active_chunk_max)
        else {
            return false;
        };
        let cell = world_to_cell(self.pos[index]);
        let chunk = IVec2::new(
            cell.x.div_euclid(CHUNK_SIZE_I32),
            cell.y.div_euclid(CHUNK_SIZE_I32),
        );

        let inside_active = chunk.x >= chunk_min.x
            && chunk.x <= chunk_max.x
            && chunk.y >= chunk_min.y
            && chunk.y <= chunk_max.y;
        if inside_active {
            return false;
        }
        let expanded_min = chunk_min - IVec2::splat(halo);
        let expanded_max = chunk_max + IVec2::splat(halo);
        chunk.x >= expanded_min.x
            && chunk.x <= expanded_max.x
            && chunk.y >= expanded_min.y
            && chunk.y <= expanded_max.y
    }

    fn is_active_or_halo_particle(&self, index: usize) -> bool {
        self.is_active_particle(index) || self.is_halo_particle(index)
    }

    pub(crate) fn request_wake(&mut self, index: usize) {
        if index >= self.particle_count() {
            return;
        }
        if self.pending_wake.len() < self.particle_count() {
            self.pending_wake.resize(self.particle_count(), false);
        }
        if self.activity_state.len() < self.particle_count() {
            self.activity_state
                .resize(self.particle_count(), ParticleActivityState::Active);
        }
        if self.sleep_candidate_frames.len() < self.particle_count() {
            self.sleep_candidate_frames.resize(self.particle_count(), 0);
        }
        if self.active_hold_frames.len() < self.particle_count() {
            self.active_hold_frames.resize(self.particle_count(), 0);
        }
        if self.sleep_lock_frames.len() < self.particle_count() {
            self.sleep_lock_frames.resize(self.particle_count(), 0);
        }
        self.pending_wake[index] = true;
    }

    fn request_wake_near(&mut self, center: Vec2, radius: f32) {
        let radius2 = radius * radius;
        for i in 0..self.particle_count() {
            if self.pos[i].distance_squared(center) <= radius2 {
                self.request_wake(i);
            }
        }
    }

    fn propagate_and_apply_wake_requests(&mut self) {
        let wake_radius2 = WAKE_RADIUS * WAKE_RADIUS;
        let mut seeds = Vec::new();
        for i in 0..self.particle_count() {
            if self.pending_wake[i] {
                seeds.push(i);
            }
        }

        let mut neighbors = Vec::new();
        for &seed in &seeds {
            self.neighbor_grid.gather(self.pos[seed], &mut neighbors);
            for &j in &neighbors {
                if j == seed || self.is_active_particle(j) {
                    continue;
                }
                if self.pos[seed].distance_squared(self.pos[j]) <= wake_radius2 {
                    self.pending_wake[j] = true;
                }
            }
        }

        for i in 0..self.particle_count() {
            if !self.pending_wake[i] {
                continue;
            }
            self.activity_state[i] = ParticleActivityState::Active;
            self.sleep_candidate_frames[i] = 0;
            self.active_hold_frames[i] = ACTIVE_MIN_FRAMES;
            self.pending_wake[i] = false;
        }
    }

    fn update_sleep_states(&mut self, object_world: &ObjectWorld) {
        for i in 0..self.particle_count() {
            // Keep object-owned particles active as a group.
            // Per-particle sleep inside an object can desynchronize shape matching and
            // create outlier particles that later inject large impulses.
            if object_world.object_of_particle(i).is_some() {
                self.activity_state[i] = ParticleActivityState::Active;
                self.sleep_candidate_frames[i] = 0;
                if self.active_hold_frames[i] == 0 {
                    self.active_hold_frames[i] = 1;
                }
                continue;
            }
            if self.pending_wake[i] {
                continue;
            }
            if self.activity_state[i] == ParticleActivityState::Sleeping {
                self.vel[i] = Vec2::ZERO;
                self.prev_pos[i] = self.pos[i];
                continue;
            }

            if self.active_hold_frames[i] > 0 {
                self.active_hold_frames[i] -= 1;
                self.sleep_candidate_frames[i] = 0;
                continue;
            }
            if self.sleep_lock_frames[i] > 0 {
                self.sleep_lock_frames[i] -= 1;
                self.sleep_candidate_frames[i] = 0;
                continue;
            }

            let displacement = (self.pos[i] - self.prev_pos[i]).length();
            let speed = self.vel[i].length();
            let should_sleep_sample =
                displacement < SLEEP_DISP_THRESHOLD && speed < SLEEP_VEL_THRESHOLD;

            if should_sleep_sample {
                self.sleep_candidate_frames[i] = self.sleep_candidate_frames[i].saturating_add(1);
                if self.sleep_candidate_frames[i] >= SLEEP_FRAMES {
                    self.activity_state[i] = ParticleActivityState::Sleeping;
                    self.sleep_candidate_frames[i] = 0;
                    self.vel[i] = Vec2::ZERO;
                    self.prev_pos[i] = self.pos[i];
                }
            } else {
                self.sleep_candidate_frames[i] = 0;
            }
        }
    }

    fn detect_wake_events(
        &mut self,
        terrain: &TerrainWorld,
        object_field: &ObjectPhysicsField,
        object_world: &ObjectWorld,
    ) {
        let count = self.particle_count();
        if count == 0 {
            return;
        }

        #[derive(Default)]
        struct WakeDetectAccum {
            neighbors: Vec<usize>,
            object_contacts: Vec<ObjectId>,
            wake_indices: Vec<usize>,
            wake_centers: Vec<Vec2>,
        }

        let use_parallel = self.parallel_enabled && count >= PARALLEL_PARTICLE_THRESHOLD;
        let mut wake_indices = Vec::new();
        let mut wake_centers = Vec::new();
        if use_parallel {
            let positions = &self.pos;
            let prev_positions = &self.prev_pos;
            let materials = &self.material;
            let activity = &self.activity_state;
            let neighbor_grid = &self.neighbor_grid;
            let object_particles_by_id: HashMap<ObjectId, Vec<usize>> = object_world
                .objects()
                .iter()
                .map(|object| (object.id, object.particle_indices.clone()))
                .collect();
            let reduced = (0..count)
                .into_par_iter()
                .fold(WakeDetectAccum::default, |mut local, i| {
                    let active = activity
                        .get(i)
                        .copied()
                        .unwrap_or(ParticleActivityState::Active)
                        == ParticleActivityState::Active;
                    if !active {
                        return local;
                    }

                    let displacement_i = positions[i] - prev_positions[i];
                    if displacement_i.length() > WAKE_DISP_THRESHOLD {
                        local.wake_indices.push(i);
                    }

                    neighbor_grid.gather(positions[i], &mut local.neighbors);
                    for &j in &local.neighbors {
                        let neighbor_active = activity
                            .get(j)
                            .copied()
                            .unwrap_or(ParticleActivityState::Active)
                            == ParticleActivityState::Active;
                        if i == j || neighbor_active {
                            continue;
                        }
                        let r = positions[i] - positions[j];
                        let contact_radius = particle_properties(materials[i]).radius_m
                            + particle_properties(materials[j]).radius_m;
                        let dist2 = r.length_squared();
                        if dist2 <= 1e-12 || dist2 >= contact_radius * contact_radius {
                            continue;
                        }
                        let dist = dist2.sqrt();
                        let normal = r / dist;
                        let displacement_j = positions[j] - prev_positions[j];
                        let relative_normal_displacement =
                            (displacement_i - displacement_j).dot(normal).abs();
                        if relative_normal_displacement > WAKE_DISP_THRESHOLD {
                            local.wake_indices.push(i);
                            local.wake_indices.push(j);
                        }
                    }
                    local.neighbors.clear();

                    if let Some((signed_distance, normal)) =
                        terrain.sample_signed_distance_and_normal(positions[i])
                    {
                        let props = particle_properties(materials[i]);
                        let penetration = props.terrain_push_radius_m - signed_distance;
                        if penetration > 0.0 {
                            let normal_displacement = displacement_i.dot(normal).abs();
                            if normal_displacement > WAKE_DISP_THRESHOLD {
                                local.wake_indices.push(i);
                                local.wake_centers.push(positions[i]);
                            }
                        }
                    }

                    let self_object = object_world.object_of_particle(i);
                    object_field
                        .gather_candidate_object_ids(positions[i], &mut local.object_contacts);
                    for &object_id in &local.object_contacts {
                        if self_object == Some(object_id) {
                            continue;
                        }
                        let Some(sample) =
                            object_world.evaluate_object_sdf(object_id, positions[i])
                        else {
                            continue;
                        };
                        let props = particle_properties(materials[i]);
                        let penetration = props.object_push_radius_m - sample.distance_m;
                        if penetration <= 0.0 {
                            continue;
                        }
                        let normal_displacement = displacement_i.dot(sample.normal_world).abs();
                        if normal_displacement <= WAKE_DISP_THRESHOLD {
                            continue;
                        }
                        local.wake_indices.push(i);
                        if let Some(target_particles) = object_particles_by_id.get(&object_id) {
                            local.wake_indices.extend(target_particles.iter().copied());
                        }
                    }
                    local.object_contacts.clear();
                    local
                })
                .reduce(WakeDetectAccum::default, |mut a, mut b| {
                    a.wake_indices.append(&mut b.wake_indices);
                    a.wake_centers.append(&mut b.wake_centers);
                    a
                });
            wake_indices = reduced.wake_indices;
            wake_centers = reduced.wake_centers;
        } else {
            let mut neighbors = Vec::new();
            let mut object_contacts = Vec::new();
            let object_particles_by_id: HashMap<ObjectId, Vec<usize>> = object_world
                .objects()
                .iter()
                .map(|object| (object.id, object.particle_indices.clone()))
                .collect();
            for i in 0..count {
                if !self.is_active_particle(i) {
                    continue;
                }
                let displacement_i = self.pos[i] - self.prev_pos[i];
                if displacement_i.length() > WAKE_DISP_THRESHOLD {
                    wake_indices.push(i);
                }

                self.neighbor_grid.gather(self.pos[i], &mut neighbors);
                for &j in &neighbors {
                    if i == j || self.is_active_particle(j) {
                        continue;
                    }
                    let r = self.pos[i] - self.pos[j];
                    let contact_radius = particle_properties(self.material[i]).radius_m
                        + particle_properties(self.material[j]).radius_m;
                    let dist2 = r.length_squared();
                    if dist2 <= 1e-12 || dist2 >= contact_radius * contact_radius {
                        continue;
                    }
                    let dist = dist2.sqrt();
                    let normal = r / dist;
                    let displacement_j = self.pos[j] - self.prev_pos[j];
                    let relative_normal_displacement =
                        (displacement_i - displacement_j).dot(normal).abs();
                    if relative_normal_displacement > WAKE_DISP_THRESHOLD {
                        wake_indices.push(i);
                        wake_indices.push(j);
                    }
                }

                if let Some((signed_distance, normal)) =
                    terrain.sample_signed_distance_and_normal(self.pos[i])
                {
                    let props = particle_properties(self.material[i]);
                    let penetration = props.terrain_push_radius_m - signed_distance;
                    if penetration > 0.0 {
                        let normal_displacement = displacement_i.dot(normal).abs();
                        if normal_displacement > WAKE_DISP_THRESHOLD {
                            wake_indices.push(i);
                            wake_centers.push(self.pos[i]);
                        }
                    }
                }

                let self_object = object_world.object_of_particle(i);
                object_field.gather_candidate_object_ids(self.pos[i], &mut object_contacts);
                for &object_id in &object_contacts {
                    if self_object == Some(object_id) {
                        continue;
                    }
                    let Some(sample) = object_world.evaluate_object_sdf(object_id, self.pos[i])
                    else {
                        continue;
                    };
                    let props = particle_properties(self.material[i]);
                    let penetration = props.object_push_radius_m - sample.distance_m;
                    if penetration <= 0.0 {
                        continue;
                    }
                    let normal_displacement = displacement_i.dot(sample.normal_world).abs();
                    if normal_displacement <= WAKE_DISP_THRESHOLD {
                        continue;
                    }
                    wake_indices.push(i);
                    if let Some(target_particles) = object_particles_by_id.get(&object_id) {
                        wake_indices.extend(target_particles.iter().copied());
                    }
                }
            }
        }

        wake_indices.sort_unstable();
        wake_indices.dedup();
        for index in wake_indices {
            self.request_wake(index);
        }
        for center in wake_centers {
            self.request_wake_near(center, WAKE_RADIUS);
        }

        // Even outside the current active simulation region, particles can drift ballistically.
        // If they start penetrating terrain/objects, promote them to active so the region can expand.
        for i in 0..count {
            if self.is_active_particle(i) {
                continue;
            }
            let displacement_i = self.pos[i] - self.prev_pos[i];
            if displacement_i.length() <= WAKE_DISP_THRESHOLD {
                continue;
            }
            let props = particle_properties(self.material[i]);

            let mut should_wake = false;
            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(self.pos[i])
            {
                let penetration = props.terrain_push_radius_m - signed_distance;
                if penetration > 0.0 && displacement_i.dot(normal).abs() > WAKE_DISP_THRESHOLD {
                    should_wake = true;
                }
            }

            if !should_wake {
                let mut object_contacts = Vec::new();
                let self_object = object_world.object_of_particle(i);
                object_field.gather_candidate_object_ids(self.pos[i], &mut object_contacts);
                for &object_id in &object_contacts {
                    if self_object == Some(object_id) {
                        continue;
                    }
                    let Some(sample) = object_world.evaluate_object_sdf(object_id, self.pos[i])
                    else {
                        continue;
                    };
                    let penetration = props.object_push_radius_m - sample.distance_m;
                    if penetration > 0.0
                        && displacement_i.dot(sample.normal_world).abs() > WAKE_DISP_THRESHOLD
                    {
                        should_wake = true;
                        break;
                    }
                }
            }

            if should_wake {
                self.request_wake(i);
            }
        }
    }

    fn capture_far_field_intrusions(&mut self, object_world: &mut ObjectWorld) {
        let Some(center_chunk) = self.far_field_center_chunk else {
            return;
        };
        let active_radius = self.far_field_release_radius_chunks.max(0);
        let live_radius = active_radius + self.active_halo_chunks.max(0);
        let freeze_radius = self.far_field_freeze_radius_chunks.max(0);
        if self.particle_count() == 0 {
            return;
        }

        let old_count = self.particle_count();
        let mut keep = vec![true; old_count];
        let mut removed_count = 0usize;
        for i in 0..old_count {
            if object_world.object_of_particle(i).is_some() {
                continue;
            }
            let chunk = world_pos_to_chunk(self.pos[i]);
            if !is_chunk_outside_radius(chunk, center_chunk, active_radius) {
                continue;
            }
            let prev_chunk = world_pos_to_chunk(self.prev_pos[i]);
            let crossed_active_boundary =
                !is_chunk_outside_radius(prev_chunk, center_chunk, active_radius);
            let outside_live = is_chunk_outside_radius(chunk, center_chunk, live_radius);
            let outside_freeze = is_chunk_outside_radius(chunk, center_chunk, freeze_radius);

            if crossed_active_boundary && !outside_freeze {
                let (edge, boundary_position, grid_segment) = edge_sample_for_active_boundary(
                    chunk,
                    self.pos[i],
                    center_chunk,
                    active_radius,
                );
                self.deferred_boundary_particles
                    .entry(DeferredEdgeKey {
                        chunk,
                        edge,
                        grid_segment,
                    })
                    .or_default()
                    .push_back(DeferredParticleState {
                        position: boundary_position,
                        velocity: self.vel[i],
                        mass: self.mass[i],
                        material: self.material[i],
                    });
                keep[i] = false;
                removed_count += 1;
            } else if outside_live && outside_freeze {
                self.deferred_inactive_chunk_particles
                    .entry(chunk)
                    .or_default()
                    .push_back(DeferredParticleState {
                        position: self.pos[i],
                        velocity: self.vel[i],
                        mass: self.mass[i],
                        material: self.material[i],
                    });
                keep[i] = false;
                removed_count += 1;
            }
        }

        if removed_count > 0 {
            self.compact_particles_with_keep_mask(&keep, object_world);
        }
    }

    fn release_far_field_particles_for_active_chunks(&mut self) {
        let Some(center_chunk) = self.far_field_center_chunk else {
            return;
        };
        let active_radius = self.far_field_release_radius_chunks.max(0);
        let mut budget = self.far_field_release_particles_per_frame;
        let clearance_radius = self.far_field_release_clearance_radius_m.max(0.0);
        let clearance2 = clearance_radius * clearance_radius;
        let clearance_wait_max = self.far_field_release_clearance_max_wait_frames;
        let mut released_count = 0usize;
        let mut neighbors = Vec::new();

        if !self.pos.is_empty() {
            self.neighbor_grid.rebuild(&self.pos);
        }

        let mut chunk_keys_to_release: Vec<IVec2> = self
            .deferred_inactive_chunk_particles
            .keys()
            .copied()
            .filter(|&chunk| !is_chunk_outside_radius(chunk, center_chunk, active_radius))
            .collect();
        chunk_keys_to_release.sort_by_key(|chunk| (chunk.y, chunk.x));
        for chunk in chunk_keys_to_release {
            loop {
                let state = {
                    let Some(queue) = self.deferred_inactive_chunk_particles.get_mut(&chunk) else {
                        break;
                    };
                    queue.pop_front()
                };
                let Some(state) = state else {
                    break;
                };
                let new_index = self.pos.len();
                self.pos.push(state.position);
                self.prev_pos.push(state.position);
                self.vel
                    .push(state.velocity.clamp_length_max(PARTICLE_SPEED_LIMIT_MPS));
                self.mass.push(state.mass);
                self.material.push(state.material);
                self.request_wake(new_index);
                self.active_hold_frames[new_index] = ACTIVE_MIN_FRAMES;
                released_count += 1;
            }
        }

        self.deferred_inactive_chunk_particles
            .retain(|_, queue| !queue.is_empty());
        if clearance2 > 0.0 && !self.pos.is_empty() {
            self.neighbor_grid.rebuild(&self.pos);
        }

        if budget == 0 {
            if released_count > 0 {
                self.resize_work_buffers();
            }
            return;
        }

        let mut keys_to_release: Vec<DeferredEdgeKey> = self
            .deferred_boundary_particles
            .keys()
            .copied()
            .filter(|key| !is_chunk_outside_radius(key.chunk, center_chunk, active_radius))
            .collect();
        keys_to_release.sort_by_key(|key| {
            (
                key.chunk.y,
                key.chunk.x,
                edge_sort_key(key.edge),
                key.grid_segment,
            )
        });

        for key in keys_to_release {
            if budget == 0 {
                break;
            }
            let front = self
                .deferred_boundary_particles
                .get(&key)
                .and_then(|queue| queue.front())
                .copied();
            let Some(state) = front else {
                continue;
            };

            if clearance2 > 0.0 {
                let wait_center = self
                    .deferred_boundary_release_locks
                    .get(&key)
                    .map(|lock| lock.position)
                    .unwrap_or(state.position);
                if self.has_particle_within_radius(wait_center, clearance2, &mut neighbors) {
                    let force_release =
                        if let Some(lock) = self.deferred_boundary_release_locks.get_mut(&key) {
                            if lock.wait_frames >= clearance_wait_max {
                                true
                            } else {
                                lock.wait_frames = lock.wait_frames.saturating_add(1);
                                false
                            }
                        } else if clearance_wait_max == 0 {
                            true
                        } else {
                            self.deferred_boundary_release_locks.insert(
                                key,
                                BoundaryReleaseLock {
                                    position: wait_center,
                                    wait_frames: 1,
                                },
                            );
                            false
                        };
                    if !force_release {
                        continue;
                    }
                }
                self.deferred_boundary_release_locks.remove(&key);
            }

            let popped = self
                .deferred_boundary_particles
                .get_mut(&key)
                .and_then(|queue| queue.pop_front());
            let Some(state) = popped else {
                continue;
            };
            let new_index = self.pos.len();
            self.pos.push(state.position);
            self.prev_pos.push(state.position);
            self.vel
                .push(state.velocity.clamp_length_max(PARTICLE_SPEED_LIMIT_MPS));
            self.mass.push(state.mass);
            self.material.push(state.material);
            self.request_wake(new_index);
            self.active_hold_frames[new_index] = ACTIVE_MIN_FRAMES;
            if clearance2 > 0.0 {
                self.deferred_boundary_release_locks.insert(
                    key,
                    BoundaryReleaseLock {
                        position: state.position,
                        wait_frames: 0,
                    },
                );
            }
            budget -= 1;
            released_count += 1;
        }

        self.deferred_boundary_particles
            .retain(|_, queue| !queue.is_empty());
        self.deferred_boundary_release_locks
            .retain(|key, _| self.deferred_boundary_particles.contains_key(key));
        if released_count > 0 {
            self.resize_work_buffers();
        }
    }

    fn cull_escaped_particles(&mut self, terrain: &TerrainWorld, object_world: &mut ObjectWorld) {
        if self.particle_count() == 0 {
            return;
        }

        let (loaded_min_cell, loaded_max_cell) = terrain.loaded_cell_bounds().unwrap_or((
            IVec2::new(-CHUNK_SIZE_I32, -CHUNK_SIZE_I32),
            IVec2::new(CHUNK_SIZE_I32, CHUNK_SIZE_I32),
        ));
        let min_cell_x = loaded_min_cell.x - PARTICLE_ESCAPE_MARGIN_X_CELLS;
        let max_cell_x = loaded_max_cell.x + PARTICLE_ESCAPE_MARGIN_X_CELLS;
        let min_cell_y = loaded_min_cell.y - PARTICLE_ESCAPE_MARGIN_BOTTOM_CELLS;
        let max_cell_y = loaded_max_cell.y + PARTICLE_ESCAPE_MARGIN_TOP_CELLS;

        let min_world = cell_to_world_center(IVec2::new(min_cell_x, min_cell_y))
            - Vec2::splat(CELL_SIZE_M * 0.5);
        let max_world = cell_to_world_center(IVec2::new(max_cell_x, max_cell_y))
            + Vec2::splat(CELL_SIZE_M * 0.5);

        let old_count = self.particle_count();
        let mut keep = vec![true; old_count];
        for (index, position) in self.pos.iter().enumerate() {
            let out_of_bounds = !position.is_finite()
                || position.x < min_world.x
                || position.x > max_world.x
                || position.y < min_world.y
                || position.y > max_world.y;
            if out_of_bounds {
                keep[index] = false;
            }
        }
        self.compact_particles_with_keep_mask(&keep, object_world);
    }

    fn has_particle_within_radius(
        &self,
        center: Vec2,
        radius2: f32,
        neighbors: &mut Vec<usize>,
    ) -> bool {
        if radius2 <= 0.0 || self.pos.is_empty() {
            return false;
        }
        self.neighbor_grid.gather(center, neighbors);
        for &index in neighbors.iter() {
            if index >= self.pos.len() {
                continue;
            }
            if self.pos[index].distance_squared(center) <= radius2 {
                return true;
            }
        }
        false
    }

    fn compact_particles_with_keep_mask(&mut self, keep: &[bool], object_world: &mut ObjectWorld) {
        let old_count = self.particle_count();
        if keep.len() != old_count {
            return;
        }

        let removed_count = keep.iter().filter(|&&k| !k).count();
        if removed_count == 0 {
            return;
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
        let mut new_activity_state = Vec::with_capacity(new_count);
        let mut new_sleep_candidate_frames = Vec::with_capacity(new_count);
        let mut new_active_hold_frames = Vec::with_capacity(new_count);
        let mut new_sleep_lock_frames = Vec::with_capacity(new_count);
        let mut new_pending_wake = Vec::with_capacity(new_count);
        for old_index in 0..old_count {
            if !keep[old_index] {
                continue;
            }
            new_pos.push(self.pos[old_index]);
            new_prev_pos.push(self.prev_pos[old_index]);
            new_vel.push(self.vel[old_index]);
            new_mass.push(self.mass[old_index]);
            new_material.push(self.material[old_index]);
            new_activity_state.push(self.activity_state[old_index]);
            new_sleep_candidate_frames.push(self.sleep_candidate_frames[old_index]);
            new_active_hold_frames.push(self.active_hold_frames[old_index]);
            new_sleep_lock_frames.push(self.sleep_lock_frames[old_index]);
            new_pending_wake.push(self.pending_wake[old_index]);
        }

        self.pos = new_pos;
        self.prev_pos = new_prev_pos;
        self.vel = new_vel;
        self.mass = new_mass;
        self.material = new_material;
        self.activity_state = new_activity_state;
        self.sleep_candidate_frames = new_sleep_candidate_frames;
        self.active_hold_frames = new_active_hold_frames;
        self.sleep_lock_frames = new_sleep_lock_frames;
        self.pending_wake = new_pending_wake;
        object_world.apply_particle_remap(&old_to_new, self.masses());
        self.resize_work_buffers();
    }

    fn resize_work_buffers(&mut self) {
        let count = self.particle_count();
        self.prev_pos.resize(count, Vec2::ZERO);
        self.vel.resize(count, Vec2::ZERO);
        self.mass.resize(count, default_particle_mass());
        self.material.resize(count, ParticleMaterial::WaterLiquid);
        self.density.resize(count, 0.0);
        self.lambda.resize(count, 0.0);
        self.delta_pos.resize(count, Vec2::ZERO);
        self.activity_state
            .resize(count, ParticleActivityState::Active);
        self.sleep_candidate_frames.resize(count, 0);
        self.active_hold_frames.resize(count, 0);
        self.sleep_lock_frames.resize(count, 0);
        self.pending_wake.resize(count, false);
        self.neighbor_cache.resize_with(count, Vec::new);
        self.neighbor_cache_anchor_pos.resize(count, Vec2::ZERO);
        self.neighbor_cache_anchor_valid = false;
        self.viscosity_work.resize(count, Vec2::ZERO);
    }

    fn terrain_fracture_seed_velocity(&self, fracture_cell: IVec2) -> Vec2 {
        self.pending_terrain_fracture_seeds
            .get(&fracture_cell)
            .copied()
            .map(TerrainFractureSeed::velocity)
            .unwrap_or(Vec2::ZERO)
    }

    fn collect_detached_terrain_components(
        &self,
        terrain: &TerrainWorld,
        removed_cells: &HashSet<IVec2>,
    ) -> Vec<Vec<IVec2>> {
        let mut checked = HashSet::new();
        let mut detached_components = Vec::<Vec<IVec2>>::new();
        for &removed_cell in removed_cells {
            for offset in FOUR_NEIGHBOR_OFFSETS {
                let start = removed_cell + offset;
                if checked.contains(&start)
                    || !matches!(
                        terrain.get_loaded_cell_or_empty(start),
                        TerrainCell::Solid { .. }
                    )
                {
                    continue;
                }
                let fill = flood_fill_4_limited(start, DETACH_FLOOD_FILL_MAX_CELLS, |cell| {
                    matches!(
                        terrain.get_loaded_cell_or_empty(cell),
                        TerrainCell::Solid { .. }
                    )
                });
                checked.extend(fill.cells.iter().copied());
                if fill.reached_limit {
                    continue;
                }
                let mut cells = fill.cells.into_iter().collect::<Vec<_>>();
                cells.sort_by_key(|cell| (cell.y, cell.x));
                detached_components.push(cells);
            }
        }
        detached_components
    }

    fn evaluate_terrain_persistent_load(
        &mut self,
        terrain: &TerrainWorld,
        terrain_load_metric: &HashMap<IVec2, f32>,
    ) {
        if TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS == 0 {
            return;
        }
        if self.terrain_load_substep_counter % TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS as u64 != 0 {
            return;
        }
        let dt_sample = (FIXED_DT / SUBSTEPS as f32) * TERRAIN_LOAD_SAMPLE_INTERVAL_SUBSTEPS as f32;
        let mut seen_cells = HashSet::new();

        for (&cell, &metric) in terrain_load_metric {
            if !matches!(
                terrain.get_loaded_cell_or_empty(cell),
                TerrainCell::Solid { .. }
            ) {
                self.terrain_persistent_load.remove(&cell);
                continue;
            }
            seen_cells.insert(cell);
            let mut should_wake = false;
            let mut should_fracture = false;
            {
                let state = self.terrain_persistent_load.entry(cell).or_default();
                state.strain_metric = metric;
                if metric >= TERRAIN_LOAD_STRAIN_THRESHOLD {
                    state.sustained_seconds += dt_sample;
                    should_wake = true;
                    should_fracture =
                        state.sustained_seconds >= TERRAIN_LOAD_BREAK_DURATION_SECONDS;
                } else {
                    state.sustained_seconds = (state.sustained_seconds - dt_sample).max(0.0);
                }
            }
            if should_wake {
                self.request_wake_near(cell_to_world_center(cell), WAKE_RADIUS);
            }
            if should_fracture {
                self.pending_terrain_fractures.insert(cell);
            }
        }

        let cells: Vec<_> = self.terrain_persistent_load.keys().copied().collect();
        for cell in cells {
            if seen_cells.contains(&cell) {
                continue;
            }
            if !matches!(
                terrain.get_loaded_cell_or_empty(cell),
                TerrainCell::Solid { .. }
            ) {
                self.terrain_persistent_load.remove(&cell);
                continue;
            }
            if let Some(state) = self.terrain_persistent_load.get_mut(&cell) {
                state.strain_metric *= TERRAIN_LOAD_DECAY_PER_SAMPLE;
                state.sustained_seconds = (state.sustained_seconds - dt_sample).max(0.0);
                if state.sustained_seconds <= 0.0 && state.strain_metric <= 1e-4 {
                    self.terrain_persistent_load.remove(&cell);
                }
            }
        }
    }

    fn build_terrain_detach_spawn_plan(
        &self,
        component_cells: &[IVec2],
        terrain: &TerrainWorld,
        fracture_cell_set: &HashSet<IVec2>,
    ) -> TerrainDetachSpawnPlan {
        let mut plan_cells = Vec::with_capacity(component_cells.len());
        let mut cell_materials = Vec::with_capacity(component_cells.len());
        for &cell in component_cells {
            let TerrainCell::Solid { material, .. } = terrain.get_loaded_cell_or_empty(cell) else {
                continue;
            };
            let Some(particle_material) = terrain_solid_particle(material) else {
                continue;
            };
            plan_cells.push(cell);
            cell_materials.push((cell, particle_material));
        }

        if plan_cells.is_empty() {
            return TerrainDetachSpawnPlan::default();
        }

        let mut influencing_fracture_cells = HashSet::new();
        for &cell in &plan_cells {
            for offset in FOUR_NEIGHBOR_OFFSETS {
                let fracture_cell = cell + offset;
                if fracture_cell_set.contains(&fracture_cell) {
                    influencing_fracture_cells.insert(fracture_cell);
                }
            }
        }

        let mut seed_weight_sum = 0.0f32;
        let mut seed_velocity_weighted_sum = Vec2::ZERO;
        let mut seed_contact_weighted_sum = Vec2::ZERO;
        for fracture_cell in &influencing_fracture_cells {
            let Some(seed) = self
                .pending_terrain_fracture_seeds
                .get(fracture_cell)
                .copied()
            else {
                continue;
            };
            seed_weight_sum += seed.impulse_sum.max(1e-6);
            seed_velocity_weighted_sum += seed.velocity_weighted_sum;
            seed_contact_weighted_sum += seed.contact_weighted_sum;
        }

        let linear_velocity = if seed_weight_sum > 1e-6 {
            seed_velocity_weighted_sum / seed_weight_sum
        } else {
            Vec2::ZERO
        };
        let angular_origin = if seed_weight_sum > 1e-6 {
            seed_contact_weighted_sum / seed_weight_sum
        } else {
            let mut weighted_com = Vec2::ZERO;
            let mut mass_sum = 0.0f32;
            for &(cell, material) in &cell_materials {
                let mass = particle_properties(material).mass;
                weighted_com += cell_to_world_center(cell) * mass;
                mass_sum += mass;
            }
            if mass_sum > 1e-6 {
                weighted_com / mass_sum
            } else {
                cell_to_world_center(plan_cells[0])
            }
        };

        let mut angular_velocity = 0.0f32;
        if seed_weight_sum > 1e-6 {
            let mut numerator = 0.0f32;
            let mut denominator = 0.0f32;
            for fracture_cell in &influencing_fracture_cells {
                let Some(seed) = self
                    .pending_terrain_fracture_seeds
                    .get(fracture_cell)
                    .copied()
                else {
                    continue;
                };
                let weight = seed.impulse_sum.max(1e-6);
                let r = seed.contact_pos() - angular_origin;
                let rel_v = seed.velocity() - linear_velocity;
                numerator += weight * (r.x * rel_v.y - r.y * rel_v.x);
                denominator += weight * r.length_squared();
            }
            if denominator > 1e-6 {
                angular_velocity = numerator / denominator;
            }
        }

        TerrainDetachSpawnPlan {
            cells: plan_cells,
            cell_materials,
            linear_velocity,
            angular_velocity,
            angular_origin,
        }
    }

    fn spawn_terrain_detach_component(&mut self, plan: &TerrainDetachSpawnPlan) -> Vec<usize> {
        let mut indices = Vec::new();
        for &(cell, material) in &plan.cell_materials {
            let mut cell_indices = Vec::new();
            self.append_material_particles_in_cell(cell, material, Vec2::ZERO, &mut cell_indices);
            indices.extend(cell_indices);
        }

        for &index in &indices {
            let position = self.pos[index];
            let r = position - plan.angular_origin;
            let angular_velocity_vec = Vec2::new(-r.y, r.x) * plan.angular_velocity;
            let velocity = plan.linear_velocity + angular_velocity_vec;
            self.vel[index] = velocity;
            self.prev_pos[index] = self.pos[index] - velocity * FIXED_DT;
        }
        indices
    }
}

#[derive(Debug, Default)]
struct TerrainDetachSpawnPlan {
    cells: Vec<IVec2>,
    cell_materials: Vec<(IVec2, ParticleMaterial)>,
    linear_velocity: Vec2,
    angular_velocity: f32,
    angular_origin: Vec2,
}

#[cfg(test)]
mod tests;
