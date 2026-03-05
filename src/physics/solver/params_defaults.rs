use bevy::prelude::Vec2;

use super::params_types::SolverParams;
use crate::physics::material::{DEFAULT_MATERIAL_PARAMS, particle_radius_m, water_kernel_radius_m};
use crate::physics::world::constants::{CELL_SIZE_M, CHUNK_SIZE_I32};

pub const DEFAULT_SOLVER_PARAMS: SolverParams = SolverParams {
    gravity_mps2: Vec2::new(0.0, -9.81),
    fixed_dt: 1.0 / 240.0,
    mpm_block_rate_level_min: 0,
    substeps: 2,
    solver_iters: 6,
    solver_min_iters: 2,
    solver_error_tolerance: 0.01,
    epsilon_lambda: 1e-6,
    terrain_ghost_density_scale: 1.0,
    terrain_ghost_delta_scale: 1.0,
    parallel_particle_threshold: 512,
    particle_contact_push_factor: 0.5,
    detach_flood_fill_max_cells: 128,
    sleep_disp_threshold: 0.001,
    sleep_vel_threshold: 0.01,
    sleep_frames: 30,
    wake_disp_threshold: 0.001,
    wake_radius_m: CELL_SIZE_M * 1.75,
    active_min_frames: 8,
    enable_sleep_wake: false,
    granular_substeps: 1,
    granular_iters: 4,
    xpbd_contact_compliance_n: 1e-6,
    xpbd_contact_compliance_t: 6e-6,
    wake_impulse_threshold: 2.5,
    fracture_wake_lock_frames: 45,
    terrain_load_sample_interval_substeps: 2,
    terrain_load_strain_threshold: 0.28,
    terrain_load_break_duration_seconds: 0.45,
    terrain_load_decay_per_sample: 0.8,
    granular_contact_friction_scale: 2.0,
    granular_granular_contact_friction_boost: 2.5,
    granular_solid_contact_friction_boost: 1.5,
    terrain_contact_friction_scale: 1.75,
    granular_contact_normal_damping: 0.50,
    terrain_contact_normal_damping: 0.65,
    granular_spawn_jitter_ratio: 0.01,
    object_reaction_max_dv_per_substep_mps: 1.5,
    fracture_min_impact_speed_mps: 2.0,
    particle_escape_margin_x_cells: CHUNK_SIZE_I32,
    particle_escape_margin_bottom_cells: CHUNK_SIZE_I32,
    particle_escape_margin_top_cells: CHUNK_SIZE_I32 * 8,
    far_field_edge_inset_m: particle_radius_m(DEFAULT_MATERIAL_PARAMS) * 1.25,
    neighbor_list_skin_m: water_kernel_radius_m(DEFAULT_MATERIAL_PARAMS) * 0.25,
    object_shape_stiffness_alpha: 0.92,
    object_shape_iters: 3,
    object_local_sdf_samples_per_cell: 2,
    object_physics_sdf_cell_size_m: CELL_SIZE_M * 0.5,
    object_broadphase_cell_size_m: CELL_SIZE_M * 4.0,
    object_sdf_max_distance_m: CELL_SIZE_M * 2.5,
    object_sdf_max_splats_per_cell: 4,
    object_sdf_max_contacts_per_query: 4,
};

impl Default for SolverParams {
    fn default() -> Self {
        DEFAULT_SOLVER_PARAMS
    }
}
