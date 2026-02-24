use bevy::prelude::{Resource, Vec2};

#[derive(Resource, Clone, Copy, Debug)]
pub struct SolverParams {
    pub gravity_mps2: Vec2,
    pub fixed_dt: f32,
    // Runtime clamps to at least 1 to avoid zero substep execution.
    pub substeps: usize,
    pub solver_iters: usize,
    pub solver_min_iters: usize,
    pub solver_error_tolerance: f32,
    pub epsilon_lambda: f32,
    pub terrain_ghost_density_scale: f32,
    pub terrain_ghost_delta_scale: f32,
    // Runtime clamps to at least 1 before parallel loop partitioning.
    pub parallel_particle_threshold: usize,
    pub particle_contact_push_factor: f32,
    pub detach_flood_fill_max_cells: usize,
    pub sleep_disp_threshold: f32,
    pub sleep_vel_threshold: f32,
    pub sleep_frames: u16,
    pub wake_disp_threshold: f32,
    pub wake_radius_m: f32,
    pub active_min_frames: u16,
    pub enable_sleep_wake: bool,
    // Runtime clamps to at least 1 to keep granular dt finite.
    pub granular_substeps: usize,
    pub granular_iters: usize,
    pub xpbd_contact_compliance_n: f32,
    pub xpbd_contact_compliance_t: f32,
    pub wake_impulse_threshold: f32,
    pub fracture_wake_lock_frames: u16,
    pub terrain_load_sample_interval_substeps: u32,
    pub terrain_load_strain_threshold: f32,
    pub terrain_load_break_duration_seconds: f32,
    pub terrain_load_decay_per_sample: f32,
    pub neighbor_grid_max_axis_cells: i64,
    pub neighbor_grid_max_dense_cells: i64,
    pub granular_contact_friction_scale: f32,
    pub granular_granular_contact_friction_boost: f32,
    pub granular_solid_contact_friction_boost: f32,
    pub terrain_contact_friction_scale: f32,
    pub granular_contact_normal_damping: f32,
    pub terrain_contact_normal_damping: f32,
    pub granular_spawn_jitter_ratio: f32,
    pub object_reaction_max_dv_per_substep_mps: f32,
    pub fracture_min_impact_speed_mps: f32,
    pub particle_escape_margin_x_cells: i32,
    pub particle_escape_margin_bottom_cells: i32,
    pub particle_escape_margin_top_cells: i32,
    pub far_field_edge_inset_m: f32,
    pub neighbor_list_skin_m: f32,
    // Runtime clamps to at least 1 cell.
    pub sub_block_size_cells: i32,
    // Runtime clamps to at least 1 frame.
    pub sub_block_promote_frames: u16,
    // Runtime clamps to at least 1 frame.
    pub sub_block_demote_frames: u16,
    pub sub_block_min_active_frames: u16,
    // Runtime clamps to at least 0.
    pub sub_block_promote_neighbor_radius: i32,
    pub sub_block_debt_promote_threshold: f32,
    pub sub_block_mass_flux_promote_threshold: f32,
    pub sub_block_penetration_promote_threshold_m: f32,
    // Runtime clamps to at least 1 frame.
    pub sub_block_contact_promote_frames: u16,
    // Runtime clamps by `RATE_LEVEL_MAX_SUPPORTED` (current divisor storage limit).
    pub sub_block_max_level: u8,
    // Runtime clamps to at least 1e-6 to keep intrusion threshold finite.
    pub sub_block_level0_max_norm_disp: f32,
    // Runtime clamps to at least 0.
    pub sub_block_near_field_extra_chunks: i32,

    pub object_shape_stiffness_alpha: f32,
    pub object_shape_iters: usize,
    pub object_local_sdf_samples_per_cell: i32,
    pub object_physics_sdf_cell_size_m: f32,
    pub object_broadphase_cell_size_m: f32,
    pub object_sdf_max_distance_m: f32,
    pub object_sdf_max_splats_per_cell: usize,
    pub object_sdf_max_contacts_per_query: usize,
}
