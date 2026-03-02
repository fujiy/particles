// CPU → GPU synchronization systems for MLS-MPM.
//
// Runs in main-world (not render-world): prepares upload requests,
// which are then executed in the render-world extract/prepare phase.

use bevy::prelude::*;

use super::buffers::{GpuMpmParams, GpuParticle};
use super::gpu_resources::{
    MpmGpuControl, MpmGpuParamsRequest, MpmGpuRunRequest, MpmGpuStepClock, MpmGpuUploadRequest,
    world_grid_layout,
};
use super::readback::GpuReadbackResult;
use crate::physics::material::{DEFAULT_MATERIAL_PARAMS, particle_radius_m};
use crate::physics::solver::mpm_water::{
    MpmWaterParams, is_mpm_managed_particle,
    rebuild_continuum_from_particle_world, sync_continuum_to_particle_world,
};
use crate::physics::solver::params_types::SolverParams;
use crate::physics::state::{ReplayState, SimulationState};
use crate::physics::world::constants::CELL_SIZE_M;
use crate::physics::world::continuum::ContinuumParticleWorld;
use crate::physics::world::particle::ParticleWorld;
use crate::physics::world::terrain::{TerrainCell, TerrainWorld, world_to_cell};

const SDF_QUERY_RADIUS_CELLS: i32 = 10;
const SDF_INF: f32 = 1.0e9;
const MPM_TARGET_SOUND_SPEED_MPS: f32 = 16.0;
const MPM_TARGET_RHO0: f32 = 1_000.0;
const MPM_BOUNDARY_THRESHOLD_SCALE_DIAMETER: f32 = 0.01;
/// Coulomb friction μ_b for water at terrain boundary [Eq.29, physics.md].
const BOUNDARY_FRICTION_WATER: f32 = 0.3;
/// Coulomb friction μ_b for granular at terrain boundary [Eq.29, physics.md].
const BOUNDARY_FRICTION_GRANULAR: f32 = 0.6;
const SOIL_YOUNGS_MODULUS_PA: f32 = 8.0e2;
const SOIL_POISSON_RATIO: f32 = 0.28;
const SOIL_FRICTION_DEG: f32 = 36.0;
const SOIL_COHESION_PA: f32 = 0.0;
const SOIL_HARDENING: f32 = 0.0;
const SAND_YOUNGS_MODULUS_PA: f32 = 7.0e2;
const SAND_POISSON_RATIO: f32 = 0.25;
const SAND_FRICTION_DEG: f32 = 34.0;
const SAND_COHESION_PA: f32 = 0.0;
const SAND_HARDENING: f32 = 0.0;
const GRANULAR_TENSION_CLAMP_PA: f32 = 0.0;
const COUPLING_NORMAL_STIFFNESS: f32 = 0.55;
const COUPLING_TANGENT_DRAG: f32 = 0.30;
const COUPLING_FRICTION: f32 = 0.45;
const COUPLING_MAX_IMPULSE_RATIO: f32 = 0.50;

/// Ensure GPU upload source (`ContinuumParticleWorld`) is seeded from `ParticleWorld`.
///
/// This prevents "overlay ON but nothing visible" when a map/scenario populates
/// `ParticleWorld` but does not explicitly rebuild continuum data.
pub fn ensure_continuum_seed_from_particle_world(
    control: Res<MpmGpuControl>,
    particle_world: Res<ParticleWorld>,
    solver_params: Res<SolverParams>,
    mut continuum: ResMut<ContinuumParticleWorld>,
) {
    if control.init_only {
        return;
    }

    let mpm_count = particle_world
        .materials()
        .iter()
        .filter(|&&mat| is_mpm_managed_particle(mat))
        .count();
    if mpm_count == 0 {
        if continuum.is_empty() {
            return;
        }
        continuum.clear();
        return;
    }

    let needs_seed = continuum.len() != mpm_count
        || (continuum.is_empty() && particle_world.particle_count() > 0);
    if !needs_seed {
        return;
    }

    let _ = rebuild_continuum_from_particle_world(
        &particle_world,
        &mut continuum,
        &MpmWaterParams {
            dt: solver_params.fixed_dt,
            gravity: solver_params.gravity_mps2,
            ..Default::default()
        },
    );
}

/// System: build upload request from ContinuumParticleWorld.
///
/// Called when particles are modified (spawn/clear). Marks full particle upload.
pub fn prepare_particle_upload(
    control: Res<MpmGpuControl>,
    particle_world: Res<ParticleWorld>,
    continuum: Res<ContinuumParticleWorld>,
    mut upload: ResMut<MpmGpuUploadRequest>,
    mut sim_state: ResMut<SimulationState>,
) {
    // Default: no particle upload this frame unless explicitly requested below.
    upload.upload_particles = false;

    if control.init_only {
        sim_state.gpu_mpm_active = false;
        upload.particles.clear();
        return;
    }
    let mpm_count = particle_world_mpm_count(&particle_world);
    let unmanaged_count = particle_world
        .materials()
        .iter()
        .filter(|&&material| !is_mpm_managed_particle(material))
        .count();
    // Keep GPU MPM on a single-route path: either all dynamic particles are MPM-managed,
    // or we stay on the legacy CPU particle step to avoid partially frozen worlds.
    sim_state.gpu_mpm_active = sim_state.mpm_enabled && mpm_count > 0 && unmanaged_count == 0;
    // Ensure at least one upload after startup/scenario load even when change detection
    // is not observed in this system's frame.
    if !continuum.is_changed() && !upload.particles.is_empty() {
        return;
    }
    let particles: Vec<GpuParticle> = (0..continuum.len())
        .map(|i| {
            GpuParticle::from_cpu(
                continuum.x[i],
                continuum.v[i],
                continuum.m[i],
                continuum.v0[i],
                continuum.f[i],
                continuum.c[i],
                continuum.jp[i],
                continuum.material_id[i],
            )
        })
        .collect();
    upload.particles = particles;
    upload.upload_particles = !upload.particles.is_empty();
}

/// System: build terrain SDF/normal upload request.
///
/// Only re-uploads when TerrainWorld changes.
pub fn prepare_terrain_upload(
    control: Res<MpmGpuControl>,
    sim_state: Res<SimulationState>,
    terrain: Res<TerrainWorld>,
    mut upload: ResMut<MpmGpuUploadRequest>,
) {
    // Default: no terrain upload this frame unless explicitly requested below.
    upload.upload_terrain = false;

    if control.init_only {
        upload.terrain_sdf.clear();
        upload.terrain_normal.clear();
        return;
    }
    // When MLS-MPM stepping is disabled, terrain upload is unnecessary for overlay debug.
    if !sim_state.mpm_enabled {
        return;
    }
    if !terrain.is_changed() {
        return;
    }
    let layout = world_grid_layout();
    let node_count = layout.node_count();
    let mut sdf = vec![SDF_INF; node_count];
    let mut normals = vec![[0.0_f32, 1.0]; node_count];

    let h = CELL_SIZE_M;
    let eps = h * 0.5;

    for ny in 0..layout.dims.y {
        for nx in 0..layout.dims.x {
            let idx = (ny * layout.dims.x + nx) as usize;
            let node = layout.origin + IVec2::new(nx as i32, ny as i32);
            let world_pos = Vec2::new(node.x as f32 * h, node.y as f32 * h);
            if let Some((signed_distance, normal)) =
                terrain.sample_signed_distance_and_normal(world_pos)
            {
                sdf[idx] = signed_distance;
                if normal != Vec2::ZERO {
                    normals[idx] = normal.to_array();
                }
                continue;
            }

            let sdf_val = sample_terrain_sdf_for_gpu(&terrain, world_pos);
            sdf[idx] = sdf_val;

            let dx = sample_terrain_sdf_for_gpu(&terrain, world_pos + Vec2::new(eps, 0.0))
                - sample_terrain_sdf_for_gpu(&terrain, world_pos - Vec2::new(eps, 0.0));
            let dy = sample_terrain_sdf_for_gpu(&terrain, world_pos + Vec2::new(0.0, eps))
                - sample_terrain_sdf_for_gpu(&terrain, world_pos - Vec2::new(0.0, eps));
            let len = (dx * dx + dy * dy).sqrt();
            if len > 1e-6 {
                normals[idx] = [dx / len, dy / len];
            }
        }
    }

    upload.terrain_sdf = sdf;
    upload.terrain_normal = normals;
    upload.upload_terrain = true;
}

/// System: update MpmGpuParamsRequest from SolverParams and simulation state.
pub fn prepare_gpu_params(
    control: Res<MpmGpuControl>,
    solver_params: Res<SolverParams>,
    mut params_req: ResMut<MpmGpuParamsRequest>,
    continuum: Res<ContinuumParticleWorld>,
) {
    if control.init_only {
        return;
    }
    let layout = world_grid_layout();
    let h = CELL_SIZE_M;
    let soil = drucker_prager_params(
        SOIL_YOUNGS_MODULUS_PA,
        SOIL_POISSON_RATIO,
        SOIL_FRICTION_DEG,
        SOIL_COHESION_PA,
        SOIL_HARDENING,
    );
    let sand = drucker_prager_params(
        SAND_YOUNGS_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_FRICTION_DEG,
        SAND_COHESION_PA,
        SAND_HARDENING,
    );
    // Keep GPU params in parity with CPU MPM setup in solver/step.rs.
    let boundary_threshold_m =
        particle_radius_m(DEFAULT_MATERIAL_PARAMS) * MPM_BOUNDARY_THRESHOLD_SCALE_DIAMETER;

    params_req.params = GpuMpmParams {
        dt: solver_params.fixed_dt,
        gx: 0.0,
        gy: -9.81,
        rho0: MPM_TARGET_RHO0,
        bulk_modulus: MPM_TARGET_RHO0 * MPM_TARGET_SOUND_SPEED_MPS * MPM_TARGET_SOUND_SPEED_MPS,
        h,
        grid_origin_x: layout.origin.x,
        grid_origin_y: layout.origin.y,
        grid_width: layout.dims.x,
        grid_height: layout.dims.y,
        particle_count: continuum.len() as u32,
        j_min: 0.6,
        j_max: 1.4,
        c_max_norm: 80.0,
        sdf_velocity_threshold_m: boundary_threshold_m,
        boundary_friction_water: BOUNDARY_FRICTION_WATER,
        boundary_friction_granular: BOUNDARY_FRICTION_GRANULAR,
        _pad_friction: 0,
        dp_lambda_soil: soil.lambda,
        dp_mu_soil: soil.mu,
        dp_alpha_soil: soil.alpha,
        dp_k_soil: soil.k,
        dp_hardening_soil: soil.hardening,
        dp_lambda_sand: sand.lambda,
        dp_mu_sand: sand.mu,
        dp_alpha_sand: sand.alpha,
        dp_k_sand: sand.k,
        dp_hardening_sand: sand.hardening,
        granular_tensile_clamp: GRANULAR_TENSION_CLAMP_PA,
        coupling_normal_stiffness: COUPLING_NORMAL_STIFFNESS,
        coupling_tangent_drag: COUPLING_TANGENT_DRAG,
        coupling_friction: COUPLING_FRICTION,
        coupling_max_impulse_ratio: COUPLING_MAX_IMPULSE_RATIO,
        alpha_apic_water: 0.95,
        alpha_apic_granular: 0.78,
        _pad: [0; 1],
    };
}

/// System: update GPU compute run state from SimulationState.
pub fn prepare_gpu_run_state(
    control: Res<MpmGpuControl>,
    sim_state: Res<SimulationState>,
    mut replay_state: ResMut<ReplayState>,
    solver_params: Res<SolverParams>,
    time: Res<Time>,
    mut step_clock: ResMut<MpmGpuStepClock>,
    mut run_req: ResMut<MpmGpuRunRequest>,
) {
    if control.init_only {
        run_req.enabled = false;
        run_req.substeps = 0;
        step_clock.accumulator_secs = 0.0;
        return;
    }
    let active = sim_state.mpm_enabled && sim_state.gpu_mpm_active;
    if !active {
        run_req.enabled = false;
        run_req.substeps = 0;
        step_clock.accumulator_secs = 0.0;
        return;
    }

    // Single-step command always executes exactly one substep.
    if sim_state.step_once {
        run_req.enabled = true;
        run_req.substeps = 1;
        step_clock.accumulator_secs = 0.0;
        if replay_state.enabled {
            replay_state.current_step = replay_state.current_step.saturating_add(1);
        }
        return;
    }

    if !sim_state.running {
        run_req.enabled = false;
        run_req.substeps = 0;
        return;
    }

    let fixed_dt = solver_params.fixed_dt.max(1.0e-5);
    // Avoid runaway catch-up (spiral of death). If we can't keep up, simulation slows down.
    let max_catchup = fixed_dt * step_clock.max_substeps_per_frame as f32;
    step_clock.accumulator_secs =
        (step_clock.accumulator_secs + time.delta_secs()).min(max_catchup);

    let mut substeps = (step_clock.accumulator_secs / fixed_dt).floor() as u32;
    if substeps > step_clock.max_substeps_per_frame {
        substeps = step_clock.max_substeps_per_frame;
    }
    if substeps == 0 {
        run_req.enabled = false;
        run_req.substeps = 0;
        return;
    }

    step_clock.accumulator_secs =
        (step_clock.accumulator_secs - fixed_dt * substeps as f32).max(0.0);
    run_req.enabled = true;
    run_req.substeps = substeps;
    if replay_state.enabled {
        replay_state.current_step = replay_state.current_step.saturating_add(substeps as usize);
    }
}

/// Diagnostics counter for GPU readback frames.
static GPU_READBACK_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// System: apply GPU readback results to ContinuumParticleWorld.
///
/// Runs in main-world Update (after rendering). Consumes the latest readback
/// from the render world and writes x, v, F, C back to CPU state so the rest
/// of the simulation pipeline sees up-to-date positions.
pub fn apply_gpu_readback(
    control: Res<MpmGpuControl>,
    readback_result: Res<GpuReadbackResult>,
    mut continuum: ResMut<ContinuumParticleWorld>,
    mut particle_world: ResMut<ParticleWorld>,
) {
    if !control.readback_enabled {
        return;
    }
    let Some(particles) = readback_result.take() else {
        return;
    };
    let frame = GPU_READBACK_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let n = particles.len().min(continuum.len());
    {
        // Use bypass_change_detection so this write does NOT trigger prepare_particle_upload.
        // Without this, the CPU would re-upload stale data every frame, overwriting GPU results.
        let continuum = continuum.bypass_change_detection();
        for i in 0..n {
            let p = &particles[i];
            continuum.x[i] = Vec2::from_array(p.x);
            continuum.v[i] = Vec2::from_array(p.v);
            continuum.f[i] = Mat2::from_cols(Vec2::new(p.f[0], p.f[1]), Vec2::new(p.f[2], p.f[3]));
            continuum.c[i] = Mat2::from_cols(Vec2::new(p.c[0], p.c[1]), Vec2::new(p.c[2], p.c[3]));
            continuum.jp[i] = p.jp.max(1e-6);
        }
    }
    // Reflect readback positions to ParticleWorld so particle overlay can render GPU data.
    let _ = sync_continuum_to_particle_world(&mut particle_world, &continuum);
    // Periodic NaN / divergence check (every 60 readback frames).
    if frame % 60 == 0 {
        let mut nan_count = 0u32;
        let mut max_speed: f32 = 0.0;
        for i in 0..n {
            let p = &particles[i];
            if p.x[0].is_nan() || p.x[1].is_nan() || p.v[0].is_nan() || p.v[1].is_nan() {
                nan_count += 1;
            }
            let speed_sq = p.v[0] * p.v[0] + p.v[1] * p.v[1];
            if speed_sq > max_speed {
                max_speed = speed_sq;
            }
        }
        let max_speed = max_speed.sqrt();
        if nan_count > 0 {
            bevy::log::error!("[gpu_mpm] frame={frame} NaN detected in {nan_count}/{n} particles");
        } else {
            bevy::log::info!("[gpu_mpm] frame={frame} n={n} max_speed={max_speed:.2} m/s — OK");
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain SDF helper (mirrors terrain_boundary.rs logic, CPU-side, no cache)
// ---------------------------------------------------------------------------

fn sample_terrain_sdf_for_gpu(terrain: &TerrainWorld, world_pos: Vec2) -> f32 {
    let center_cell = world_to_cell(world_pos);
    let inside = matches!(
        terrain.get_cell_or_generated(center_cell),
        TerrainCell::Solid { .. }
    );
    let h = CELL_SIZE_M;
    let mut best = SDF_INF;
    for dy in -SDF_QUERY_RADIUS_CELLS..=SDF_QUERY_RADIUS_CELLS {
        for dx in -SDF_QUERY_RADIUS_CELLS..=SDF_QUERY_RADIUS_CELLS {
            let cell = center_cell + IVec2::new(dx, dy);
            let cell_is_solid = matches!(
                terrain.get_cell_or_generated(cell),
                TerrainCell::Solid { .. }
            );
            if cell_is_solid == inside {
                continue;
            }
            let min = cell.as_vec2() * h;
            let max = min + Vec2::splat(h);
            let closest = world_pos.clamp(min, max);
            best = best.min(world_pos.distance(closest));
        }
    }
    if best >= SDF_INF {
        best = SDF_QUERY_RADIUS_CELLS as f32 * h;
    }
    if inside { -best } else { best }
}

fn particle_world_mpm_count(particles: &ParticleWorld) -> usize {
    particles
        .materials()
        .iter()
        .filter(|&&material| is_mpm_managed_particle(material))
        .count()
}

#[derive(Clone, Copy, Debug)]
struct GpuDruckerPragerParams {
    lambda: f32,
    mu: f32,
    alpha: f32,
    k: f32,
    hardening: f32,
}

fn drucker_prager_params(
    youngs_modulus: f32,
    poisson_ratio: f32,
    friction_deg: f32,
    cohesion_pa: f32,
    hardening: f32,
) -> GpuDruckerPragerParams {
    let e = youngs_modulus.max(1.0);
    let nu = poisson_ratio.clamp(0.0, 0.45);
    let mu = e / (2.0 * (1.0 + nu).max(1e-6));
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)).max(1e-6);

    let phi = friction_deg.to_radians().clamp(0.0, 1.3);
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    let denom = (3.0 - sin_phi).max(1e-4);
    let alpha = (2.0 * sin_phi) / (3.0_f32.sqrt() * denom);
    let k = (6.0 * cohesion_pa.max(0.0) * cos_phi) / (3.0_f32.sqrt() * denom);

    GpuDruckerPragerParams {
        lambda,
        mu,
        alpha,
        k,
        hardening: hardening.max(0.0),
    }
}
