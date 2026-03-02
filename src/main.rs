use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::log::{BoxedLayer, LogPlugin};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_inspector_egui::bevy_egui::EguiPlugin;
use particles::camera_controller::CameraControllerPlugin;
use particles::interface::InterfacePlugin;
use particles::overlay::OverlayPlugin;
use particles::physics::PhysicsPlugin;
use particles::physics::gpu_mpm::GpuMpmPlugin;
use particles::physics::gpu_mpm::gpu_resources::MpmGpuControl;
use particles::physics::gpu_mpm::sync::apply_gpu_readback;
use particles::physics::scenario::{default_scenario_spec_by_name, evaluate_scenario_state};
use particles::physics::solver::mpm_water::is_mpm_managed_particle;
use particles::physics::state::{ReplayLoadScenarioRequest, ReplayState, SimulationState};
use particles::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use particles::physics::world::particle::{ParticleMaterial, ParticleWorld};
use particles::physics::world::terrain::{TerrainCell, TerrainWorld, world_to_cell};
use particles::render::{TerrainDotGpuPlugin, TerrainRenderDiagnostics, WaterDotGpuPlugin};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

fn tracy_layer(_app: &mut App) -> Option<BoxedLayer> {
    #[cfg(feature = "tracy")]
    {
        Some(Box::new(tracing_tracy::TracyLayer::default()))
    }
    #[cfg(not(feature = "tracy"))]
    {
        None
    }
}

#[derive(Resource, Debug)]
struct MpmAutoVerifyState {
    enabled: bool,
    scenario_name: String,
    output_path: String,
    sample_material: Option<ParticleMaterial>,
    scenario_requested: bool,
    captured_start: bool,
    wait_frames: u32,
    run_frames: u32,
    run_steps: u32,
    max_wait_frames: u32,
    min_avg_fps: f32,
    max_penetration_ratio: f32,
    min_mean_drop: f32,
    start_mean_y: f32,
    elapsed_secs: f32,
    phase: u8,
    end_sample_wait_frames: u32,
    end_sample_wait_min: u32,
    end_sample_wait_max: u32,
    start_step: usize,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct AutoVerifyConfigFile {
    #[serde(default)]
    mpm: MpmAutoVerifyConfig,
    #[serde(default)]
    screenshot: ScreenshotAutoVerifyConfig,
    gpu_readback_interval_frames: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct MpmAutoVerifyConfig {
    enabled: Option<bool>,
    scenario_name: Option<String>,
    output_path: Option<String>,
    sample_material: Option<String>,
    run_frames: Option<u32>,
    run_steps: Option<u32>,
    min_avg_fps: Option<f32>,
    max_penetration_ratio: Option<f32>,
    min_mean_drop: Option<f32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ScreenshotAutoVerifyConfig {
    enabled: Option<bool>,
    scenario_name: Option<String>,
    output_path: Option<String>,
    warmup_frames: Option<u32>,
    max_wait_after_capture_frames: Option<u32>,
    camera_scale: Option<f32>,
    camera_center_x: Option<f32>,
    camera_center_y: Option<f32>,
}

impl MpmAutoVerifyState {
    fn from_env_and_config(config: &MpmAutoVerifyConfig) -> Self {
        let enabled = config
            .enabled
            .unwrap_or_else(|| env_bool("PARTICLES_AUTOVERIFY_MPM"));
        let scenario_name = config
            .scenario_name
            .clone()
            .or_else(|| std::env::var("PARTICLES_AUTOVERIFY_SCENARIO").ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "water_drop".to_string());
        let output_path = config
            .output_path
            .clone()
            .or_else(|| std::env::var("PARTICLES_AUTOVERIFY_MPM_OUT").ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "artifacts/mpm_autoverify.json".to_string());
        let sample_material = config
            .sample_material
            .as_deref()
            .and_then(parse_particle_material_name)
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_SAMPLE_MATERIAL")
                    .ok()
                    .and_then(|raw| parse_particle_material_name(&raw))
            });
        let run_frames = config
            .run_frames
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_MPM_FRAMES")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
            })
            .unwrap_or(240);
        let run_steps = config
            .run_steps
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_MPM_STEPS")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
            })
            .unwrap_or(0);
        let min_avg_fps = config
            .min_avg_fps
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_MIN_FPS")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
            })
            .unwrap_or(20.0);
        let max_penetration_ratio = config
            .max_penetration_ratio
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_MAX_PENETRATION")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
            })
            .unwrap_or(0.10);
        let min_mean_drop = config
            .min_mean_drop
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_MIN_DROP")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
            })
            .unwrap_or(0.05);
        Self {
            enabled,
            scenario_name,
            output_path,
            sample_material,
            scenario_requested: false,
            captured_start: false,
            wait_frames: 0,
            run_frames,
            run_steps,
            max_wait_frames: 900,
            min_avg_fps,
            max_penetration_ratio,
            min_mean_drop,
            start_mean_y: 0.0,
            elapsed_secs: 0.0,
            phase: 0,
            end_sample_wait_frames: 0,
            end_sample_wait_min: 24,
            end_sample_wait_max: 120,
            start_step: 0,
        }
    }
}

#[derive(Resource, Debug)]
struct ScreenshotVerifyState {
    enabled: bool,
    scenario_name: String,
    skip_scenario_load: bool,
    output_path: String,
    warmup_frames: u32,
    max_wait_after_capture_frames: u32,
    scenario_requested: bool,
    frame_counter: u32,
    capture_requested: bool,
    wait_after_capture_frames: u32,
    camera_scale: Option<f32>,
    camera_center: Option<Vec2>,
    camera_scale_applied: bool,
}

impl ScreenshotVerifyState {
    fn from_env_and_config(config: &ScreenshotAutoVerifyConfig) -> Self {
        let enabled = config
            .enabled
            .unwrap_or_else(|| env_bool("PARTICLES_AUTOVERIFY_SCREENSHOT"));
        let scenario_name = config
            .scenario_name
            .clone()
            .or_else(|| std::env::var("PARTICLES_AUTOVERIFY_SCENARIO").ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "water_drop".to_string());
        let skip_scenario_load = scenario_name.eq_ignore_ascii_case("default_world")
            || scenario_name.eq_ignore_ascii_case("none");
        let output_path = config
            .output_path
            .clone()
            .or_else(|| std::env::var("PARTICLES_AUTOVERIFY_SCREENSHOT_OUT").ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "artifacts/water_drop_gpu_render.png".to_string());
        let warmup_frames = config
            .warmup_frames
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_SCREENSHOT_WARMUP_FRAMES")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
            })
            .unwrap_or(180);
        let max_wait_after_capture_frames = config
            .max_wait_after_capture_frames
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_SCREENSHOT_WAIT_FRAMES")
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
            })
            .unwrap_or(180);
        let camera_scale = config
            .camera_scale
            .or_else(|| {
                std::env::var("PARTICLES_AUTOVERIFY_SCREENSHOT_CAMERA_SCALE")
                    .ok()
                    .and_then(|s| s.parse::<f32>().ok())
            })
            .filter(|scale| scale.is_finite() && *scale > 0.0);
        let camera_center_x = config.camera_center_x.or_else(|| {
            std::env::var("PARTICLES_AUTOVERIFY_SCREENSHOT_CAMERA_CENTER_X")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
        });
        let camera_center_y = config.camera_center_y.or_else(|| {
            std::env::var("PARTICLES_AUTOVERIFY_SCREENSHOT_CAMERA_CENTER_Y")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
        });
        let camera_center = match (camera_center_x, camera_center_y) {
            (Some(x), Some(y)) if x.is_finite() && y.is_finite() => Some(Vec2::new(x, y)),
            _ => None,
        };
        Self {
            enabled,
            scenario_name,
            skip_scenario_load,
            output_path,
            warmup_frames,
            max_wait_after_capture_frames,
            scenario_requested: false,
            frame_counter: 0,
            capture_requested: false,
            wait_after_capture_frames: 0,
            camera_scale,
            camera_center,
            camera_scale_applied: false,
        }
    }
}

#[derive(Serialize)]
struct MpmAutoVerifyReport {
    passed: bool,
    note: String,
    scenario: String,
    sampled_particles: usize,
    run_frames: u32,
    run_frames_target: u32,
    run_steps: u32,
    run_steps_target: u32,
    avg_fps: f32,
    start_mean_y: f32,
    end_mean_y: f32,
    mean_drop: f32,
    tracked_min_x: f32,
    tracked_max_x: f32,
    tracked_min_y: f32,
    tracked_max_y: f32,
    terrain_penetration_ratio: f32,
    max_speed_mps: f32,
    max_speed_ok: bool,
    water_surface_height_ok: bool,
    water_surface_assertion_active: bool,
    max_speed_expected: String,
    max_speed_actual: String,
    water_surface_expected: String,
    water_surface_actual: String,
    failed_assertions: Vec<String>,
}

fn env_bool(key: &str) -> bool {
    std::env::var(key)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn parse_particle_material_name(raw: &str) -> Option<ParticleMaterial> {
    let normalized = raw.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "water" | "waterliquid" | "water_liquid" => Some(ParticleMaterial::WaterLiquid),
        "stonesolid" | "stone_solid" => Some(ParticleMaterial::StoneSolid),
        "stonegranular" | "stone_granular" => Some(ParticleMaterial::StoneGranular),
        "soilsolid" | "soil_solid" => Some(ParticleMaterial::SoilSolid),
        "soilgranular" | "soil_granular" | "soil" => Some(ParticleMaterial::SoilGranular),
        "sandsolid" | "sand_solid" => Some(ParticleMaterial::SandSolid),
        "sandgranular" | "sand_granular" | "sand" => Some(ParticleMaterial::SandGranular),
        _ => None,
    }
}

fn collect_tracked_positions(
    particles: &ParticleWorld,
    sample_material: Option<ParticleMaterial>,
) -> Vec<Vec2> {
    if let Some(material) = sample_material {
        return particles
            .materials()
            .iter()
            .enumerate()
            .filter_map(|(i, &mat)| (mat == material).then_some(particles.positions()[i]))
            .collect();
    }
    let water_positions: Vec<Vec2> = particles
        .materials()
        .iter()
        .enumerate()
        .filter_map(|(i, &mat)| (mat == ParticleMaterial::WaterLiquid).then_some(particles.pos[i]))
        .collect();
    if !water_positions.is_empty() {
        return water_positions;
    }
    particles
        .materials()
        .iter()
        .enumerate()
        .filter_map(|(i, &mat)| is_mpm_managed_particle(mat).then_some(particles.pos[i]))
        .collect()
}

fn parse_autoverify_config_path() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--autoverify-config" {
            return args.next();
        }
    }
    None
}

fn load_autoverify_config(path: Option<&str>) -> AutoVerifyConfigFile {
    let Some(path) = path else {
        return AutoVerifyConfigFile::default();
    };
    let Ok(text) = fs::read_to_string(path) else {
        eprintln!("warning: failed to read autoverify config: {path}");
        return AutoVerifyConfigFile::default();
    };
    match serde_json::from_str::<AutoVerifyConfigFile>(&text) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("warning: failed to parse autoverify config {path}: {error}");
            AutoVerifyConfigFile::default()
        }
    }
}

fn write_report<T: Serialize>(path: &str, report: &T) {
    if let Some(parent) = Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string_pretty(report) {
        let _ = fs::write(path, text);
    }
}

fn run_mpm_autoverify(
    mut state: ResMut<MpmAutoVerifyState>,
    time: Res<Time>,
    replay_state: Res<ReplayState>,
    particle_world: Res<ParticleWorld>,
    terrain_world: Res<TerrainWorld>,
    object_world: Res<ObjectWorld>,
    object_field: Res<ObjectPhysicsField>,
    mut sim_state: ResMut<SimulationState>,
    mut gpu_control: ResMut<MpmGpuControl>,
    mut scenario_writer: MessageWriter<ReplayLoadScenarioRequest>,
    mut exit_writer: MessageWriter<bevy::app::AppExit>,
) {
    if !state.enabled {
        return;
    }

    if !state.scenario_requested {
        scenario_writer.write(ReplayLoadScenarioRequest {
            scenario_name: state.scenario_name.clone(),
        });
        state.scenario_requested = true;
        sim_state.running = false;
        sim_state.step_once = false;
        gpu_control.readback_enabled = false;
        return;
    }

    let tracked_positions = collect_tracked_positions(&particle_world, state.sample_material);

    if !state.captured_start {
        if tracked_positions.is_empty() {
            state.wait_frames = state.wait_frames.saturating_add(1);
            if state.wait_frames > state.max_wait_frames {
                let report = MpmAutoVerifyReport {
                    passed: false,
                    note: "No tracked particles were available for MPM verification.".to_string(),
                    scenario: state.scenario_name.clone(),
                    sampled_particles: 0,
                    run_frames: 0,
                    run_frames_target: state.run_frames,
                    run_steps: 0,
                    run_steps_target: state.run_steps,
                    avg_fps: 0.0,
                    start_mean_y: 0.0,
                    end_mean_y: 0.0,
                    mean_drop: 0.0,
                    tracked_min_x: 0.0,
                    tracked_max_x: 0.0,
                    tracked_min_y: 0.0,
                    tracked_max_y: 0.0,
                    terrain_penetration_ratio: 0.0,
                    max_speed_mps: 0.0,
                    max_speed_ok: false,
                    water_surface_height_ok: false,
                    water_surface_assertion_active: false,
                    max_speed_expected: String::new(),
                    max_speed_actual: String::new(),
                    water_surface_expected: String::new(),
                    water_surface_actual: String::new(),
                    failed_assertions: Vec::new(),
                };
                write_report(&state.output_path, &report);
                exit_writer.write(bevy::app::AppExit::from_code(4));
            }
            return;
        }
        state.start_mean_y =
            tracked_positions.iter().map(|p| p.y).sum::<f32>() / tracked_positions.len() as f32;
        state.captured_start = true;
        state.wait_frames = 0;
        state.elapsed_secs = 0.0;
        state.phase = 1;
        state.start_step = replay_state.current_step;
        sim_state.running = true;
        gpu_control.readback_enabled = false;
        return;
    }

    if state.phase == 1 {
        state.wait_frames = state.wait_frames.saturating_add(1);
        state.elapsed_secs += time.delta_secs();
        let simulated_steps = replay_state.current_step.saturating_sub(state.start_step) as u32;
        let run_steps_reached = state.run_steps > 0 && simulated_steps >= state.run_steps;
        let run_frames_reached = state.run_steps == 0 && state.wait_frames >= state.run_frames;
        if !run_steps_reached && !run_frames_reached {
            return;
        }
        // Stop simulation and request one final readback snapshot.
        sim_state.running = false;
        sim_state.step_once = true;
        gpu_control.readback_enabled = true;
        gpu_control.readback_interval_frames = 1;
        state.phase = 2;
        state.end_sample_wait_frames = 0;
        return;
    }

    if state.phase == 2 {
        // Drive single-step updates so readback always receives a fresh GPU snapshot.
        sim_state.running = false;
        sim_state.step_once = true;
        state.end_sample_wait_frames = state.end_sample_wait_frames.saturating_add(1);
        let end_mean_y_live =
            tracked_positions.iter().map(|p| p.y).sum::<f32>() / tracked_positions.len() as f32;
        let got_fresh_sample = (end_mean_y_live - state.start_mean_y).abs() > 1.0e-4;
        let waited_min = state.end_sample_wait_frames >= state.end_sample_wait_min;
        if (!waited_min || !got_fresh_sample)
            && state.end_sample_wait_frames < state.end_sample_wait_max
        {
            return;
        }
    }

    if tracked_positions.is_empty() {
        let report = MpmAutoVerifyReport {
            passed: false,
            note: "Tracked particles disappeared during MPM verification.".to_string(),
            scenario: state.scenario_name.clone(),
            sampled_particles: 0,
            run_frames: state.wait_frames,
            run_frames_target: state.run_frames,
            run_steps: replay_state.current_step.saturating_sub(state.start_step) as u32,
            run_steps_target: state.run_steps,
            avg_fps: 0.0,
            start_mean_y: state.start_mean_y,
            end_mean_y: 0.0,
            mean_drop: 0.0,
            tracked_min_x: 0.0,
            tracked_max_x: 0.0,
            tracked_min_y: 0.0,
            tracked_max_y: 0.0,
            terrain_penetration_ratio: 1.0,
            max_speed_mps: 0.0,
            max_speed_ok: false,
            water_surface_height_ok: false,
            water_surface_assertion_active: false,
            max_speed_expected: String::new(),
            max_speed_actual: String::new(),
            water_surface_expected: String::new(),
            water_surface_actual: String::new(),
            failed_assertions: Vec::new(),
        };
        write_report(&state.output_path, &report);
        exit_writer.write(bevy::app::AppExit::from_code(5));
        return;
    }

    let end_mean_y =
        tracked_positions.iter().map(|p| p.y).sum::<f32>() / tracked_positions.len() as f32;
    let tracked_min_x = tracked_positions
        .iter()
        .map(|p| p.x)
        .fold(f32::INFINITY, f32::min);
    let tracked_max_x = tracked_positions
        .iter()
        .map(|p| p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let tracked_min_y = tracked_positions
        .iter()
        .map(|p| p.y)
        .fold(f32::INFINITY, f32::min);
    let tracked_max_y = tracked_positions
        .iter()
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let mean_drop = state.start_mean_y - end_mean_y;
    let penetration_count = tracked_positions
        .iter()
        .filter(|&&pos| {
            if let Some((distance, _)) = terrain_world.sample_signed_distance_and_normal(pos) {
                return distance < -1.0e-3;
            }
            let cell = world_to_cell(pos);
            matches!(
                terrain_world.get_cell_or_generated(cell),
                TerrainCell::Solid { .. }
            )
        })
        .count();
    let penetration_ratio = penetration_count as f32 / tracked_positions.len() as f32;
    let avg_fps = state.wait_frames as f32 / state.elapsed_secs.max(1.0e-5);
    let simulated_steps = replay_state.current_step.saturating_sub(state.start_step) as u32;

    let fps_ok = avg_fps >= state.min_avg_fps;
    let drop_ok = mean_drop >= state.min_mean_drop;
    let terrain_ok = penetration_ratio <= state.max_penetration_ratio;
    let mut max_speed_mps = 0.0_f32;
    let mut max_speed_ok = true;
    let mut water_surface_height_ok = true;
    let mut water_surface_assertion_active = false;
    let mut max_speed_expected = String::new();
    let mut max_speed_actual = String::new();
    let mut water_surface_expected = String::new();
    let mut water_surface_actual = String::new();
    let mut assertions_ok = true;
    let mut failed_assertions = Vec::new();
    if let Some(spec) = default_scenario_spec_by_name(&state.scenario_name) {
        let (metrics, assertions) = evaluate_scenario_state(
            &spec,
            replay_state.current_step,
            replay_state.baseline_particle_count,
            replay_state.baseline_solid_cell_count,
            &terrain_world,
            &particle_world,
            &object_world,
            &object_field,
        );
        max_speed_mps = metrics.max_speed_mps;
        for row in assertions {
            if row.active && !row.ok {
                assertions_ok = false;
                failed_assertions.push(format!(
                    "{}: expected {}, actual {}",
                    row.label, row.expected, row.actual
                ));
            }
            if row.label == "max_speed_mps" && row.active {
                max_speed_ok = row.ok;
                max_speed_expected = row.expected;
                max_speed_actual = row.actual;
            } else if row.label == "water_surface_height_p95" {
                water_surface_assertion_active = row.active;
                water_surface_expected = row.expected;
                water_surface_actual = row.actual;
                if row.active {
                    water_surface_height_ok = row.ok;
                }
            }
        }
    }

    let passed = fps_ok
        && drop_ok
        && terrain_ok
        && max_speed_ok
        && (!water_surface_assertion_active || water_surface_height_ok)
        && assertions_ok;
    let note = if passed {
        "MPM verification passed: falling, terrain interaction, FPS, and assertions are within thresholds."
            .to_string()
    } else {
        format!(
            "MPM verification failed: fps_ok={fps_ok}, drop_ok={drop_ok}, terrain_ok={terrain_ok}, max_speed_ok={max_speed_ok}, water_surface_ok={water_surface_height_ok}, water_surface_active={water_surface_assertion_active}, assertions_ok={assertions_ok}"
        )
    };

    let report = MpmAutoVerifyReport {
        passed,
        note,
        scenario: state.scenario_name.clone(),
        sampled_particles: tracked_positions.len(),
        run_frames: state.wait_frames,
        run_frames_target: state.run_frames,
        run_steps: simulated_steps,
        run_steps_target: state.run_steps,
        avg_fps,
        start_mean_y: state.start_mean_y,
        end_mean_y,
        mean_drop,
        tracked_min_x,
        tracked_max_x,
        tracked_min_y,
        tracked_max_y,
        terrain_penetration_ratio: penetration_ratio,
        max_speed_mps,
        max_speed_ok,
        water_surface_height_ok,
        water_surface_assertion_active,
        max_speed_expected,
        max_speed_actual,
        water_surface_expected,
        water_surface_actual,
        failed_assertions,
    };
    write_report(&state.output_path, &report);
    gpu_control.readback_enabled = false;
    sim_state.step_once = false;
    exit_writer.write(if passed {
        bevy::app::AppExit::Success
    } else {
        bevy::app::AppExit::from_code(6)
    });
}

fn run_screenshot_autoverify(
    mut commands: Commands,
    mut state: ResMut<ScreenshotVerifyState>,
    mut sim_state: ResMut<SimulationState>,
    mut scenario_writer: MessageWriter<ReplayLoadScenarioRequest>,
    mut exit_writer: MessageWriter<bevy::app::AppExit>,
    mut camera_query: Query<(&mut Projection, &mut Transform), With<Camera2d>>,
) {
    if !state.enabled {
        return;
    }

    if !state.scenario_requested {
        if let Some(parent) = Path::new(&state.output_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::remove_file(&state.output_path);
        if !state.skip_scenario_load {
            scenario_writer.write(ReplayLoadScenarioRequest {
                scenario_name: state.scenario_name.clone(),
            });
        }
        sim_state.running = true;
        sim_state.step_once = false;
        state.scenario_requested = true;
        return;
    }

    if !state.capture_requested {
        // Keep simulation advancing even if scenario load reset it to paused.
        sim_state.running = true;
        sim_state.step_once = false;
        if !state.camera_scale_applied {
            if let Ok((mut projection, mut transform)) = camera_query.single_mut() {
                if let Some(scale) = state.camera_scale {
                    if let Projection::Orthographic(ortho) = projection.as_mut() {
                        ortho.scale = scale;
                    }
                }
                if let Some(center) = state.camera_center {
                    transform.translation.x = center.x;
                    transform.translation.y = center.y;
                }
            }
            state.camera_scale_applied = true;
        }
        state.frame_counter = state.frame_counter.saturating_add(1);
        if state.frame_counter < state.warmup_frames {
            return;
        }
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(state.output_path.clone()));
        state.capture_requested = true;
        state.wait_after_capture_frames = 0;
        return;
    }

    state.wait_after_capture_frames = state.wait_after_capture_frames.saturating_add(1);
    sim_state.running = true;
    sim_state.step_once = false;
    let ready = fs::metadata(&state.output_path)
        .map(|meta| meta.len() > 0)
        .unwrap_or(false);
    if ready {
        state.enabled = false;
        exit_writer.write(bevy::app::AppExit::Success);
        return;
    }

    if state.wait_after_capture_frames > state.max_wait_after_capture_frames {
        state.enabled = false;
        exit_writer.write(bevy::app::AppExit::from_code(7));
    }
}

fn main() {
    let autoverify_config_path = parse_autoverify_config_path();
    let autoverify_config = load_autoverify_config(autoverify_config_path.as_deref());
    let mpm_autoverify = autoverify_config
        .mpm
        .enabled
        .unwrap_or_else(|| env_bool("PARTICLES_AUTOVERIFY_MPM"));
    let readback_interval_frames = autoverify_config
        .gpu_readback_interval_frames
        .or_else(|| {
            std::env::var("PARTICLES_GPU_READBACK_INTERVAL_FRAMES")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|&v| v > 0)
        })
        .unwrap_or(if mpm_autoverify { 1 } else { 60 });

    let default_plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        })
        .set(LogPlugin {
            custom_layer: tracy_layer,
            ..default()
        });
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.1)))
        .init_resource::<TerrainRenderDiagnostics>()
        .insert_resource(SimulationState {
            mpm_enabled: true,
            ..Default::default()
        })
        .insert_resource(MpmGpuControl {
            init_only: false,
            readback_enabled: true,
            readback_interval_frames,
        })
        .insert_resource(MpmAutoVerifyState::from_env_and_config(
            &autoverify_config.mpm,
        ))
        .insert_resource(ScreenshotVerifyState::from_env_and_config(
            &autoverify_config.screenshot,
        ))
        .add_plugins(default_plugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(EguiPlugin::default());
    app.add_plugins((
        PhysicsPlugin,
        GpuMpmPlugin,
        TerrainDotGpuPlugin,
        WaterDotGpuPlugin,
        InterfacePlugin,
        OverlayPlugin,
        CameraControllerPlugin,
    ))
    .add_systems(Update, run_mpm_autoverify.after(apply_gpu_readback))
    .add_systems(Update, run_screenshot_autoverify.after(run_mpm_autoverify))
    .run();
}
