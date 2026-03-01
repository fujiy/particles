use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::log::{BoxedLayer, LogPlugin};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_inspector_egui::bevy_egui::EguiPlugin;
use serde::Serialize;
use std::fs;
use std::path::Path;
use particles::camera_controller::CameraControllerPlugin;
use particles::interface::InterfacePlugin;
use particles::overlay::OverlayPlugin;
use particles::physics::PhysicsPlugin;
use particles::physics::gpu_mpm::GpuMpmPlugin;
use particles::physics::gpu_mpm::sync::apply_gpu_readback;
use particles::physics::gpu_mpm::gpu_resources::MpmGpuControl;
use particles::physics::scenario::{default_scenario_spec_by_name, evaluate_scenario_state};
use particles::physics::state::{ReplayLoadScenarioRequest, ReplayState, SimulationState};
use particles::physics::world::object::{ObjectPhysicsField, ObjectWorld};
use particles::physics::world::particle::{ParticleMaterial, ParticleWorld};
use particles::physics::world::terrain::{TerrainCell, TerrainWorld, world_to_cell};
use particles::render::TerrainRenderDiagnostics;

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
    scenario_requested: bool,
    captured_start: bool,
    wait_frames: u32,
    run_frames: u32,
    max_wait_frames: u32,
    min_avg_fps: f32,
    max_penetration_ratio: f32,
    min_mean_drop: f32,
    start_mean_y: f32,
    elapsed_secs: f32,
    phase: u8,
    end_sample_wait_frames: u32,
    end_sample_wait_max: u32,
}

impl MpmAutoVerifyState {
    fn from_env() -> Self {
        let enabled = env_bool("PARTICLES_AUTOVERIFY_MPM");
        let scenario_name = std::env::var("PARTICLES_AUTOVERIFY_SCENARIO")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "water_drop".to_string());
        let output_path = std::env::var("PARTICLES_AUTOVERIFY_MPM_OUT")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "artifacts/mpm_autoverify.json".to_string());
        let run_frames = std::env::var("PARTICLES_AUTOVERIFY_MPM_FRAMES")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(240);
        let min_avg_fps = std::env::var("PARTICLES_AUTOVERIFY_MIN_FPS")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(20.0);
        let max_penetration_ratio = std::env::var("PARTICLES_AUTOVERIFY_MAX_PENETRATION")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.10);
        let min_mean_drop = std::env::var("PARTICLES_AUTOVERIFY_MIN_DROP")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.05);
        Self {
            enabled,
            scenario_name,
            output_path,
            scenario_requested: false,
            captured_start: false,
            wait_frames: 0,
            run_frames,
            max_wait_frames: 900,
            min_avg_fps,
            max_penetration_ratio,
            min_mean_drop,
            start_mean_y: 0.0,
            elapsed_secs: 0.0,
            phase: 0,
            end_sample_wait_frames: 0,
            end_sample_wait_max: 120,
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
    avg_fps: f32,
    start_mean_y: f32,
    end_mean_y: f32,
    mean_drop: f32,
    terrain_penetration_ratio: f32,
    max_speed_mps: f32,
    max_speed_ok: bool,
    water_surface_height_ok: bool,
    water_surface_assertion_active: bool,
    max_speed_expected: String,
    max_speed_actual: String,
    water_surface_expected: String,
    water_surface_actual: String,
}

fn env_bool(key: &str) -> bool {
    std::env::var(key)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
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

    let water_positions: Vec<Vec2> = particle_world
        .materials()
        .iter()
        .enumerate()
        .filter_map(|(i, &mat)| {
            if matches!(mat, ParticleMaterial::WaterLiquid) {
                Some(particle_world.positions()[i])
            } else {
                None
            }
        })
        .collect();

    if !state.captured_start {
        if water_positions.is_empty() {
            state.wait_frames = state.wait_frames.saturating_add(1);
            if state.wait_frames > state.max_wait_frames {
                let report = MpmAutoVerifyReport {
                    passed: false,
                    note: "No water particles were available for MPM verification.".to_string(),
                    scenario: state.scenario_name.clone(),
                    sampled_particles: 0,
                    run_frames: 0,
                    avg_fps: 0.0,
                    start_mean_y: 0.0,
                    end_mean_y: 0.0,
                    mean_drop: 0.0,
                    terrain_penetration_ratio: 0.0,
                    max_speed_mps: 0.0,
                    max_speed_ok: false,
                    water_surface_height_ok: false,
                    water_surface_assertion_active: false,
                    max_speed_expected: String::new(),
                    max_speed_actual: String::new(),
                    water_surface_expected: String::new(),
                    water_surface_actual: String::new(),
                };
                write_report(&state.output_path, &report);
                exit_writer.write(bevy::app::AppExit::from_code(4));
            }
            return;
        }
        state.start_mean_y =
            water_positions.iter().map(|p| p.y).sum::<f32>() / water_positions.len() as f32;
        state.captured_start = true;
        state.wait_frames = 0;
        state.elapsed_secs = 0.0;
        state.phase = 1;
        sim_state.running = true;
        gpu_control.readback_enabled = false;
        return;
    }

    if state.phase == 1 {
        state.wait_frames = state.wait_frames.saturating_add(1);
        state.elapsed_secs += time.delta_secs();
        if state.wait_frames < state.run_frames {
            return;
        }
        // Stop simulation and request one final readback snapshot.
        sim_state.running = false;
        sim_state.step_once = false;
        gpu_control.readback_enabled = true;
        gpu_control.readback_interval_frames = 1;
        state.phase = 2;
        state.end_sample_wait_frames = 0;
        return;
    }

    if state.phase == 2 {
        state.end_sample_wait_frames = state.end_sample_wait_frames.saturating_add(1);
        let end_mean_y_live =
            water_positions.iter().map(|p| p.y).sum::<f32>() / water_positions.len() as f32;
        let got_fresh_sample = (end_mean_y_live - state.start_mean_y).abs() > 1.0e-4;
        if !got_fresh_sample && state.end_sample_wait_frames < state.end_sample_wait_max {
            return;
        }
    }

    if water_positions.is_empty() {
        let report = MpmAutoVerifyReport {
            passed: false,
            note: "Water particles disappeared during MPM verification.".to_string(),
            scenario: state.scenario_name.clone(),
            sampled_particles: 0,
            run_frames: state.run_frames,
            avg_fps: 0.0,
            start_mean_y: state.start_mean_y,
            end_mean_y: 0.0,
            mean_drop: 0.0,
            terrain_penetration_ratio: 1.0,
            max_speed_mps: 0.0,
            max_speed_ok: false,
            water_surface_height_ok: false,
            water_surface_assertion_active: false,
            max_speed_expected: String::new(),
            max_speed_actual: String::new(),
            water_surface_expected: String::new(),
            water_surface_actual: String::new(),
        };
        write_report(&state.output_path, &report);
        exit_writer.write(bevy::app::AppExit::from_code(5));
        return;
    }

    let end_mean_y = water_positions.iter().map(|p| p.y).sum::<f32>() / water_positions.len() as f32;
    let mean_drop = state.start_mean_y - end_mean_y;
    let penetration_count = water_positions
        .iter()
        .filter(|&&pos| {
            let cell = world_to_cell(pos);
            matches!(terrain_world.get_cell_or_generated(cell), TerrainCell::Solid { .. })
        })
        .count();
    let penetration_ratio = penetration_count as f32 / water_positions.len() as f32;
    let avg_fps = state.run_frames as f32 / state.elapsed_secs.max(1.0e-5);

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
        && (!water_surface_assertion_active || water_surface_height_ok);
    let note = if passed {
        "MPM verification passed: falling, terrain interaction, FPS, and assertions are within thresholds."
            .to_string()
    } else {
        format!(
            "MPM verification failed: fps_ok={fps_ok}, drop_ok={drop_ok}, terrain_ok={terrain_ok}, max_speed_ok={max_speed_ok}, water_surface_ok={water_surface_height_ok}, water_surface_active={water_surface_assertion_active}"
        )
    };

    let report = MpmAutoVerifyReport {
        passed,
        note,
        scenario: state.scenario_name.clone(),
        sampled_particles: water_positions.len(),
        run_frames: state.run_frames,
        avg_fps,
        start_mean_y: state.start_mean_y,
        end_mean_y,
        mean_drop,
        terrain_penetration_ratio: penetration_ratio,
        max_speed_mps,
        max_speed_ok,
        water_surface_height_ok,
        water_surface_assertion_active,
        max_speed_expected,
        max_speed_actual,
        water_surface_expected,
        water_surface_actual,
    };
    write_report(&state.output_path, &report);
    gpu_control.readback_enabled = false;
    exit_writer.write(if passed {
        bevy::app::AppExit::Success
    } else {
        bevy::app::AppExit::from_code(6)
    });
}

fn main() {
    let mpm_autoverify = env_bool("PARTICLES_AUTOVERIFY_MPM");
    let readback_interval_frames = std::env::var("PARTICLES_GPU_READBACK_INTERVAL_FRAMES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v > 0)
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
        .insert_resource(MpmAutoVerifyState::from_env())
        .add_plugins(default_plugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(EguiPlugin::default());
    app.add_plugins((
        PhysicsPlugin,
        GpuMpmPlugin,
        InterfacePlugin,
        OverlayPlugin,
        CameraControllerPlugin,
    ))
    .add_systems(Update, run_mpm_autoverify.after(apply_gpu_readback))
    .run();
}
