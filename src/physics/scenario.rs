use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use image::{ColorType, ImageFormat};
use serde::Serialize;

use super::material::{ParticleMaterial, TerrainMaterial};
use super::object::{
    OBJECT_SHAPE_ITERS, OBJECT_SHAPE_STIFFNESS_ALPHA, ObjectPhysicsField, ObjectWorld,
};
use super::particle::{FIXED_DT, ParticleActivityState, ParticleWorld, TERRAIN_BOUNDARY_RADIUS_M};
use super::terrain::world_to_cell;
use super::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, TerrainCell, TerrainWorld, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y,
    WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y,
};

pub const TEST_ARTIFACT_ROOT: &str = "artifacts/tests";
pub const SCENARIO_ARTIFACT_PNG_ENV: &str = "PARTICLES_SCENARIO_ARTIFACT_PNG";
const DEFAULT_PENETRATION_EPSILON_M: f32 = 1.0e-3;
const FINAL_STATE_PNG_CELL_RESOLUTION: u32 = 8;
const FINAL_STATE_PNG_BACKGROUND: [u8; 4] = [14, 18, 24, 255];

#[derive(Clone, Copy, Debug)]
pub struct CellRect {
    pub min: IVec2,
    pub max: IVec2,
}

impl CellRect {
    pub const fn new(min: IVec2, max: IVec2) -> Self {
        Self { min, max }
    }

    fn cells(self) -> Vec<IVec2> {
        let min_x = self.min.x.min(self.max.x);
        let max_x = self.min.x.max(self.max.x);
        let min_y = self.min.y.min(self.max.y);
        let max_y = self.min.y.max(self.max.y);
        let mut cells = Vec::with_capacity(((max_x - min_x + 1) * (max_y - min_y + 1)) as usize);
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                cells.push(IVec2::new(x, y));
            }
        }
        cells
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TerrainFillSpec {
    pub rect: CellRect,
    pub material: TerrainMaterial,
}

#[derive(Clone, Copy, Debug)]
pub struct ParticleSpawnSpec {
    pub rect: CellRect,
    pub material: ParticleMaterial,
    pub initial_velocity: Vec2,
}

#[derive(Clone, Debug)]
pub struct ObjectSpawnSpec {
    pub cells: Vec<IVec2>,
    pub material: ParticleMaterial,
    pub initial_velocity: Vec2,
    pub shape_stiffness_alpha: f32,
    pub shape_iters: usize,
}

impl ObjectSpawnSpec {
    fn from_rects(
        rects: &[CellRect],
        material: ParticleMaterial,
        initial_velocity: Vec2,
        shape_stiffness_alpha: f32,
        shape_iters: usize,
    ) -> Self {
        let mut cells = Vec::new();
        for rect in rects {
            cells.extend(rect.cells());
        }
        cells.sort_by_key(|cell| (cell.y, cell.x));
        cells.dedup();
        Self {
            cells,
            material,
            initial_velocity,
            shape_stiffness_alpha,
            shape_iters,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScenarioThresholds {
    pub max_penetration_rate: Option<f32>,
    pub max_max_speed_mps: Option<f32>,
    pub min_sleep_ratio: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct WaterSurfaceAssertionSpec {
    pub active_after_seconds: f32,
    pub basin_min_x: i32,
    pub basin_max_x: i32,
    pub margin_cells: i32,
}

#[derive(Clone, Debug, Default)]
pub struct ScenarioSpec {
    pub name: String,
    pub reset_fixed_world: bool,
    pub loaded_chunk_min: Option<IVec2>,
    pub loaded_chunk_max: Option<IVec2>,
    pub terrain_fills: Vec<TerrainFillSpec>,
    pub free_particles: Vec<ParticleSpawnSpec>,
    pub objects: Vec<ObjectSpawnSpec>,
    pub step_count: usize,
    pub thresholds: ScenarioThresholds,
    pub water_surface_assertion: Option<WaterSurfaceAssertionSpec>,
}

#[derive(Clone, Debug)]
pub struct ScenarioRunOutput {
    pub metrics: ScenarioMetrics,
    pub violations: Vec<String>,
    pub artifact_dir: PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub struct ScenarioArtifactOptions {
    pub write_final_state_png: bool,
}

impl Default for ScenarioArtifactOptions {
    fn default() -> Self {
        Self {
            write_final_state_png: true,
        }
    }
}

impl ScenarioArtifactOptions {
    pub fn from_env() -> Self {
        match std::env::var(SCENARIO_ARTIFACT_PNG_ENV) {
            Ok(raw) => {
                let mut options = Self::default();
                if let Some(enabled) = parse_toggle_value(&raw) {
                    options.write_final_state_png = enabled;
                }
                options
            }
            Err(_) => Self::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ScenarioMetrics {
    pub scenario: String,
    pub steps: usize,
    pub particle_count: usize,
    pub object_count: usize,
    pub sleeping_ratio: f32,
    pub max_speed_mps: f32,
    pub terrain_penetration_rate: f32,
    pub object_penetration_rate: f32,
    pub combined_penetration_rate: f32,
}

#[derive(Clone, Debug)]
pub struct ScenarioAssertionResult {
    pub label: String,
    pub expected: String,
    pub actual: String,
    pub ok: bool,
    pub active: bool,
}

#[derive(Debug, Serialize)]
struct ParticleStateJson {
    position: [f32; 2],
    velocity: [f32; 2],
    material: &'static str,
    activity: &'static str,
}

#[derive(Debug, Serialize)]
struct TerrainCellJson {
    cell: [i32; 2],
    material: &'static str,
}

#[derive(Debug, Serialize)]
struct ObjectStateJson {
    id: u32,
    particle_count: usize,
    shape_stiffness_alpha: f32,
    shape_iters: usize,
}

#[derive(Debug, Serialize)]
struct FinalStateJson {
    scenario: String,
    steps: usize,
    particles: Vec<ParticleStateJson>,
    terrain_cells: Vec<TerrainCellJson>,
    objects: Vec<ObjectStateJson>,
}

pub fn default_scenario_specs() -> Vec<ScenarioSpec> {
    let world_min_cell: i32 = -32;
    let world_max_cell: i32 = 31;
    let loaded_chunk_min = IVec2::new(
        world_min_cell.div_euclid(CHUNK_SIZE_I32),
        world_min_cell.div_euclid(CHUNK_SIZE_I32),
    );
    let loaded_chunk_max = IVec2::new(
        world_max_cell.div_euclid(CHUNK_SIZE_I32),
        world_max_cell.div_euclid(CHUNK_SIZE_I32),
    );
    let wall_thickness = 2;
    let left_wall_max = world_min_cell + wall_thickness - 1;
    let right_wall_min = world_max_cell - wall_thickness + 1;

    vec![
        ScenarioSpec {
            name: "objects_drop".to_string(),
            reset_fixed_world: false,
            loaded_chunk_min: Some(loaded_chunk_min),
            loaded_chunk_max: Some(loaded_chunk_max),
            terrain_fills: vec![
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(world_min_cell, world_min_cell),
                        IVec2::new(world_max_cell, world_min_cell + wall_thickness - 1),
                    ),
                    material: TerrainMaterial::Stone,
                },
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(world_min_cell, world_min_cell),
                        IVec2::new(left_wall_max, world_max_cell),
                    ),
                    material: TerrainMaterial::Stone,
                },
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(right_wall_min, world_min_cell),
                        IVec2::new(world_max_cell, world_max_cell),
                    ),
                    material: TerrainMaterial::Stone,
                },
            ],
            free_particles: Vec::new(),
            objects: vec![
                ObjectSpawnSpec::from_rects(
                    &[
                        CellRect::new(IVec2::new(-24, -14), IVec2::new(-22, -14)),
                        CellRect::new(IVec2::new(-24, -13), IVec2::new(-24, -12)),
                    ],
                    ParticleMaterial::StoneSolid,
                    Vec2::ZERO,
                    OBJECT_SHAPE_STIFFNESS_ALPHA,
                    OBJECT_SHAPE_ITERS,
                ),
                ObjectSpawnSpec::from_rects(
                    &[
                        CellRect::new(IVec2::new(-2, -14), IVec2::new(0, -14)),
                        CellRect::new(IVec2::new(0, -13), IVec2::new(0, -12)),
                    ],
                    ParticleMaterial::StoneSolid,
                    Vec2::ZERO,
                    OBJECT_SHAPE_STIFFNESS_ALPHA,
                    OBJECT_SHAPE_ITERS,
                ),
                ObjectSpawnSpec::from_rects(
                    &[
                        CellRect::new(IVec2::new(17, -14), IVec2::new(19, -14)),
                        CellRect::new(IVec2::new(18, -13), IVec2::new(18, -12)),
                    ],
                    ParticleMaterial::StoneSolid,
                    Vec2::ZERO,
                    OBJECT_SHAPE_STIFFNESS_ALPHA,
                    OBJECT_SHAPE_ITERS,
                ),
            ],
            step_count: 300,
            thresholds: ScenarioThresholds {
                max_penetration_rate: Some(0.02),
                max_max_speed_mps: Some(25.0),
                min_sleep_ratio: None,
            },
            water_surface_assertion: None,
        },
        ScenarioSpec {
            name: "water_drop".to_string(),
            reset_fixed_world: false,
            loaded_chunk_min: Some(loaded_chunk_min),
            loaded_chunk_max: Some(loaded_chunk_max),
            terrain_fills: vec![
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(world_min_cell, world_min_cell),
                        IVec2::new(world_max_cell, world_min_cell + wall_thickness - 1),
                    ),
                    material: TerrainMaterial::Stone,
                },
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(world_min_cell, world_min_cell),
                        IVec2::new(left_wall_max, world_max_cell),
                    ),
                    material: TerrainMaterial::Stone,
                },
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(right_wall_min, world_min_cell),
                        IVec2::new(world_max_cell, world_max_cell),
                    ),
                    material: TerrainMaterial::Stone,
                },
            ],
            free_particles: vec![ParticleSpawnSpec {
                rect: CellRect::new(IVec2::new(-10, -2), IVec2::new(10, 14)),
                material: ParticleMaterial::WaterLiquid,
                initial_velocity: Vec2::ZERO,
            }],
            objects: Vec::new(),
            step_count: 600,
            thresholds: ScenarioThresholds {
                max_penetration_rate: Some(0.02),
                max_max_speed_mps: Some(30.0),
                min_sleep_ratio: None,
            },
            water_surface_assertion: Some(WaterSurfaceAssertionSpec {
                active_after_seconds: 8.0,
                basin_min_x: world_min_cell + wall_thickness,
                basin_max_x: right_wall_min - 1,
                margin_cells: 2,
            }),
        },
        ScenarioSpec {
            name: "terrain_contact_stability".to_string(),
            reset_fixed_world: false,
            loaded_chunk_min: Some(loaded_chunk_min),
            loaded_chunk_max: Some(loaded_chunk_max),
            terrain_fills: vec![
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(world_min_cell, world_min_cell),
                        IVec2::new(world_max_cell, world_min_cell + wall_thickness - 1),
                    ),
                    material: TerrainMaterial::Stone,
                },
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(world_min_cell, world_min_cell),
                        IVec2::new(left_wall_max, world_max_cell),
                    ),
                    material: TerrainMaterial::Stone,
                },
                TerrainFillSpec {
                    rect: CellRect::new(
                        IVec2::new(right_wall_min, world_min_cell),
                        IVec2::new(world_max_cell, world_max_cell),
                    ),
                    material: TerrainMaterial::Stone,
                },
                TerrainFillSpec {
                    rect: CellRect::new(IVec2::new(-8, 2), IVec2::new(-3, 2)),
                    material: TerrainMaterial::Stone,
                },
            ],
            free_particles: vec![ParticleSpawnSpec {
                rect: CellRect::new(IVec2::new(-6, 8), IVec2::new(6, 12)),
                material: ParticleMaterial::WaterLiquid,
                initial_velocity: Vec2::ZERO,
            }],
            objects: Vec::new(),
            step_count: 180,
            thresholds: ScenarioThresholds {
                max_penetration_rate: Some(0.02),
                max_max_speed_mps: Some(35.0),
                min_sleep_ratio: None,
            },
            water_surface_assertion: None,
        },
    ]
}

pub fn default_scenario_names() -> Vec<String> {
    default_scenario_specs()
        .into_iter()
        .map(|spec| spec.name)
        .collect()
}

pub fn default_scenario_spec_by_name(name: &str) -> Option<ScenarioSpec> {
    default_scenario_specs()
        .into_iter()
        .find(|spec| spec.name == name)
}

pub fn apply_scenario_spec(
    spec: &ScenarioSpec,
    terrain: &mut TerrainWorld,
    particles: &mut ParticleWorld,
    objects: &mut ObjectWorld,
    object_field: &mut ObjectPhysicsField,
) -> Result<(), String> {
    if spec.reset_fixed_world {
        terrain.reset_fixed_world();
    } else {
        terrain.clear();
        if let (Some(min_chunk), Some(max_chunk)) = (spec.loaded_chunk_min, spec.loaded_chunk_max) {
            for y in min_chunk.y.min(max_chunk.y)..=min_chunk.y.max(max_chunk.y) {
                for x in min_chunk.x.min(max_chunk.x)..=min_chunk.x.max(max_chunk.x) {
                    terrain.ensure_chunk_loaded(IVec2::new(x, y));
                }
            }
        }
    }
    terrain.clear_loaded_cells();
    for fill in &spec.terrain_fills {
        terrain.fill_rect(
            fill.rect.min,
            fill.rect.max,
            TerrainCell::solid(fill.material),
        );
    }
    terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);

    particles.restore_from_snapshot(Vec::new(), Vec::new(), Vec::new())?;

    for spawn in &spec.free_particles {
        let cells = spawn.rect.cells();
        particles.spawn_material_particles_from_cells(
            &cells,
            spawn.material,
            spawn.initial_velocity,
        );
    }

    objects.clear();
    for object_spawn in &spec.objects {
        let indices = particles.spawn_material_particles_from_cells(
            &object_spawn.cells,
            object_spawn.material,
            object_spawn.initial_velocity,
        );
        objects.create_object(
            indices,
            particles.positions(),
            particles.masses(),
            object_spawn.shape_stiffness_alpha,
            object_spawn.shape_iters,
        );
    }
    object_field.clear();
    objects.update_physics_field(particles.positions(), particles.masses(), object_field);
    Ok(())
}

pub fn write_scenario_artifacts_for_state(
    spec: &ScenarioSpec,
    step_count: usize,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    objects: &ObjectWorld,
    object_field: &ObjectPhysicsField,
) -> Result<PathBuf, String> {
    write_scenario_artifacts_for_state_with_options(
        spec,
        step_count,
        terrain,
        particles,
        objects,
        object_field,
        ScenarioArtifactOptions::from_env(),
    )
}

pub fn write_scenario_artifacts_for_state_with_options(
    spec: &ScenarioSpec,
    step_count: usize,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    objects: &ObjectWorld,
    object_field: &ObjectPhysicsField,
    options: ScenarioArtifactOptions,
) -> Result<PathBuf, String> {
    let metrics = compute_metrics(spec, step_count, terrain, particles, objects, object_field);
    let artifact_dir = artifact_dir_for_scenario(&spec.name);
    fs::create_dir_all(&artifact_dir)
        .map_err(|error| format!("failed to create artifact directory: {error}"))?;
    let final_state = capture_final_state_json(spec, step_count, terrain, particles, objects);
    write_json_file(&artifact_dir.join("final_state.json"), &final_state)?;
    write_json_file(&artifact_dir.join("metrics.json"), &metrics)?;
    if options.write_final_state_png {
        write_final_state_png(&artifact_dir.join("final_state.png"), terrain, particles)?;
    }
    Ok(artifact_dir)
}

pub fn run_scenario_and_write_artifacts(spec: &ScenarioSpec) -> Result<ScenarioRunOutput, String> {
    let mut terrain = TerrainWorld::default();
    let mut particles = ParticleWorld::default();
    let mut objects = ObjectWorld::default();
    let mut object_field = ObjectPhysicsField::default();
    apply_scenario_spec(
        spec,
        &mut terrain,
        &mut particles,
        &mut objects,
        &mut object_field,
    )?;
    let baseline_particle_count = particles.particle_count();
    let baseline_solid_cell_count = count_solid_cells(&terrain);

    for _ in 0..spec.step_count {
        objects.update_physics_field(particles.positions(), particles.masses(), &mut object_field);
        terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        particles.step_if_running(&terrain, &object_field, &mut objects, true);
        if particles.apply_pending_terrain_fractures(&mut terrain, &mut objects) {
            terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
        }
    }

    objects.update_physics_field(particles.positions(), particles.masses(), &mut object_field);
    let metrics = compute_metrics(
        spec,
        spec.step_count,
        &terrain,
        &particles,
        &objects,
        &object_field,
    );
    let assertions = evaluate_assertions_with_context(
        spec,
        spec.step_count,
        &metrics,
        &terrain,
        &particles,
        baseline_particle_count,
        baseline_solid_cell_count,
        count_solid_cells(&terrain),
    );
    let violations = assertions
        .into_iter()
        .filter(|assertion| assertion.active && !assertion.ok)
        .map(|assertion| {
            format!(
                "{} {} (actual: {})",
                assertion.label, assertion.expected, assertion.actual
            )
        })
        .collect();
    let artifact_dir = write_scenario_artifacts_for_state(
        spec,
        spec.step_count,
        &terrain,
        &particles,
        &objects,
        &object_field,
    )?;

    Ok(ScenarioRunOutput {
        metrics,
        violations,
        artifact_dir,
    })
}

pub fn run_default_scenarios() -> Result<Vec<ScenarioRunOutput>, String> {
    let mut outputs = Vec::new();
    for spec in default_scenario_specs() {
        outputs.push(run_scenario_and_write_artifacts(&spec)?);
    }
    Ok(outputs)
}

pub fn count_solid_cells(terrain: &TerrainWorld) -> usize {
    let mut count = 0usize;
    for chunk_coord in terrain.loaded_chunk_coords() {
        let base = chunk_coord * CHUNK_SIZE_I32;
        for local_y in 0..CHUNK_SIZE_I32 {
            for local_x in 0..CHUNK_SIZE_I32 {
                let cell = base + IVec2::new(local_x, local_y);
                if matches!(
                    terrain.get_loaded_cell_or_empty(cell),
                    TerrainCell::Solid { .. }
                ) {
                    count += 1;
                }
            }
        }
    }
    count
}

pub fn evaluate_scenario_state(
    spec: &ScenarioSpec,
    step_count: usize,
    baseline_particle_count: usize,
    baseline_solid_cell_count: usize,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    objects: &ObjectWorld,
    object_field: &ObjectPhysicsField,
) -> (ScenarioMetrics, Vec<ScenarioAssertionResult>) {
    let metrics = compute_metrics(spec, step_count, terrain, particles, objects, object_field);
    let assertions = evaluate_assertions_with_context(
        spec,
        step_count,
        &metrics,
        terrain,
        particles,
        baseline_particle_count,
        baseline_solid_cell_count,
        count_solid_cells(terrain),
    );
    (metrics, assertions)
}

fn compute_metrics(
    spec: &ScenarioSpec,
    step_count: usize,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    objects: &ObjectWorld,
    object_field: &ObjectPhysicsField,
) -> ScenarioMetrics {
    let mut max_speed = 0.0f32;
    for &velocity in &particles.vel {
        max_speed = max_speed.max(velocity.length());
    }

    let particle_count = particles.particle_count();
    let sleeping_count = particles
        .activity_states()
        .iter()
        .filter(|&&state| matches!(state, ParticleActivityState::Sleeping))
        .count();

    let mut terrain_penetration_count = 0usize;
    let mut object_penetration_count = 0usize;
    let mut candidates = Vec::new();

    for (index, &position) in particles.positions().iter().enumerate() {
        if let Some((distance, _)) = terrain.sample_signed_distance_and_normal(position) {
            if distance < -DEFAULT_PENETRATION_EPSILON_M {
                terrain_penetration_count += 1;
            }
        }

        let mut penetrated_object = false;
        object_field.gather_candidate_object_ids(position, &mut candidates);
        let owner = objects.object_of_particle(index);
        for &object_id in &candidates {
            if owner == Some(object_id) {
                continue;
            }
            let Some(sample) = objects.evaluate_object_sdf(object_id, position) else {
                continue;
            };
            if sample.distance_m < -DEFAULT_PENETRATION_EPSILON_M {
                penetrated_object = true;
                break;
            }
        }
        if penetrated_object {
            object_penetration_count += 1;
        }
    }

    let particle_denominator = particle_count.max(1) as f32;
    let combined_penetration_count = terrain_penetration_count.max(object_penetration_count);

    ScenarioMetrics {
        scenario: spec.name.clone(),
        steps: step_count,
        particle_count,
        object_count: objects.objects().len(),
        sleeping_ratio: sleeping_count as f32 / particle_denominator,
        max_speed_mps: max_speed,
        terrain_penetration_rate: terrain_penetration_count as f32 / particle_denominator,
        object_penetration_rate: object_penetration_count as f32 / particle_denominator,
        combined_penetration_rate: combined_penetration_count as f32 / particle_denominator,
    }
}

fn evaluate_assertions_with_context(
    spec: &ScenarioSpec,
    step_count: usize,
    metrics: &ScenarioMetrics,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    baseline_particle_count: usize,
    baseline_solid_cell_count: usize,
    current_solid_cell_count: usize,
) -> Vec<ScenarioAssertionResult> {
    let mut rows = Vec::new();
    let thresholds = spec.thresholds;
    if let Some(max_penetration_rate) = thresholds.max_penetration_rate {
        rows.push(ScenarioAssertionResult {
            label: "penetration_rate".to_string(),
            expected: format!("<= {:.6}", max_penetration_rate),
            actual: format!("{:.6}", metrics.combined_penetration_rate),
            ok: metrics.combined_penetration_rate <= max_penetration_rate,
            active: true,
        });
    }
    if let Some(max_speed) = thresholds.max_max_speed_mps {
        rows.push(ScenarioAssertionResult {
            label: "max_speed_mps".to_string(),
            expected: format!("<= {:.6}", max_speed),
            actual: format!("{:.6}", metrics.max_speed_mps),
            ok: metrics.max_speed_mps <= max_speed,
            active: true,
        });
    }
    if let Some(min_sleep_ratio) = thresholds.min_sleep_ratio {
        rows.push(ScenarioAssertionResult {
            label: "sleeping_ratio".to_string(),
            expected: format!(">= {:.6}", min_sleep_ratio),
            actual: format!("{:.6}", metrics.sleeping_ratio),
            ok: metrics.sleeping_ratio >= min_sleep_ratio,
            active: true,
        });
    }
    rows.push(ScenarioAssertionResult {
        label: "mass_particle_count".to_string(),
        expected: format!("== {}", baseline_particle_count),
        actual: metrics.particle_count.to_string(),
        ok: metrics.particle_count == baseline_particle_count,
        active: true,
    });
    rows.push(ScenarioAssertionResult {
        label: "mass_solid_cell_count".to_string(),
        expected: format!("== {}", baseline_solid_cell_count),
        actual: current_solid_cell_count.to_string(),
        ok: current_solid_cell_count == baseline_solid_cell_count,
        active: true,
    });
    if let Some(surface) = spec.water_surface_assertion {
        let active_after_steps = (surface.active_after_seconds / FIXED_DT).ceil() as usize;
        let active = step_count >= active_after_steps;
        let water_cell_count = initial_water_cell_count(spec).max(1);
        let basin_width = (surface.basin_max_x - surface.basin_min_x + 1).max(1) as usize;
        let expected_height_cells = water_cell_count.div_ceil(basin_width) as i32;
        let initial_min_y_cell = initial_water_min_y_cell(spec).unwrap_or(0);
        let bottom_free_y_cell =
            basin_bottom_free_y_cell(terrain, surface).unwrap_or(initial_min_y_cell);
        let expected_max_y_cell =
            bottom_free_y_cell + expected_height_cells - 1 + surface.margin_cells;
        let mut water_y_cells: Vec<i32> = particles
            .positions()
            .iter()
            .zip(particles.materials().iter())
            .filter_map(|(&position, &material)| {
                if matches!(material, ParticleMaterial::WaterLiquid) {
                    Some((position.y / CELL_SIZE_M).floor() as i32)
                } else {
                    None
                }
            })
            .collect();
        water_y_cells.sort_unstable();
        let actual_max_y_cell = if water_y_cells.is_empty() {
            i32::MIN
        } else {
            let idx = ((water_y_cells.len() as f32) * 0.95).floor() as usize;
            water_y_cells[idx.min(water_y_cells.len() - 1)]
        };
        rows.push(ScenarioAssertionResult {
            label: "water_surface_height_p95".to_string(),
            expected: format!("<= {}", expected_max_y_cell),
            actual: actual_max_y_cell.to_string(),
            ok: actual_max_y_cell <= expected_max_y_cell,
            active,
        });
    }
    rows
}

fn initial_water_cell_count(spec: &ScenarioSpec) -> usize {
    spec.free_particles
        .iter()
        .filter(|spawn| matches!(spawn.material, ParticleMaterial::WaterLiquid))
        .map(|spawn| spawn.rect.cells().len())
        .sum()
}

fn initial_water_min_y_cell(spec: &ScenarioSpec) -> Option<i32> {
    spec.free_particles
        .iter()
        .filter(|spawn| matches!(spawn.material, ParticleMaterial::WaterLiquid))
        .map(|spawn| spawn.rect.min.y.min(spawn.rect.max.y))
        .min()
}

fn basin_bottom_free_y_cell(
    terrain: &TerrainWorld,
    surface: WaterSurfaceAssertionSpec,
) -> Option<i32> {
    let (min_cell, max_cell) = terrain.loaded_cell_bounds()?;
    let mut highest_solid_y = i32::MIN;
    for y in min_cell.y..=max_cell.y {
        for x in surface.basin_min_x..=surface.basin_max_x {
            if matches!(
                terrain.get_loaded_cell_or_empty(IVec2::new(x, y)),
                TerrainCell::Solid { .. }
            ) {
                highest_solid_y = highest_solid_y.max(y);
            }
        }
    }
    if highest_solid_y == i32::MIN {
        None
    } else {
        Some(highest_solid_y + 1)
    }
}

fn capture_final_state_json(
    spec: &ScenarioSpec,
    step_count: usize,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    objects: &ObjectWorld,
) -> FinalStateJson {
    let particle_states = particles
        .positions()
        .iter()
        .zip(particles.vel.iter())
        .zip(particles.materials().iter())
        .zip(particles.activity_states().iter())
        .map(
            |(((&position, &velocity), &material), &activity)| ParticleStateJson {
                position: [position.x, position.y],
                velocity: [velocity.x, velocity.y],
                material: particle_material_label(material),
                activity: activity_label(activity),
            },
        )
        .collect();

    let mut terrain_cells = Vec::new();
    for y in (WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32)..=((WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1) {
        for x in
            (WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32)..=((WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1)
        {
            let cell = IVec2::new(x, y);
            let TerrainCell::Solid { material, .. } = terrain.get_loaded_cell_or_empty(cell) else {
                continue;
            };
            terrain_cells.push(TerrainCellJson {
                cell: [x, y],
                material: terrain_material_label(material),
            });
        }
    }

    let objects_json = objects
        .snapshot_data()
        .into_iter()
        .map(|snapshot| ObjectStateJson {
            id: snapshot.id,
            particle_count: snapshot.particle_indices.len(),
            shape_stiffness_alpha: snapshot.shape_stiffness_alpha,
            shape_iters: snapshot.shape_iters,
        })
        .collect();

    FinalStateJson {
        scenario: spec.name.clone(),
        steps: step_count,
        particles: particle_states,
        terrain_cells,
        objects: objects_json,
    }
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize {}: {error}", path.display()))?;
    fs::write(path, json).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn write_final_state_png(
    path: &Path,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
) -> Result<(), String> {
    let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32;
    let max_cell_x = (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1;
    let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32;
    let max_cell_y = (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1;

    let cell_width = (max_cell_x - min_cell_x + 1) as u32;
    let cell_height = (max_cell_y - min_cell_y + 1) as u32;
    let width = cell_width * FINAL_STATE_PNG_CELL_RESOLUTION;
    let height = cell_height * FINAL_STATE_PNG_CELL_RESOLUTION;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.copy_from_slice(&FINAL_STATE_PNG_BACKGROUND);
    }

    for y in min_cell_y..=max_cell_y {
        for x in min_cell_x..=max_cell_x {
            let TerrainCell::Solid { material, .. } =
                terrain.get_loaded_cell_or_empty(IVec2::new(x, y))
            else {
                continue;
            };
            let palette = terrain_palette(material);
            let color = palette[deterministic_palette_index(x, y)];
            let local_x = (x - min_cell_x) as u32;
            let local_y = (y - min_cell_y) as u32;
            fill_cell_rect(
                &mut pixels,
                width,
                height,
                local_x,
                local_y,
                FINAL_STATE_PNG_CELL_RESOLUTION,
                color,
            );
        }
    }

    let particle_radius_px_min = (FINAL_STATE_PNG_CELL_RESOLUTION as f32 * 0.35).max(1.0);
    for (&position, &material) in particles
        .positions()
        .iter()
        .zip(particles.materials().iter())
    {
        let cell = world_to_cell(position);
        let color = particle_palette(material)[deterministic_palette_index(cell.x, cell.y)];
        let rel_x = (position.x / CELL_SIZE_M) - min_cell_x as f32;
        let rel_y = (position.y / CELL_SIZE_M) - min_cell_y as f32;
        let center_x = rel_x * FINAL_STATE_PNG_CELL_RESOLUTION as f32;
        let center_y = rel_y * FINAL_STATE_PNG_CELL_RESOLUTION as f32;
        let world_radius_px = (particle_draw_radius_in_cells(material)
            * FINAL_STATE_PNG_CELL_RESOLUTION as f32)
            .max(particle_radius_px_min);
        draw_disc(
            &mut pixels,
            width,
            height,
            center_x,
            center_y,
            world_radius_px,
            color,
        );
    }

    image::save_buffer_with_format(
        path,
        &pixels,
        width,
        height,
        ColorType::Rgba8,
        ImageFormat::Png,
    )
    .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn fill_cell_rect(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    cell_x: u32,
    cell_y: u32,
    scale: u32,
    color: [u8; 4],
) {
    let start_x = cell_x.saturating_mul(scale);
    let end_x = (start_x + scale).min(width);
    let start_y = cell_y.saturating_mul(scale);
    let end_y = (start_y + scale).min(height);
    for py in start_y..end_y {
        for px in start_x..end_x {
            blend_pixel(pixels, width, height, px as i32, py as i32, color);
        }
    }
}

fn draw_disc(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    radius_px: f32,
    color: [u8; 4],
) {
    let min_x = (center_x - radius_px).floor() as i32;
    let max_x = (center_x + radius_px).ceil() as i32;
    let min_y = (center_y - radius_px).floor() as i32;
    let max_y = (center_y + radius_px).ceil() as i32;
    let r2 = radius_px * radius_px;
    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let dx = px as f32 - center_x;
            let dy = py as f32 - center_y;
            if dx * dx + dy * dy > r2 {
                continue;
            }
            blend_pixel(pixels, width, height, px, py, color);
        }
    }
}

fn blend_pixel(pixels: &mut [u8], width: u32, height: u32, px: i32, py: i32, src: [u8; 4]) {
    if px < 0 || py < 0 || px >= width as i32 || py >= height as i32 {
        return;
    }
    let image_y = (height as i32 - 1 - py) as u32;
    let idx = ((image_y * width + px as u32) * 4) as usize;
    let dst = &mut pixels[idx..idx + 4];
    let src_a = src[3] as f32 / 255.0;
    let inv_a = 1.0 - src_a;
    dst[0] = (src[0] as f32 * src_a + dst[0] as f32 * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    dst[1] = (src[1] as f32 * src_a + dst[1] as f32 * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    dst[2] = (src[2] as f32 * src_a + dst[2] as f32 * inv_a)
        .round()
        .clamp(0.0, 255.0) as u8;
    dst[3] = 255;
}

fn terrain_palette(material: TerrainMaterial) -> [[u8; 4]; 4] {
    match material {
        TerrainMaterial::Stone => [
            [70, 67, 63, 255],
            [83, 79, 74, 255],
            [95, 90, 84, 255],
            [108, 103, 96, 255],
        ],
        TerrainMaterial::Soil => [
            [105, 79, 56, 255],
            [119, 91, 67, 255],
            [133, 103, 78, 255],
            [147, 115, 88, 255],
        ],
        TerrainMaterial::Sand => [
            [170, 150, 110, 255],
            [186, 166, 124, 255],
            [201, 181, 137, 255],
            [216, 196, 150, 255],
        ],
    }
}

fn particle_palette(material: ParticleMaterial) -> [[u8; 4]; 4] {
    match material {
        ParticleMaterial::WaterLiquid => [
            [42, 120, 202, 235],
            [52, 136, 218, 240],
            [65, 152, 228, 245],
            [78, 167, 238, 250],
        ],
        ParticleMaterial::StoneSolid | ParticleMaterial::StoneGranular => {
            terrain_palette(TerrainMaterial::Stone)
        }
        ParticleMaterial::SoilSolid | ParticleMaterial::SoilGranular => {
            terrain_palette(TerrainMaterial::Soil)
        }
        ParticleMaterial::SandSolid | ParticleMaterial::SandGranular => {
            terrain_palette(TerrainMaterial::Sand)
        }
    }
}

fn particle_draw_radius_in_cells(material: ParticleMaterial) -> f32 {
    super::material::particle_properties(material).radius_m / CELL_SIZE_M
}

fn deterministic_palette_index(x: i32, y: i32) -> usize {
    let mut state = (x as u32).wrapping_mul(0x45d9_f3b);
    state ^= (y as u32).wrapping_mul(0x27d4_eb2d);
    state ^= 0xa53c_9e4d;
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    (state & 0b11) as usize
}

fn parse_toggle_value(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn artifact_dir_for_scenario(scenario_name: &str) -> PathBuf {
    let run_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    PathBuf::from(TEST_ARTIFACT_ROOT)
        .join(sanitize_name(scenario_name))
        .join(run_id.to_string())
}

fn sanitize_name(name: &str) -> String {
    let mut output = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    if output.is_empty() {
        "scenario".to_string()
    } else {
        output
    }
}

fn particle_material_label(material: ParticleMaterial) -> &'static str {
    match material {
        ParticleMaterial::WaterLiquid => "WaterLiquid",
        ParticleMaterial::StoneSolid => "StoneSolid",
        ParticleMaterial::StoneGranular => "StoneGranular",
        ParticleMaterial::SoilSolid => "SoilSolid",
        ParticleMaterial::SoilGranular => "SoilGranular",
        ParticleMaterial::SandSolid => "SandSolid",
        ParticleMaterial::SandGranular => "SandGranular",
    }
}

fn terrain_material_label(material: TerrainMaterial) -> &'static str {
    match material {
        TerrainMaterial::Stone => "Stone",
        TerrainMaterial::Soil => "Soil",
        TerrainMaterial::Sand => "Sand",
    }
}

fn activity_label(state: ParticleActivityState) -> &'static str {
    match state {
        ParticleActivityState::Active => "Active",
        ParticleActivityState::Sleeping => "Sleeping",
    }
}

#[cfg(test)]
mod tests {
    use super::parse_toggle_value;

    #[test]
    fn parse_toggle_value_true_variants() {
        for raw in ["1", "true", "yes", "on", " TRUE "] {
            assert_eq!(parse_toggle_value(raw), Some(true));
        }
    }

    #[test]
    fn parse_toggle_value_false_variants() {
        for raw in ["0", "false", "no", "off", " Off "] {
            assert_eq!(parse_toggle_value(raw), Some(false));
        }
    }

    #[test]
    fn parse_toggle_value_invalid() {
        for raw in ["", "maybe", "2", "enable"] {
            assert_eq!(parse_toggle_value(raw), None);
        }
    }
}
