use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use bevy::prelude::*;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use super::material::{ParticleMaterial, TerrainMaterial};
use super::object::{ObjectSnapshotData, ObjectWorld};
use super::particle::{ParticleWorld, TERRAIN_BOUNDARY_RADIUS_M};
use super::state::SimulationState;
use super::terrain::{
    CHUNK_SIZE_I32, TerrainCell, TerrainWorld, WORLD_MAX_CHUNK_X, WORLD_MAX_CHUNK_Y,
    WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y,
};

pub const SAVE_VERSION: u32 = 1;
pub const DEFAULT_QUICK_SAVE_SLOT: &str = "quick_save";

#[derive(Debug, Serialize, Deserialize)]
struct SaveSnapshot {
    save_version: u32,
    simulation: SimulationSnapshot,
    terrain_cells: Vec<TerrainCellSnapshot>,
    particles: Vec<ParticleSnapshot>,
    objects: Vec<ObjectSnapshot>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SimulationSnapshot {
    running: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct TerrainCellSnapshot {
    cell: [i32; 2],
    material: SaveTerrainMaterial,
    hp: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParticleSnapshot {
    position: [f32; 2],
    velocity: [f32; 2],
    material: SaveParticleMaterial,
}

#[derive(Debug, Serialize, Deserialize)]
struct ObjectSnapshot {
    id: u32,
    particle_indices: Vec<usize>,
    rest_local: Vec<[f32; 2]>,
    shape_stiffness_alpha: f32,
    shape_iters: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum SaveTerrainMaterial {
    #[serde(alias = "Rock", alias = "Gravel")]
    Stone,
    Soil,
    Sand,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum SaveParticleMaterial {
    #[serde(alias = "Water")]
    WaterLiquid,
    #[serde(alias = "Rock")]
    StoneSolid,
    #[serde(alias = "Gravel")]
    StoneGranular,
    SoilSolid,
    #[serde(alias = "Soil")]
    SoilGranular,
    SandSolid,
    #[serde(alias = "Sand")]
    SandGranular,
}

impl From<TerrainMaterial> for SaveTerrainMaterial {
    fn from(value: TerrainMaterial) -> Self {
        match value {
            TerrainMaterial::Stone => SaveTerrainMaterial::Stone,
            TerrainMaterial::Soil => SaveTerrainMaterial::Soil,
            TerrainMaterial::Sand => SaveTerrainMaterial::Sand,
        }
    }
}

impl From<SaveTerrainMaterial> for TerrainMaterial {
    fn from(value: SaveTerrainMaterial) -> Self {
        match value {
            SaveTerrainMaterial::Stone => TerrainMaterial::Stone,
            SaveTerrainMaterial::Soil => TerrainMaterial::Soil,
            SaveTerrainMaterial::Sand => TerrainMaterial::Sand,
        }
    }
}

impl From<ParticleMaterial> for SaveParticleMaterial {
    fn from(value: ParticleMaterial) -> Self {
        match value {
            ParticleMaterial::WaterLiquid => SaveParticleMaterial::WaterLiquid,
            ParticleMaterial::StoneSolid => SaveParticleMaterial::StoneSolid,
            ParticleMaterial::StoneGranular => SaveParticleMaterial::StoneGranular,
            ParticleMaterial::SoilSolid => SaveParticleMaterial::SoilSolid,
            ParticleMaterial::SoilGranular => SaveParticleMaterial::SoilGranular,
            ParticleMaterial::SandSolid => SaveParticleMaterial::SandSolid,
            ParticleMaterial::SandGranular => SaveParticleMaterial::SandGranular,
        }
    }
}

impl From<SaveParticleMaterial> for ParticleMaterial {
    fn from(value: SaveParticleMaterial) -> Self {
        match value {
            SaveParticleMaterial::WaterLiquid => ParticleMaterial::WaterLiquid,
            SaveParticleMaterial::StoneSolid => ParticleMaterial::StoneSolid,
            SaveParticleMaterial::StoneGranular => ParticleMaterial::StoneGranular,
            SaveParticleMaterial::SoilSolid => ParticleMaterial::SoilSolid,
            SaveParticleMaterial::SoilGranular => ParticleMaterial::SoilGranular,
            SaveParticleMaterial::SandSolid => ParticleMaterial::SandSolid,
            SaveParticleMaterial::SandGranular => ParticleMaterial::SandGranular,
        }
    }
}

pub fn save_root_dir() -> PathBuf {
    if let Some(project_dirs) = ProjectDirs::from("com", "fujiy", "particles") {
        return project_dirs.data_local_dir().join("saves");
    }

    PathBuf::from("saves")
}

pub fn list_save_slots() -> Result<Vec<String>, String> {
    let root = save_root_dir();
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut slots = Vec::new();
    let entries =
        fs::read_dir(&root).map_err(|error| format!("failed to read save directory: {error}"))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("failed to read save entry: {error}"))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if stem.is_empty() {
            continue;
        }
        slots.push(stem.to_string());
    }

    slots.sort();
    Ok(slots)
}

pub fn save_to_slot(
    slot_name: &str,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    objects: &ObjectWorld,
    sim_state: &SimulationState,
) -> Result<PathBuf, String> {
    let slot = sanitize_slot_name(slot_name)?;
    let path = save_root_dir().join(format!("{slot}.json"));
    save_to_path(path.to_string_lossy().as_ref(), terrain, particles, objects, sim_state)?;
    Ok(path)
}

pub fn load_from_slot(
    slot_name: &str,
    terrain: &mut TerrainWorld,
    particles: &mut ParticleWorld,
    objects: &mut ObjectWorld,
    sim_state: &mut SimulationState,
) -> Result<PathBuf, String> {
    let slot = sanitize_slot_name(slot_name)?;
    let path = save_root_dir().join(format!("{slot}.json"));
    load_from_path(
        path.to_string_lossy().as_ref(),
        terrain,
        particles,
        objects,
        sim_state,
    )?;
    Ok(path)
}

pub fn save_to_path(
    path: &str,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    objects: &ObjectWorld,
    sim_state: &SimulationState,
) -> Result<(), String> {
    let snapshot = SaveSnapshot {
        save_version: SAVE_VERSION,
        simulation: SimulationSnapshot {
            running: sim_state.running,
        },
        terrain_cells: collect_terrain_cells(terrain),
        particles: particles
            .positions()
            .iter()
            .zip(particles.vel.iter())
            .zip(particles.materials().iter())
            .map(|((&position, &velocity), &material)| ParticleSnapshot {
                position: [position.x, position.y],
                velocity: [velocity.x, velocity.y],
                material: material.into(),
            })
            .collect(),
        objects: objects
            .snapshot_data()
            .into_iter()
            .map(|object| ObjectSnapshot {
                id: object.id,
                particle_indices: object.particle_indices,
                rest_local: object.rest_local.into_iter().map(|v| [v.x, v.y]).collect(),
                shape_stiffness_alpha: object.shape_stiffness_alpha,
                shape_iters: object.shape_iters,
            })
            .collect(),
    };

    let json = serde_json::to_string_pretty(&snapshot)
        .map_err(|error| format!("failed to serialize snapshot: {error}"))?;
    let save_path = Path::new(path);
    if let Some(parent) = save_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create save directory: {error}"))?;
    }
    fs::write(save_path, json).map_err(|error| format!("failed to write save file: {error}"))?;
    Ok(())
}

fn sanitize_slot_name(slot_name: &str) -> Result<String, String> {
    let trimmed = slot_name.trim();
    if trimmed.is_empty() {
        return Err("save name is empty".to_string());
    }
    let mut sanitized = String::with_capacity(trimmed.len().min(96));
    for ch in trimmed.chars() {
        if ch.is_control() {
            continue;
        }
        let mapped = match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => ch,
        };
        sanitized.push(mapped);
        if sanitized.chars().count() >= 96 {
            break;
        }
    }
    let sanitized = sanitized.trim().trim_matches('.').to_string();
    if sanitized.is_empty() {
        return Err("save name contains no valid characters".to_string());
    }
    Ok(sanitized)
}

pub fn load_from_path(
    path: &str,
    terrain: &mut TerrainWorld,
    particles: &mut ParticleWorld,
    objects: &mut ObjectWorld,
    sim_state: &mut SimulationState,
) -> Result<(), String> {
    let json =
        fs::read_to_string(path).map_err(|error| format!("failed to read save file: {error}"))?;
    let snapshot: SaveSnapshot = serde_json::from_str(&json)
        .map_err(|error| format!("failed to parse save file: {error}"))?;

    if snapshot.save_version != SAVE_VERSION {
        return Err(format!(
            "incompatible save version: file={}, supported={}",
            snapshot.save_version, SAVE_VERSION
        ));
    }
    validate_snapshot(&snapshot)?;

    let min_cell = IVec2::new(
        WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32,
        WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32,
    );
    let max_cell = IVec2::new(
        (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1,
        (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1,
    );

    terrain.reset_fixed_world();
    terrain.fill_rect(min_cell, max_cell, TerrainCell::Empty);
    for cell in &snapshot.terrain_cells {
        let cell_coord = IVec2::new(cell.cell[0], cell.cell[1]);
        let _ = terrain.set_cell(
            cell_coord,
            TerrainCell::Solid {
                material: cell.material.into(),
                hp: cell.hp,
            },
        );
    }
    terrain.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);

    let positions = snapshot
        .particles
        .iter()
        .map(|particle| Vec2::new(particle.position[0], particle.position[1]))
        .collect();
    let velocities = snapshot
        .particles
        .iter()
        .map(|particle| Vec2::new(particle.velocity[0], particle.velocity[1]))
        .collect();
    let materials = snapshot
        .particles
        .iter()
        .map(|particle| particle.material.into())
        .collect();
    particles.restore_from_snapshot(positions, velocities, materials)?;

    let object_snapshots: Vec<ObjectSnapshotData> = snapshot
        .objects
        .iter()
        .map(|object| ObjectSnapshotData {
            id: object.id,
            particle_indices: object.particle_indices.clone(),
            rest_local: object
                .rest_local
                .iter()
                .map(|rest| Vec2::new(rest[0], rest[1]))
                .collect(),
            shape_stiffness_alpha: object.shape_stiffness_alpha,
            shape_iters: object.shape_iters,
        })
        .collect();
    objects.restore_from_snapshot_data(
        &object_snapshots,
        particles.positions(),
        particles.masses(),
    )?;

    sim_state.running = snapshot.simulation.running;
    Ok(())
}

fn collect_terrain_cells(terrain: &TerrainWorld) -> Vec<TerrainCellSnapshot> {
    let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32;
    let max_cell_x = (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1;
    let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32;
    let max_cell_y = (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1;
    let mut cells = Vec::new();
    for y in min_cell_y..=max_cell_y {
        for x in min_cell_x..=max_cell_x {
            let cell_coord = IVec2::new(x, y);
            let TerrainCell::Solid { material, hp } = terrain.get_loaded_cell_or_empty(cell_coord)
            else {
                continue;
            };
            cells.push(TerrainCellSnapshot {
                cell: [x, y],
                material: material.into(),
                hp,
            });
        }
    }
    cells
}

fn validate_snapshot(snapshot: &SaveSnapshot) -> Result<(), String> {
    let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32;
    let max_cell_x = (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1;
    let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32;
    let max_cell_y = (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1;

    for cell in &snapshot.terrain_cells {
        let [x, y] = cell.cell;
        if x < min_cell_x || x > max_cell_x || y < min_cell_y || y > max_cell_y {
            return Err(format!("terrain cell out of bounds: [{x}, {y}]"));
        }
    }

    for (index, particle) in snapshot.particles.iter().enumerate() {
        if !particle.position[0].is_finite()
            || !particle.position[1].is_finite()
            || !particle.velocity[0].is_finite()
            || !particle.velocity[1].is_finite()
        {
            return Err(format!("particle {index} contains non-finite values"));
        }
    }

    let mut seen_ids = HashSet::new();
    for object in &snapshot.objects {
        if !seen_ids.insert(object.id) {
            return Err(format!("duplicate object id: {}", object.id));
        }
        if object.particle_indices.len() != object.rest_local.len() {
            return Err(format!(
                "object {} has mismatched particle/rest counts",
                object.id
            ));
        }
        for &particle_index in &object.particle_indices {
            if particle_index >= snapshot.particles.len() {
                return Err(format!(
                    "object {} references out-of-range particle index {}",
                    object.id, particle_index
                ));
            }
        }
    }

    Ok(())
}
