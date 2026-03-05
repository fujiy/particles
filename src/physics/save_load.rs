use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use bevy::prelude::*;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use super::material::{
    DEFAULT_MATERIAL_PARAMS, ParticleMaterial, TerrainMaterial, terrain_boundary_radius_m,
};
use super::state::SimulationState;
use super::world::particle::ParticleWorld;
use super::world::terrain::{CHUNK_SIZE_I32, TERRAIN_GENERATOR_VERSION, TerrainCell, TerrainWorld};

pub const SAVE_VERSION: u32 = 1;
pub const DEFAULT_QUICK_SAVE_SLOT: &str = "quick_save";

#[derive(Debug, Serialize, Deserialize)]
struct SaveSnapshot {
    save_version: u32,
    #[serde(default)]
    generator_version: u32,
    #[serde(default = "default_terrain_generation_enabled")]
    terrain_generation_enabled: bool,
    simulation: SimulationSnapshot,
    #[serde(default)]
    loaded_chunks: Vec<ChunkSnapshot>,
    terrain_cells: Vec<TerrainCellSnapshot>,
    particles: Vec<ParticleSnapshot>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SimulationSnapshot {
    running: bool,
}

fn default_terrain_generation_enabled() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize)]
struct TerrainCellSnapshot {
    cell: [i32; 2],
    material: SaveTerrainMaterial,
    hp: u16,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkSnapshot {
    chunk: [i32; 2],
}

#[derive(Debug, Serialize, Deserialize)]
struct ParticleSnapshot {
    position: [f32; 2],
    velocity: [f32; 2],
    material: SaveParticleMaterial,
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
    sim_state: &SimulationState,
) -> Result<PathBuf, String> {
    let slot = sanitize_slot_name(slot_name)?;
    let path = save_root_dir().join(format!("{slot}.json"));
    save_to_path(
        path.to_string_lossy().as_ref(),
        terrain,
        particles,
        sim_state,
    )?;
    Ok(path)
}

pub fn load_from_slot(
    slot_name: &str,
    terrain: &mut TerrainWorld,
    particles: &mut ParticleWorld,
    sim_state: &mut SimulationState,
) -> Result<PathBuf, String> {
    let slot = sanitize_slot_name(slot_name)?;
    let path = save_root_dir().join(format!("{slot}.json"));
    load_from_path(
        path.to_string_lossy().as_ref(),
        terrain,
        particles,
        sim_state,
    )?;
    Ok(path)
}

pub fn save_to_path(
    path: &str,
    terrain: &TerrainWorld,
    particles: &ParticleWorld,
    sim_state: &SimulationState,
) -> Result<(), String> {
    let snapshot = SaveSnapshot {
        save_version: SAVE_VERSION,
        generator_version: TERRAIN_GENERATOR_VERSION,
        terrain_generation_enabled: terrain.generation_enabled(),
        simulation: SimulationSnapshot {
            running: sim_state.running,
        },
        loaded_chunks: terrain
            .loaded_chunk_coords()
            .into_iter()
            .map(|chunk| ChunkSnapshot {
                chunk: [chunk.x, chunk.y],
            })
            .collect(),
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
    if snapshot.generator_version != TERRAIN_GENERATOR_VERSION {
        return Err(format!(
            "incompatible terrain generator version: file={}, supported={}",
            snapshot.generator_version, TERRAIN_GENERATOR_VERSION
        ));
    }
    validate_snapshot(&snapshot)?;

    terrain.set_generation_enabled(snapshot.terrain_generation_enabled);
    terrain.clear();
    if snapshot.loaded_chunks.is_empty() {
        let mut seen_chunks = HashSet::new();
        for cell in &snapshot.terrain_cells {
            let coord = IVec2::new(cell.cell[0], cell.cell[1]);
            let chunk = IVec2::new(
                coord.x.div_euclid(CHUNK_SIZE_I32),
                coord.y.div_euclid(CHUNK_SIZE_I32),
            );
            if seen_chunks.insert(chunk) {
                terrain.ensure_chunk_loaded(chunk);
            }
        }
    } else {
        for chunk in &snapshot.loaded_chunks {
            terrain.ensure_chunk_loaded(IVec2::new(chunk.chunk[0], chunk.chunk[1]));
        }
    }
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
    terrain.rebuild_static_particles_if_dirty(terrain_boundary_radius_m(DEFAULT_MATERIAL_PARAMS));

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

    sim_state.running = snapshot.simulation.running;
    Ok(())
}

fn collect_terrain_cells(terrain: &TerrainWorld) -> Vec<TerrainCellSnapshot> {
    let mut cells = Vec::new();
    for chunk in terrain.loaded_chunk_coords() {
        let base_cell = chunk * CHUNK_SIZE_I32;
        for local_y in 0..CHUNK_SIZE_I32 {
            for local_x in 0..CHUNK_SIZE_I32 {
                let cell_coord = base_cell + IVec2::new(local_x, local_y);
                let TerrainCell::Solid { material, hp } =
                    terrain.get_loaded_cell_or_empty(cell_coord)
                else {
                    continue;
                };
                cells.push(TerrainCellSnapshot {
                    cell: [cell_coord.x, cell_coord.y],
                    material: material.into(),
                    hp,
                });
            }
        }
    }
    cells
}

fn validate_snapshot(snapshot: &SaveSnapshot) -> Result<(), String> {
    for (index, particle) in snapshot.particles.iter().enumerate() {
        if !particle.position[0].is_finite()
            || !particle.position[1].is_finite()
            || !particle.velocity[0].is_finite()
            || !particle.velocity[1].is_finite()
        {
            return Err(format!("particle {index} contains non-finite values"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::physics::world::particle::ParticleWorld;

    #[test]
    fn load_restores_loaded_chunks_as_empty_when_terrain_cells_are_empty() {
        let mut terrain = TerrainWorld::default();
        let mut particles = ParticleWorld::default();
        let mut sim_state = SimulationState::default();

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("particles_load_regen_{nanos}.json"));
        let json = format!(
            "{{\"save_version\":{},\"generator_version\":{},\"simulation\":{{\"running\":false}},\"loaded_chunks\":[{{\"chunk\":[0,0]}}],\"terrain_cells\":[],\"particles\":[]}}",
            SAVE_VERSION, TERRAIN_GENERATOR_VERSION
        );
        fs::write(&path, json).expect("should write temporary save file");
        load_from_path(
            path.to_str().expect("temp path should be utf-8"),
            &mut terrain,
            &mut particles,
            &mut sim_state,
        )
        .expect("loading regeneration-only snapshot should succeed");

        for local_y in 0..CHUNK_SIZE_I32 {
            for local_x in 0..CHUNK_SIZE_I32 {
                let cell = IVec2::new(local_x, local_y);
                assert_eq!(terrain.get_loaded_cell_or_empty(cell), TerrainCell::Empty);
            }
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn load_rejects_incompatible_generator_version() {
        let mut terrain = TerrainWorld::default();
        let mut particles = ParticleWorld::default();
        let mut sim_state = SimulationState::default();

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("particles_bad_generator_{nanos}.json"));
        let json = format!(
            "{{\"save_version\":{},\"generator_version\":{},\"simulation\":{{\"running\":false}},\"loaded_chunks\":[],\"terrain_cells\":[],\"particles\":[]}}",
            SAVE_VERSION,
            TERRAIN_GENERATOR_VERSION + 1
        );
        fs::write(&path, json).expect("should write temporary save file");
        let error = load_from_path(
            path.to_str().expect("temp path should be utf-8"),
            &mut terrain,
            &mut particles,
            &mut sim_state,
        )
        .expect_err("mismatched generator version should fail");
        assert!(error.contains("incompatible terrain generator version"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn save_and_load_restores_modified_terrain_outside_fixed_bounds() {
        let mut terrain = TerrainWorld::default();
        let particles = ParticleWorld::default();
        let sim_state = SimulationState::default();

        let far_chunk = IVec2::new(12, -9);
        terrain.ensure_chunk_loaded(far_chunk);
        let far_cell = far_chunk * CHUNK_SIZE_I32 + IVec2::new(3, 7);
        assert!(terrain.set_cell(
            far_cell,
            TerrainCell::Solid {
                material: TerrainMaterial::Sand,
                hp: 777,
            }
        ));

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("particles_far_chunk_save_{nanos}.json"));
        save_to_path(
            path.to_str().expect("temp path should be utf-8"),
            &terrain,
            &particles,
            &sim_state,
        )
        .expect("saving snapshot should succeed");

        let mut loaded_terrain = TerrainWorld::default();
        let mut loaded_particles = ParticleWorld::default();
        let mut loaded_sim_state = SimulationState::default();
        load_from_path(
            path.to_str().expect("temp path should be utf-8"),
            &mut loaded_terrain,
            &mut loaded_particles,
            &mut loaded_sim_state,
        )
        .expect("loading snapshot should succeed");

        assert_eq!(
            loaded_terrain.get_loaded_cell_or_empty(far_cell),
            TerrainCell::Solid {
                material: TerrainMaterial::Sand,
                hp: 777,
            }
        );

        let _ = fs::remove_file(path);
    }
}
