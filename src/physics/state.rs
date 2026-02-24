use bevy::prelude::*;

#[derive(Resource, Debug, Default)]
pub struct SimulationState {
    pub running: bool,
    pub step_once: bool,
}

#[derive(Resource, Debug)]
pub struct SimulationParallelSettings {
    pub enabled: bool,
}

impl Default for SimulationParallelSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

#[derive(Resource, Debug)]
pub struct TerrainStreamingSettings {
    pub enabled: bool,
    pub load_radius_chunks: i32,
}

impl Default for TerrainStreamingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            load_radius_chunks: 5,
        }
    }
}

#[derive(Resource, Debug)]
pub struct PhysicsRegionSettings {
    pub enabled: bool,
    pub active_radius_chunks: i32,
    pub active_halo_chunks: i32,
    pub far_field_freeze_margin_chunks: i32,
    pub far_field_release_particles_per_frame: usize,
    pub far_field_release_clearance_radius_m: f32,
    pub far_field_release_clearance_max_wait_frames: u16,
}

impl Default for PhysicsRegionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            active_radius_chunks: 4,
            active_halo_chunks: 1,
            far_field_freeze_margin_chunks: 4,
            far_field_release_particles_per_frame: 24,
            far_field_release_clearance_radius_m: 0.35,
            far_field_release_clearance_max_wait_frames: 30,
        }
    }
}

#[derive(Resource, Debug, Default)]
pub struct PhysicsActiveRegion {
    pub active_chunks: Vec<IVec2>,
    pub chunk_min: Option<IVec2>,
    pub chunk_max: Option<IVec2>,
}

#[derive(Resource, Debug, Default)]
pub struct SimulationPerfMetrics {
    pub physics_time_this_frame_secs: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PhysicsStepProfileSegment {
    pub name: String,
    pub wall_duration_ms: f64,
    pub cpu_duration_ms: f64,
}

#[derive(Resource, Debug, Default)]
pub struct PhysicsStepProfiler {
    pub total_duration_ms: f64,
    pub segments: Vec<PhysicsStepProfileSegment>,
}

#[derive(Message)]
pub struct ResetSimulationRequest;

#[derive(Message)]
pub struct LoadDefaultWorldRequest;

#[derive(Message)]
pub struct SaveMapRequest {
    pub slot_name: String,
}

#[derive(Message)]
pub struct LoadMapRequest {
    pub slot_name: String,
}

#[derive(Message)]
pub struct ReplayLoadScenarioRequest {
    pub scenario_name: String,
}

#[derive(Message)]
pub struct ReplaySaveArtifactRequest {
    pub save_final: bool,
}

#[derive(Resource, Debug, Default)]
pub struct ReplayState {
    pub enabled: bool,
    pub scenario_name: Option<String>,
    pub scenario_total_steps: usize,
    pub current_step: usize,
    pub loop_enabled: bool,
    pub status_message: String,
    pub baseline_particle_count: usize,
    pub baseline_solid_cell_count: usize,
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum SimFixedSet {
    Physics,
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum SimUpdateSet {
    Controls,
    Interaction,
    Rendering,
    Ui,
    Overlay,
    Finalize,
}
