use std::collections::{HashSet, VecDeque};

use bevy::prelude::*;

use crate::params::interface::InterfaceColorParams;
use crate::physics::cell_to_world_center;
use crate::physics::material::ParticleMaterial;
use crate::physics::material::terrain_boundary_radius_m;
use crate::physics::profiler::{
    RuntimeProfileLane, RuntimeProfileSnapshot, RuntimeProfileSegment, update_runtime_profile_snapshot,
};
use crate::physics::save_load;
use crate::physics::scenario::{
    count_solid_cells, default_scenario_names, default_scenario_spec_by_name,
};
use crate::physics::state::{
    LoadDefaultWorldRequest, LoadMapRequest, ReplayLoadScenarioRequest, ReplayState,
    ResetSimulationRequest, SaveMapRequest, SimUpdateSet, SimulationState,
};
use crate::physics::world::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, TerrainCell, TerrainMaterial, TerrainWorld, world_to_cell,
};
use crate::render::TerrainRenderDiagnostics;

const TOOLBAR_ICON_SIZE_PX: u32 = 32;
const TOOLBAR_ICON_GRID_SIZE: usize = 8;
const TOOLBAR_ICON_DOT_PX: u32 = TOOLBAR_ICON_SIZE_PX / TOOLBAR_ICON_GRID_SIZE as u32;
type MaterialPattern8 = [[bool; TOOLBAR_ICON_GRID_SIZE]; TOOLBAR_ICON_GRID_SIZE];

const MATERIAL_PATTERN_LIQUID: MaterialPattern8 = [
    [false, false, false, false, false, false, false, false],
    [false, true, true, false, false, false, false, false],
    [true, true, true, true, true, false, false, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
];

const MATERIAL_PATTERN_SOLID: MaterialPattern8 = [
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
];

const MATERIAL_PATTERN_GRANULAR: MaterialPattern8 = [
    [false, false, false, true, true, false, false, false],
    [false, false, false, true, true, false, false, false],
    [false, false, true, true, true, true, false, false],
    [false, false, true, true, true, true, false, false],
    [false, true, true, true, true, true, true, false],
    [false, true, true, true, true, true, true, false],
    [true, true, true, true, true, true, true, true],
    [true, true, true, true, true, true, true, true],
];

const WORLD_TOOLBAR_TOOLS: [WorldTool; 11] = [
    WorldTool::WaterLiquid,
    WorldTool::StoneSolid,
    WorldTool::StoneGranular,
    WorldTool::SoilSolid,
    WorldTool::SoilGranular,
    WorldTool::SandSolid,
    WorldTool::SandGranular,
    WorldTool::GrassSolid,
    WorldTool::GrassGranular,
    WorldTool::Break,
    WorldTool::Delete,
];

mod icons;
mod input_handlers;
mod setup;
mod ui_systems;
mod world_edit;

use icons::create_world_tool_icon_set;

pub struct InterfacePlugin;

impl Plugin for InterfacePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldInteractionState>()
            .init_resource::<SaveLoadUiState>()
            .init_resource::<FpsHudStats>()
            .init_resource::<RuntimeProfileHudState>()
            .add_systems(Startup, setup_simulation_ui)
            .add_systems(
                Update,
                (
                    handle_save_load_open_button_interaction,
                    handle_save_load_reset_button_interaction,
                    handle_sim_play_pause_button_interaction,
                    handle_sim_step_button_interaction,
                    handle_save_load_name_input_button_interaction,
                    handle_save_load_dialog_buttons,
                    handle_save_load_slot_button_interaction,
                    sync_save_load_ime_state,
                    handle_save_load_text_input,
                    handle_world_tool_button_interaction,
                    handle_world_interactions,
                )
                    .chain()
                    .in_set(SimUpdateSet::Interaction),
            )
            .add_systems(
                Update,
                (
                    update_world_tool_button_visuals,
                    update_save_load_open_button_visuals,
                    update_save_load_reset_button_visuals,
                    update_sim_play_pause_button_visuals,
                    update_sim_step_button_visuals,
                    update_sim_play_pause_button_label,
                    update_save_load_name_input_button_visuals,
                    update_save_load_slot_button_visuals,
                    update_save_load_dialog,
                    update_test_assert_panel,
                    update_world_tool_tooltip,
                    update_fps_hud_stats,
                    update_runtime_profile_snapshot,
                    update_simulation_hud,
                    sync_runtime_profile_hud,
                    update_runtime_profile_tooltip,
                    update_scale_bar,
                )
                    .chain()
                    .in_set(SimUpdateSet::Ui),
            )
            .add_systems(
                Update,
                draw_world_tool_hover_highlight.in_set(SimUpdateSet::Overlay),
            );
    }
}

#[derive(Component)]
struct SimulationHudFpsText;

#[derive(Component)]
struct SimulationHudStatsText;

#[derive(Component)]
struct RuntimeProfileHudRoot;

#[derive(Component, Clone, Copy)]
struct RuntimeProfileLaneBar {
    lane: RuntimeProfileLane,
}

#[derive(Component, Clone, Copy)]
struct RuntimeProfileSegmentNode {
    lane: RuntimeProfileLane,
    index: usize,
}

#[derive(Component)]
struct RuntimeProfileTooltip;

#[derive(Component)]
struct RuntimeProfileTooltipText;

#[derive(Resource, Debug, Default)]
struct RuntimeProfileHudState {
    rendered_sequence: u64,
}

#[derive(Component)]
struct ScaleBarLine;

#[derive(Component)]
struct ScaleBarLabelText;

#[derive(Resource, Debug, Default)]
struct FpsHudStats {
    frame_samples: VecDeque<FpsFrameSample>,
    window_elapsed: f32,
    avg_fps: f32,
    min_fps: f32,
}

#[derive(Clone, Copy, Debug)]
struct FpsFrameSample {
    dt: f32,
    fps: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WorldTool {
    WaterLiquid,
    StoneSolid,
    StoneGranular,
    SoilSolid,
    SoilGranular,
    SandSolid,
    SandGranular,
    GrassSolid,
    GrassGranular,
    Break,
    Delete,
}

impl WorldTool {
    fn label(self) -> &'static str {
        match self {
            Self::WaterLiquid => "Water",
            Self::StoneSolid => "Stone Solid",
            Self::StoneGranular => "Stone Granular",
            Self::SoilSolid => "Soil Solid",
            Self::SoilGranular => "Soil Granular",
            Self::SandSolid => "Sand Solid",
            Self::SandGranular => "Sand Granular",
            Self::GrassSolid => "Grass Solid",
            Self::GrassGranular => "Grass Granular",
            Self::Break => "Break",
            Self::Delete => "Delete",
        }
    }

    fn material(self) -> Option<ParticleMaterial> {
        match self {
            Self::WaterLiquid => Some(ParticleMaterial::WaterLiquid),
            Self::StoneSolid => Some(ParticleMaterial::StoneSolid),
            Self::StoneGranular => Some(ParticleMaterial::StoneGranular),
            Self::SoilSolid => Some(ParticleMaterial::SoilSolid),
            Self::SoilGranular => Some(ParticleMaterial::SoilGranular),
            Self::SandSolid => Some(ParticleMaterial::SandSolid),
            Self::SandGranular => Some(ParticleMaterial::SandGranular),
            Self::GrassSolid => Some(ParticleMaterial::GrassSolid),
            Self::GrassGranular => Some(ParticleMaterial::GrassGranular),
            Self::Break => None,
            Self::Delete => None,
        }
    }

    fn terrain_material(self) -> Option<TerrainMaterial> {
        match self {
            Self::StoneSolid => Some(TerrainMaterial::Stone),
            Self::SoilSolid => Some(TerrainMaterial::Soil),
            Self::SandSolid => Some(TerrainMaterial::Sand),
            Self::GrassSolid => Some(TerrainMaterial::Grass),
            _ => None,
        }
    }

    fn is_granular(self) -> bool {
        matches!(
            self,
            Self::StoneGranular
                | Self::SoilGranular
                | Self::SandGranular
                | Self::GrassGranular
        )
    }

    fn uses_cell_hover_highlight(self) -> bool {
        self.terrain_material().is_some() || matches!(self, Self::Break | Self::Delete)
    }
}

#[derive(Resource, Debug, Default)]
struct WorldInteractionState {
    selected_tool: Option<WorldTool>,
    last_drag_world: Option<Vec2>,
    last_water_spawn_cell: Option<IVec2>,
    last_granular_spawn_cell: Option<IVec2>,
    stone_stroke: StoneStrokeState,
}

#[derive(Debug, Default)]
struct StoneStrokeState {
    active: bool,
    generated_cells: HashSet<IVec2>,
    candidate_cells: HashSet<IVec2>,
}

#[derive(Component, Clone, Copy)]
struct WorldToolButton {
    tool: WorldTool,
}

#[derive(Component)]
struct WorldToolTooltip;

#[derive(Component)]
struct WorldToolTooltipText;

#[derive(Resource, Clone)]
struct WorldToolIconSet {
    water_liquid: Handle<Image>,
    stone_solid: Handle<Image>,
    stone_granular: Handle<Image>,
    soil_solid: Handle<Image>,
    soil_granular: Handle<Image>,
    sand_solid: Handle<Image>,
    sand_granular: Handle<Image>,
    grass_solid: Handle<Image>,
    grass_granular: Handle<Image>,
    break_icon: Handle<Image>,
    delete: Handle<Image>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SaveLoadDialogMode {
    Save,
    Load,
}

#[derive(Resource, Debug, Default)]
struct SaveLoadUiState {
    mode: Option<SaveLoadDialogMode>,
    slots: Vec<String>,
    scenario_slots: Vec<String>,
    selected_slot: Option<String>,
    selected_slot_source: Option<SaveLoadSlotSource>,
    input_name: String,
    input_focused: bool,
    ime_preedit: String,
    status_message: String,
    refresh_requested: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SaveLoadSlotSource {
    DefaultWorld,
    Save,
    TestCase,
}

#[derive(Component, Clone, Copy)]
struct SaveLoadOpenButton {
    mode: SaveLoadDialogMode,
}

#[derive(Component)]
struct SaveLoadResetButton;

#[derive(Component)]
struct SimPlayPauseButton;

#[derive(Component)]
struct SimPlayPauseButtonText;

#[derive(Component)]
struct SimStepButton;

#[derive(Component)]
struct SaveLoadDialogRoot;

#[derive(Component)]
struct SaveLoadDialogTitleText;

#[derive(Component)]
struct SaveLoadDialogNameInputButton;

#[derive(Component)]
struct SaveLoadDialogNameInputText;

#[derive(Component)]
struct SaveLoadDialogStatusText;

#[derive(Component)]
struct SaveLoadDialogSlotList;

#[derive(Component)]
struct SaveLoadDialogConfirmButton;

#[derive(Component)]
struct SaveLoadDialogConfirmButtonText;

#[derive(Component)]
struct SaveLoadDialogCancelButton;

#[derive(Component, Clone)]
struct SaveLoadSlotButton {
    slot_name: String,
    source: SaveLoadSlotSource,
}

#[derive(Component)]
struct TestAssertPanelRoot;

#[derive(Component)]
struct TestAssertTitleText;

#[derive(Component)]
struct TestAssertList;

use setup::setup_simulation_ui;

use input_handlers::{
    handle_save_load_dialog_buttons, handle_save_load_name_input_button_interaction,
    handle_save_load_open_button_interaction, handle_save_load_reset_button_interaction,
    handle_save_load_slot_button_interaction, handle_save_load_text_input,
    handle_sim_play_pause_button_interaction, handle_sim_step_button_interaction,
    handle_world_tool_button_interaction, sync_save_load_ime_state,
};

use ui_systems::{
    update_fps_hud_stats, update_save_load_dialog, update_save_load_name_input_button_visuals,
    update_save_load_open_button_visuals, update_save_load_reset_button_visuals,
    update_save_load_slot_button_visuals, update_runtime_profile_tooltip, update_scale_bar,
    update_sim_play_pause_button_label, update_sim_play_pause_button_visuals,
    update_sim_step_button_visuals, update_simulation_hud, update_test_assert_panel,
    update_world_tool_button_visuals, update_world_tool_tooltip, sync_runtime_profile_hud,
};

use world_edit::{draw_world_tool_hover_highlight, handle_world_interactions};

fn toggle_button_bg(enabled: bool, colors: &InterfaceColorParams) -> BackgroundColor {
    if enabled {
        colors.button_bg_on.to_color().into()
    } else {
        colors.button_bg_off.to_color().into()
    }
}

fn spawn_assertion_line(parent: &mut ChildSpawnerCommands, ok: bool, active: bool, detail: String) {
    parent
        .spawn((Node {
            column_gap: px(6.0),
            ..default()
        },))
        .with_children(|row| {
            let state_color = if !active {
                Color::srgba(0.55, 0.55, 0.55, 1.0)
            } else if ok {
                Color::srgba(0.45, 0.95, 0.55, 1.0)
            } else {
                Color::srgba(0.95, 0.35, 0.35, 1.0)
            };
            let detail_color = if active {
                Color::WHITE
            } else {
                Color::srgba(0.62, 0.62, 0.62, 1.0)
            };
            row.spawn((
                Text::new(if ok { "[OK]" } else { "[NG]" }),
                TextFont::from_font_size(12.0),
                TextColor(state_color),
            ));
            row.spawn((
                Text::new(detail),
                TextFont::from_font_size(12.0),
                TextColor(detail_color),
            ));
        });
}

fn select_world_tool(state: &mut WorldInteractionState, next_tool: Option<WorldTool>) {
    state.selected_tool = next_tool;
    state.last_drag_world = None;
    state.last_water_spawn_cell = None;
    state.last_granular_spawn_cell = None;
    state.stone_stroke = StoneStrokeState::default();
}

fn open_save_load_dialog(state: &mut SaveLoadUiState, mode: SaveLoadDialogMode) {
    state.mode = Some(mode);
    state.status_message.clear();
    state.refresh_requested = true;
    state.ime_preedit.clear();
    if matches!(mode, SaveLoadDialogMode::Save) {
        state.input_name.clear();
        state.input_focused = true;
    } else {
        state.input_focused = false;
    }
    state.selected_slot = None;
    state.selected_slot_source = None;
}

fn close_save_load_dialog(state: &mut SaveLoadUiState) {
    state.mode = None;
    state.input_focused = false;
    state.ime_preedit.clear();
    state.status_message.clear();
    state.selected_slot = None;
    state.selected_slot_source = None;
}

fn resolve_save_slot_name(state: &SaveLoadUiState) -> Option<String> {
    let typed = state.input_name.trim();
    if !typed.is_empty() {
        return Some(typed.to_string());
    }
    state.selected_slot.clone()
}

fn append_printable_text(dst: &mut String, text: &str) {
    for chr in text.chars() {
        if is_printable_char(chr) {
            dst.push(chr);
        }
    }
}

fn is_printable_char(chr: char) -> bool {
    let is_in_private_use_area = ('\u{e000}'..='\u{f8ff}').contains(&chr)
        || ('\u{f0000}'..='\u{ffffd}').contains(&chr)
        || ('\u{100000}'..='\u{10fffd}').contains(&chr);
    !is_in_private_use_area && !chr.is_ascii_control()
}

fn clear_children_recursive(
    commands: &mut Commands,
    entity: Entity,
    children_query: &Query<&Children>,
) {
    let Ok(children) = children_query.get(entity) else {
        return;
    };
    let children: Vec<Entity> = children.iter().collect();
    for child in children {
        clear_children_recursive(commands, child, children_query);
        commands.entity(child).despawn();
    }
}
