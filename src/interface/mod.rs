use std::collections::{HashMap, HashSet, VecDeque};

use bevy::asset::RenderAssetUsages;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::window::{Ime, PrimaryWindow};

use crate::physics::cell_to_world_center;
use crate::physics::material::terrain_fracture_particle;
use crate::physics::object::{
    OBJECT_SHAPE_ITERS, OBJECT_SHAPE_STIFFNESS_ALPHA, ObjectPhysicsField, ObjectWorld,
};
use crate::physics::particle::{
    PARTICLE_SPEED_LIMIT_MPS, ParticleMaterial, ParticleWorld, TERRAIN_BOUNDARY_RADIUS_M,
    WAKE_RADIUS,
};
use crate::physics::save_load;
use crate::physics::scenario::{
    count_solid_cells, default_scenario_names, default_scenario_spec_by_name,
    evaluate_scenario_state,
};
use crate::physics::state::{
    LoadDefaultWorldRequest, LoadMapRequest, PhysicsStepProfiler, ReplayLoadScenarioRequest,
    ReplayState, ResetSimulationRequest, SaveMapRequest, SimUpdateSet, SimulationState,
};
use crate::physics::terrain::{
    CELL_SIZE_M, CHUNK_SIZE_I32, TerrainCell, TerrainMaterial, TerrainWorld, WORLD_MAX_CHUNK_X,
    WORLD_MAX_CHUNK_Y, WORLD_MIN_CHUNK_X, WORLD_MIN_CHUNK_Y, world_to_cell,
};

const HUD_BG_COLOR: Color = Color::srgba(0.05, 0.06, 0.09, 0.82);
const BUTTON_BG_OFF: Color = Color::srgba(0.17, 0.18, 0.22, 0.95);
const BUTTON_BG_ON: Color = Color::srgba(0.16, 0.30, 0.46, 0.95);
const BUTTON_BG_HOVER: Color = Color::srgba(0.24, 0.25, 0.30, 0.98);
const BUTTON_BG_PRESS: Color = Color::srgba(0.38, 0.40, 0.48, 0.98);
const BUTTON_BORDER_OFF: Color = Color::srgba(0.08, 0.10, 0.14, 1.0);
const BUTTON_BORDER_ON: Color = Color::srgba(0.80, 0.92, 1.00, 1.0);
const TOOLBAR_BOTTOM_PX: f32 = 12.0;
const TOOLBAR_BG_COLOR: Color = Color::srgba(0.05, 0.06, 0.09, 0.88);
const TOOLBAR_ICON_SIZE_PX: u32 = 32;
const TOOLBAR_ICON_GRID_SIZE: usize = 8;
const TOOLBAR_ICON_DOT_PX: u32 = TOOLBAR_ICON_SIZE_PX / TOOLBAR_ICON_GRID_SIZE as u32;
const TOOLBAR_BUTTON_SIZE_PX: f32 = 44.0;
const TOOLTIP_BG_COLOR: Color = Color::srgba(0.04, 0.05, 0.08, 0.96);
const TOOLTIP_CURSOR_OFFSET_X: f32 = 14.0;
const TOOLTIP_CURSOR_OFFSET_Y: f32 = 20.0;
const TOOLTIP_GLOBAL_Z_INDEX: i32 = 10_000;
const SAVE_LOAD_BAR_TOP_PX: f32 = 10.0;
const SAVE_LOAD_BAR_RIGHT_PX: f32 = 10.0;
const SAVE_LOAD_BUTTON_WIDTH_PX: f32 = 84.0;
const SAVE_LOAD_BUTTON_HEIGHT_PX: f32 = 34.0;
const DIALOG_BG_COLOR: Color = Color::srgba(0.06, 0.07, 0.10, 0.97);
const DIALOG_WIDTH_PX: f32 = 380.0;
const DIALOG_NAME_INPUT_HEIGHT_PX: f32 = 34.0;
const DIALOG_SLOT_LIST_MAX_HEIGHT_PX: f32 = 240.0;
const DIALOG_SLOT_BUTTON_HEIGHT_PX: f32 = 30.0;
const TEST_ASSERT_PANEL_TOP_PX: f32 = 260.0;
const TEST_ASSERT_PANEL_RIGHT_PX: f32 = 10.0;
const TEST_ASSERT_PANEL_WIDTH_PX: f32 = 360.0;
const HUD_PANEL_WIDTH_PX: f32 = 360.0;
const STEP_PROFILER_BAR_HEIGHT_PX: f32 = 100.0;
const STEP_PROFILER_BAR_MS_TO_PX: f32 = 16.0;
const STEP_PROFILER_PARALLELISM_TO_PX: f32 = 8.0;
const STEP_PROFILER_MAX_PARALLELISM_DISPLAY: f32 = 12.0;
const STEP_PROFILER_TOOLTIP_OFFSET_X: f32 = 12.0;
const STEP_PROFILER_TOOLTIP_OFFSET_Y: f32 = 18.0;
const STEP_PROFILER_FLUID_COLORS: [Color; 3] = [
    Color::srgba(0.21, 0.66, 0.95, 0.95),
    Color::srgba(0.32, 0.74, 0.98, 0.95),
    Color::srgba(0.14, 0.58, 0.88, 0.95),
];
const STEP_PROFILER_GRANULAR_COLORS: [Color; 3] = [
    Color::srgba(0.96, 0.66, 0.24, 0.95),
    Color::srgba(0.93, 0.57, 0.18, 0.95),
    Color::srgba(0.98, 0.74, 0.30, 0.95),
];
const STEP_PROFILER_OBJECT_COLORS: [Color; 3] = [
    Color::srgba(0.42, 0.79, 0.41, 0.95),
    Color::srgba(0.33, 0.70, 0.33, 0.95),
    Color::srgba(0.52, 0.86, 0.51, 0.95),
];
const DRAG_VELOCITY_BRUSH_RADIUS_M: f32 = 0.55;
const DRAG_VELOCITY_GAIN: f32 = 0.9;
const TOOL_STROKE_STEP_M: f32 = CELL_SIZE_M * 0.5;
const TOOL_DELETE_BRUSH_RADIUS_M: f32 = CELL_SIZE_M * 0.5;
const TOOL_BREAK_BRUSH_RADIUS_M: f32 = CELL_SIZE_M * 0.5;
const STONE_STROKE_NEIGHBOR_OFFSETS: [IVec2; 4] = [
    IVec2::new(1, 0),
    IVec2::new(-1, 0),
    IVec2::new(0, 1),
    IVec2::new(0, -1),
];
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

#[allow(dead_code)]
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

const MATERIAL_PALETTE_WATER: [[u8; 4]; 4] = [
    [42, 120, 202, 235],
    [52, 136, 218, 240],
    [65, 152, 228, 245],
    [78, 167, 238, 250],
];

const MATERIAL_PALETTE_STONE: [[u8; 4]; 4] = [
    [70, 67, 63, 255],
    [83, 79, 74, 255],
    [95, 90, 84, 255],
    [108, 103, 96, 255],
];

const MATERIAL_PALETTE_SAND: [[u8; 4]; 4] = [
    [172, 149, 111, 255],
    [185, 162, 124, 255],
    [198, 175, 136, 255],
    [210, 188, 148, 255],
];

const MATERIAL_PALETTE_SOIL: [[u8; 4]; 4] = [
    [105, 79, 56, 255],
    [119, 91, 67, 255],
    [133, 103, 78, 255],
    [147, 115, 88, 255],
];

const WORLD_TOOLBAR_TOOLS: [WorldTool; 9] = [
    WorldTool::WaterLiquid,
    WorldTool::StoneSolid,
    WorldTool::StoneGranular,
    WorldTool::SoilSolid,
    WorldTool::SoilGranular,
    WorldTool::SandSolid,
    WorldTool::SandGranular,
    WorldTool::Break,
    WorldTool::Delete,
];

pub struct InterfacePlugin;

impl Plugin for InterfacePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldInteractionState>()
            .init_resource::<SaveLoadUiState>()
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
                    update_simulation_hud,
                    update_step_profiler_panel,
                    update_step_profiler_tooltip,
                )
                    .chain()
                    .in_set(SimUpdateSet::Ui),
            );
    }
}

#[derive(Component)]
struct SimulationHudFpsText;

#[derive(Component)]
struct SimulationHudStatsText;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WorldTool {
    WaterLiquid,
    StoneSolid,
    StoneGranular,
    SoilSolid,
    SoilGranular,
    SandSolid,
    SandGranular,
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
            Self::Break => None,
            Self::Delete => None,
        }
    }

    fn terrain_material(self) -> Option<TerrainMaterial> {
        match self {
            Self::StoneSolid => Some(TerrainMaterial::Stone),
            Self::SoilSolid => Some(TerrainMaterial::Soil),
            Self::SandSolid => Some(TerrainMaterial::Sand),
            _ => None,
        }
    }

    fn is_granular(self) -> bool {
        matches!(
            self,
            Self::StoneGranular | Self::SoilGranular | Self::SandGranular
        )
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

#[derive(Component)]
struct StepProfilerMsText;

#[derive(Component)]
struct StepProfilerBarTrack;

#[derive(Component, Clone)]
struct StepProfilerBarSegment {
    step_name: String,
    wall_duration_ms: f64,
    cpu_duration_ms: f64,
}

#[derive(Component)]
struct StepProfilerTooltip;

#[derive(Component)]
struct StepProfilerTooltipText;

fn setup_simulation_ui(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let icon_set = create_world_tool_icon_set(&mut images);
    commands.insert_resource(icon_set.clone());

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(10.0),
                top: px(10.0),
                width: px(HUD_PANEL_WIDTH_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                row_gap: px(6.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(HUD_BG_COLOR),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("FPS: --"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                SimulationHudFpsText,
            ));
            parent.spawn((
                Text::new("Physics Step: -- ms"),
                TextFont::from_font_size(13.0),
                TextColor(Color::WHITE),
                StepProfilerMsText,
            ));
            parent.spawn((
                Node {
                    width: percent(100.0),
                    height: px(STEP_PROFILER_BAR_HEIGHT_PX),
                    align_items: AlignItems::End,
                    overflow: Overflow::clip_x(),
                    border: UiRect::all(px(1.0)),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.10, 0.12, 0.16, 0.95)),
                BorderColor::all(BUTTON_BORDER_OFF),
                StepProfilerBarTrack,
            ));
            parent.spawn((
                Text::new(
                    "Sim: --\nWater(L): --\nStone(S): --\nStone(G): --\nSoil(S): --\nSoil(G): --\nSand(S): --\nSand(G): --",
                ),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                SimulationHudStatsText,
            ));
        });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                display: Display::None,
                left: px(0.0),
                top: px(0.0),
                padding: UiRect::axes(px(8.0), px(4.0)),
                ..default()
            },
            BackgroundColor(TOOLTIP_BG_COLOR),
            GlobalZIndex(TOOLTIP_GLOBAL_Z_INDEX),
            StepProfilerTooltip,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(13.0),
                TextColor(Color::WHITE),
                StepProfilerTooltipText,
            ));
        });

    commands
        .spawn((Node {
            position_type: PositionType::Absolute,
            right: px(SAVE_LOAD_BAR_RIGHT_PX),
            top: px(SAVE_LOAD_BAR_TOP_PX),
            column_gap: px(8.0),
            ..default()
        },))
        .with_children(|parent| {
            parent
                .spawn((
                    Button,
                    Node {
                        width: px(SAVE_LOAD_BUTTON_WIDTH_PX),
                        height: px(SAVE_LOAD_BUTTON_HEIGHT_PX),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(BUTTON_BG_OFF),
                    BorderColor::all(BUTTON_BORDER_OFF),
                    SimPlayPauseButton,
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new("Play"),
                        TextFont::from_font_size(14.0),
                        TextColor(Color::WHITE),
                        SimPlayPauseButtonText,
                    ));
                });

            parent
                .spawn((
                    Button,
                    Node {
                        width: px(SAVE_LOAD_BUTTON_WIDTH_PX),
                        height: px(SAVE_LOAD_BUTTON_HEIGHT_PX),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(BUTTON_BG_OFF),
                    BorderColor::all(BUTTON_BORDER_OFF),
                    SimStepButton,
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new("Step"),
                        TextFont::from_font_size(14.0),
                        TextColor(Color::WHITE),
                    ));
                });

            parent
                .spawn((
                    Button,
                    Node {
                        width: px(SAVE_LOAD_BUTTON_WIDTH_PX),
                        height: px(SAVE_LOAD_BUTTON_HEIGHT_PX),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(BUTTON_BG_OFF),
                    BorderColor::all(BUTTON_BORDER_OFF),
                    SaveLoadOpenButton {
                        mode: SaveLoadDialogMode::Save,
                    },
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new("Save"),
                        TextFont::from_font_size(14.0),
                        TextColor(Color::WHITE),
                    ));
                });

            parent
                .spawn((
                    Button,
                    Node {
                        width: px(SAVE_LOAD_BUTTON_WIDTH_PX),
                        height: px(SAVE_LOAD_BUTTON_HEIGHT_PX),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(BUTTON_BG_OFF),
                    BorderColor::all(BUTTON_BORDER_OFF),
                    SaveLoadOpenButton {
                        mode: SaveLoadDialogMode::Load,
                    },
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new("Load"),
                        TextFont::from_font_size(14.0),
                        TextColor(Color::WHITE),
                    ));
                });

            parent
                .spawn((
                    Button,
                    Node {
                        width: px(SAVE_LOAD_BUTTON_WIDTH_PX),
                        height: px(SAVE_LOAD_BUTTON_HEIGHT_PX),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(BUTTON_BG_OFF),
                    BorderColor::all(BUTTON_BORDER_OFF),
                    SaveLoadResetButton,
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new("Reset"),
                        TextFont::from_font_size(14.0),
                        TextColor(Color::WHITE),
                    ));
                });
        });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: px(TEST_ASSERT_PANEL_RIGHT_PX),
                top: px(TEST_ASSERT_PANEL_TOP_PX),
                width: px(TEST_ASSERT_PANEL_WIDTH_PX),
                display: Display::None,
                padding: UiRect::all(px(8.0)),
                row_gap: px(6.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(HUD_BG_COLOR),
            TestAssertPanelRoot,
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("Test Assertions"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                TestAssertTitleText,
            ));
            panel
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: px(2.0),
                        ..default()
                    },
                    TestAssertList,
                ))
                .with_children(|_| {});
        });

    commands
        .spawn((Node {
            position_type: PositionType::Absolute,
            left: px(0.0),
            right: px(0.0),
            bottom: px(TOOLBAR_BOTTOM_PX),
            justify_content: JustifyContent::Center,
            ..default()
        },))
        .with_children(|parent| {
            parent
                .spawn((
                    Node {
                        padding: UiRect::axes(px(10.0), px(8.0)),
                        column_gap: px(8.0),
                        ..default()
                    },
                    BackgroundColor(TOOLBAR_BG_COLOR),
                ))
                .with_children(|toolbar| {
                    for tool in WORLD_TOOLBAR_TOOLS {
                        toolbar
                            .spawn((
                                Button,
                                Node {
                                    width: px(TOOLBAR_BUTTON_SIZE_PX),
                                    height: px(TOOLBAR_BUTTON_SIZE_PX),
                                    align_items: AlignItems::Center,
                                    justify_content: JustifyContent::Center,
                                    border: UiRect::all(px(2.0)),
                                    ..default()
                                },
                                BackgroundColor(BUTTON_BG_OFF),
                                BorderColor::all(BUTTON_BORDER_OFF),
                                WorldToolButton { tool },
                            ))
                            .with_children(|button| {
                                button.spawn((
                                    ImageNode::new(icon_set.icon_for(tool)),
                                    Node {
                                        width: px(TOOLBAR_ICON_SIZE_PX as f32),
                                        height: px(TOOLBAR_ICON_SIZE_PX as f32),
                                        ..default()
                                    },
                                ));
                            });
                    }
                });
        });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                display: Display::None,
                left: px(0.0),
                top: px(0.0),
                padding: UiRect::axes(px(8.0), px(4.0)),
                ..default()
            },
            BackgroundColor(TOOLTIP_BG_COLOR),
            GlobalZIndex(TOOLTIP_GLOBAL_Z_INDEX),
            WorldToolTooltip,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(13.0),
                TextColor(Color::WHITE),
                WorldToolTooltipText,
            ));
        });

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(0.0),
                right: px(0.0),
                top: px(0.0),
                bottom: px(0.0),
                display: Display::None,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.38)),
            GlobalZIndex(TOOLTIP_GLOBAL_Z_INDEX - 1),
            SaveLoadDialogRoot,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Node {
                        width: px(DIALOG_WIDTH_PX),
                        padding: UiRect::all(px(12.0)),
                        row_gap: px(8.0),
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                    BackgroundColor(DIALOG_BG_COLOR),
                    BorderColor::all(BUTTON_BORDER_ON),
                ))
                .with_children(|dialog| {
                    dialog.spawn((
                        Text::new("Save"),
                        TextFont::from_font_size(18.0),
                        TextColor(Color::WHITE),
                        SaveLoadDialogTitleText,
                    ));
                    dialog.spawn((
                        Text::new("Save Name"),
                        TextFont::from_font_size(14.0),
                        TextColor(Color::srgba(0.9, 0.92, 0.97, 0.95)),
                    ));
                    dialog
                        .spawn((
                            Button,
                            Node {
                                width: percent(100.0),
                                height: px(DIALOG_NAME_INPUT_HEIGHT_PX),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::FlexStart,
                                padding: UiRect::horizontal(px(8.0)),
                                border: UiRect::all(px(2.0)),
                                ..default()
                            },
                            BackgroundColor(BUTTON_BG_OFF),
                            BorderColor::all(BUTTON_BORDER_OFF),
                            SaveLoadDialogNameInputButton,
                        ))
                        .with_children(|button| {
                            button.spawn((
                                Text::new(""),
                                TextFont::from_font_size(14.0),
                                TextColor(Color::WHITE),
                                SaveLoadDialogNameInputText,
                            ));
                        });
                    dialog.spawn((
                        Text::new(""),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::srgba(0.90, 0.78, 0.52, 1.0)),
                        SaveLoadDialogStatusText,
                    ));
                    dialog
                        .spawn((
                            Node {
                                max_height: px(DIALOG_SLOT_LIST_MAX_HEIGHT_PX),
                                flex_direction: FlexDirection::Column,
                                row_gap: px(4.0),
                                overflow: Overflow::clip_y(),
                                ..default()
                            },
                            SaveLoadDialogSlotList,
                        ))
                        .with_children(|_| {});
                    dialog
                        .spawn((Node {
                            justify_content: JustifyContent::FlexEnd,
                            column_gap: px(8.0),
                            ..default()
                        },))
                        .with_children(|actions| {
                            actions
                                .spawn((
                                    Button,
                                    Node {
                                        width: px(96.0),
                                        height: px(34.0),
                                        align_items: AlignItems::Center,
                                        justify_content: JustifyContent::Center,
                                        border: UiRect::all(px(2.0)),
                                        ..default()
                                    },
                                    BackgroundColor(BUTTON_BG_OFF),
                                    BorderColor::all(BUTTON_BORDER_OFF),
                                    SaveLoadDialogCancelButton,
                                ))
                                .with_children(|button| {
                                    button.spawn((
                                        Text::new("Cancel"),
                                        TextFont::from_font_size(14.0),
                                        TextColor(Color::WHITE),
                                    ));
                                });
                            actions
                                .spawn((
                                    Button,
                                    Node {
                                        width: px(96.0),
                                        height: px(34.0),
                                        align_items: AlignItems::Center,
                                        justify_content: JustifyContent::Center,
                                        border: UiRect::all(px(2.0)),
                                        ..default()
                                    },
                                    BackgroundColor(BUTTON_BG_OFF),
                                    BorderColor::all(BUTTON_BORDER_OFF),
                                    SaveLoadDialogConfirmButton,
                                ))
                                .with_children(|button| {
                                    button.spawn((
                                        Text::new("Save"),
                                        TextFont::from_font_size(14.0),
                                        TextColor(Color::WHITE),
                                        SaveLoadDialogConfirmButtonText,
                                    ));
                                });
                        });
                });
        });
}

fn handle_world_tool_button_interaction(
    mut interactions: Query<
        (&Interaction, &WorldToolButton),
        (Changed<Interaction>, With<WorldToolButton>),
    >,
    mut interaction_state: ResMut<WorldInteractionState>,
) {
    for (interaction, button) in &mut interactions {
        if *interaction == Interaction::Pressed {
            select_world_tool(&mut interaction_state, Some(button.tool));
        }
    }
}

fn handle_save_load_open_button_interaction(
    mut interactions: Query<
        (&Interaction, &SaveLoadOpenButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    for (interaction, button) in &mut interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }
        open_save_load_dialog(&mut save_load_ui_state, button.mode);
    }
}

fn handle_save_load_reset_button_interaction(
    mut interactions: Query<
        &Interaction,
        (
            Changed<Interaction>,
            With<SaveLoadResetButton>,
            With<Button>,
        ),
    >,
    mut reset_writer: MessageWriter<ResetSimulationRequest>,
) {
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            reset_writer.write(ResetSimulationRequest);
        }
    }
}

fn handle_sim_play_pause_button_interaction(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<SimPlayPauseButton>, With<Button>),
    >,
    mut sim_state: ResMut<SimulationState>,
) {
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            sim_state.running = !sim_state.running;
        }
    }
}

fn handle_sim_step_button_interaction(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<SimStepButton>, With<Button>),
    >,
    mut sim_state: ResMut<SimulationState>,
) {
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            sim_state.running = false;
            sim_state.step_once = true;
        }
    }
}

fn handle_save_load_name_input_button_interaction(
    mut interactions: Query<
        &Interaction,
        (Changed<Interaction>, With<SaveLoadDialogNameInputButton>),
    >,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    if save_load_ui_state.mode != Some(SaveLoadDialogMode::Save) {
        return;
    }
    for interaction in &mut interactions {
        if *interaction == Interaction::Pressed {
            save_load_ui_state.input_focused = true;
        }
    }
}

fn handle_save_load_dialog_buttons(
    mut confirm_buttons: Query<
        &Interaction,
        (Changed<Interaction>, With<SaveLoadDialogConfirmButton>),
    >,
    mut cancel_buttons: Query<
        &Interaction,
        (Changed<Interaction>, With<SaveLoadDialogCancelButton>),
    >,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
    mut save_writer: MessageWriter<SaveMapRequest>,
    mut load_writer: MessageWriter<LoadMapRequest>,
    mut replay_load_writer: MessageWriter<ReplayLoadScenarioRequest>,
    mut load_default_world_writer: MessageWriter<LoadDefaultWorldRequest>,
) {
    for interaction in &mut cancel_buttons {
        if *interaction == Interaction::Pressed {
            close_save_load_dialog(&mut save_load_ui_state);
            return;
        }
    }

    for interaction in &mut confirm_buttons {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match save_load_ui_state.mode {
            Some(SaveLoadDialogMode::Save) => {
                let slot_name = resolve_save_slot_name(&save_load_ui_state);
                let Some(slot_name) = slot_name else {
                    save_load_ui_state.status_message =
                        "Enter a save name or choose an existing slot".to_string();
                    return;
                };
                save_writer.write(SaveMapRequest { slot_name });
                close_save_load_dialog(&mut save_load_ui_state);
            }
            Some(SaveLoadDialogMode::Load) => {
                let Some(slot_name) = save_load_ui_state.selected_slot.clone() else {
                    save_load_ui_state.status_message = "Select a save slot to load".to_string();
                    return;
                };
                match save_load_ui_state.selected_slot_source {
                    Some(SaveLoadSlotSource::DefaultWorld) => {
                        load_default_world_writer.write(LoadDefaultWorldRequest);
                    }
                    Some(SaveLoadSlotSource::Save) | None => {
                        load_writer.write(LoadMapRequest { slot_name });
                    }
                    Some(SaveLoadSlotSource::TestCase) => {
                        replay_load_writer.write(ReplayLoadScenarioRequest {
                            scenario_name: slot_name,
                        });
                    }
                }
                close_save_load_dialog(&mut save_load_ui_state);
            }
            None => {}
        }
    }
}

fn handle_save_load_slot_button_interaction(
    mut interactions: Query<
        (&Interaction, &SaveLoadSlotButton),
        (Changed<Interaction>, With<Button>),
    >,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    for (interaction, button) in &mut interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }
        save_load_ui_state.selected_slot = Some(button.slot_name.clone());
        save_load_ui_state.selected_slot_source = Some(button.source);
        if matches!(save_load_ui_state.mode, Some(SaveLoadDialogMode::Save)) {
            if matches!(button.source, SaveLoadSlotSource::Save) {
                save_load_ui_state.input_name = button.slot_name.clone();
            }
            save_load_ui_state.input_focused = false;
            save_load_ui_state.ime_preedit.clear();
        } else {
            save_load_ui_state.input_focused = false;
        }
        save_load_ui_state.status_message.clear();
    }
}

fn handle_save_load_text_input(
    mut keyboard_input_reader: MessageReader<KeyboardInput>,
    mut ime_reader: MessageReader<Ime>,
    mut save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    let dialog_open = save_load_ui_state.mode.is_some();
    let save_mode_focused = save_load_ui_state.mode == Some(SaveLoadDialogMode::Save)
        && save_load_ui_state.input_focused;

    for ime in ime_reader.read() {
        if !save_mode_focused {
            continue;
        }
        match ime {
            Ime::Preedit { value, cursor, .. } if cursor.is_some() => {
                save_load_ui_state.ime_preedit = value.clone();
            }
            Ime::Preedit { cursor, .. } if cursor.is_none() => {
                save_load_ui_state.ime_preedit.clear();
            }
            Ime::Commit { value, .. } => {
                append_printable_text(&mut save_load_ui_state.input_name, value);
                save_load_ui_state.ime_preedit.clear();
            }
            _ => {}
        }
    }

    for keyboard_input in keyboard_input_reader.read() {
        if !keyboard_input.state.is_pressed() {
            continue;
        }
        if !dialog_open {
            continue;
        }
        if matches!(keyboard_input.logical_key, Key::Escape) {
            close_save_load_dialog(&mut save_load_ui_state);
            return;
        }
        if !save_mode_focused {
            continue;
        }
        match (&keyboard_input.logical_key, &keyboard_input.text) {
            (Key::Backspace, _) => {
                save_load_ui_state.input_name.pop();
                save_load_ui_state.ime_preedit.clear();
            }
            (_, Some(inserted_text)) => {
                append_printable_text(&mut save_load_ui_state.input_name, inserted_text);
            }
            _ => {}
        }
    }
}

fn sync_save_load_ime_state(
    save_load_ui_state: Res<SaveLoadUiState>,
    mut primary_window: Single<&mut Window, With<PrimaryWindow>>,
) {
    let enable_ime = save_load_ui_state.mode == Some(SaveLoadDialogMode::Save)
        && save_load_ui_state.input_focused;
    primary_window.ime_enabled = enable_ime;
    if enable_ime {
        if let Some(cursor) = primary_window.cursor_position() {
            primary_window.ime_position = cursor;
        }
    }
}

fn update_world_tool_button_visuals(
    interaction_state: Res<WorldInteractionState>,
    mut buttons: Query<(
        &Interaction,
        &WorldToolButton,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (interaction, button, mut bg, mut border_color) in &mut buttons {
        let selected = interaction_state.selected_tool == Some(button.tool);
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => toggle_button_bg(selected),
        };
        *border_color = if selected {
            BorderColor::all(BUTTON_BORDER_ON)
        } else {
            BorderColor::all(BUTTON_BORDER_OFF)
        };
    }
}

fn update_save_load_open_button_visuals(
    mut buttons: Query<
        (
            &Interaction,
            &mut BackgroundColor,
            &mut BorderColor,
            &SaveLoadOpenButton,
        ),
        With<Button>,
    >,
    save_load_ui_state: Res<SaveLoadUiState>,
) {
    for (interaction, mut bg, mut border_color, button) in &mut buttons {
        let selected = save_load_ui_state.mode == Some(button.mode);
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => toggle_button_bg(selected),
        };
        *border_color = if selected {
            BorderColor::all(BUTTON_BORDER_ON)
        } else {
            BorderColor::all(BUTTON_BORDER_OFF)
        };
    }
}

fn update_save_load_reset_button_visuals(
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        (With<Button>, With<SaveLoadResetButton>),
    >,
) {
    for (interaction, mut bg, mut border_color) in &mut buttons {
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => BUTTON_BG_OFF.into(),
        };
        *border_color = BorderColor::all(BUTTON_BORDER_OFF);
    }
}

fn update_sim_play_pause_button_visuals(
    sim_state: Res<SimulationState>,
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        (With<Button>, With<SimPlayPauseButton>),
    >,
) {
    for (interaction, mut bg, mut border_color) in &mut buttons {
        let selected = sim_state.running;
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => toggle_button_bg(selected),
        };
        *border_color = if selected {
            BorderColor::all(BUTTON_BORDER_ON)
        } else {
            BorderColor::all(BUTTON_BORDER_OFF)
        };
    }
}

fn update_sim_step_button_visuals(
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        (With<Button>, With<SimStepButton>),
    >,
) {
    for (interaction, mut bg, mut border_color) in &mut buttons {
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => BUTTON_BG_OFF.into(),
        };
        *border_color = BorderColor::all(BUTTON_BORDER_OFF);
    }
}

fn update_sim_play_pause_button_label(
    sim_state: Res<SimulationState>,
    mut labels: Query<&mut Text, With<SimPlayPauseButtonText>>,
) {
    if !sim_state.is_changed() {
        return;
    }
    for mut label in &mut labels {
        label.0 = if sim_state.running {
            "Pause".to_string()
        } else {
            "Play".to_string()
        };
    }
}

fn update_save_load_name_input_button_visuals(
    save_load_ui_state: Res<SaveLoadUiState>,
    mut inputs: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        With<SaveLoadDialogNameInputButton>,
    >,
) {
    for (interaction, mut bg, mut border_color) in &mut inputs {
        let selected = save_load_ui_state.input_focused;
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => toggle_button_bg(selected),
        };
        *border_color = if selected {
            BorderColor::all(BUTTON_BORDER_ON)
        } else {
            BorderColor::all(BUTTON_BORDER_OFF)
        };
    }
}

fn update_save_load_slot_button_visuals(
    save_load_ui_state: Res<SaveLoadUiState>,
    mut buttons: Query<
        (
            &Interaction,
            &SaveLoadSlotButton,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
) {
    for (interaction, button, mut bg, mut border_color) in &mut buttons {
        let selected = save_load_ui_state.selected_slot.as_deref()
            == Some(button.slot_name.as_str())
            && save_load_ui_state.selected_slot_source == Some(button.source);
        *bg = match *interaction {
            Interaction::Pressed => BUTTON_BG_PRESS.into(),
            Interaction::Hovered => BUTTON_BG_HOVER.into(),
            Interaction::None => toggle_button_bg(selected),
        };
        *border_color = if selected {
            BorderColor::all(BUTTON_BORDER_ON)
        } else {
            BorderColor::all(BUTTON_BORDER_OFF)
        };
    }
}

fn update_test_assert_panel(
    mut commands: Commands,
    replay_state: Res<ReplayState>,
    terrain: Res<TerrainWorld>,
    particles: Res<ParticleWorld>,
    objects: Res<ObjectWorld>,
    object_field: Res<ObjectPhysicsField>,
    mut panel_node: Single<&mut Node, With<TestAssertPanelRoot>>,
    title_entity: Single<Entity, With<TestAssertTitleText>>,
    list_entity: Single<Entity, With<TestAssertList>>,
    mut text_query: Query<&mut Text>,
    children_query: Query<&Children>,
) {
    if !replay_state.enabled {
        panel_node.display = Display::None;
        return;
    }
    let Some(scenario_name) = replay_state.scenario_name.as_deref() else {
        panel_node.display = Display::None;
        return;
    };
    let Some(spec) = default_scenario_spec_by_name(scenario_name) else {
        panel_node.display = Display::None;
        return;
    };
    panel_node.display = Display::Flex;

    let (metrics, assertions) = evaluate_scenario_state(
        &spec,
        replay_state.current_step,
        replay_state.baseline_particle_count,
        replay_state.baseline_solid_cell_count,
        &terrain,
        &particles,
        &objects,
        &object_field,
    );
    let overall_ok = assertions.iter().filter(|row| row.active).all(|row| row.ok);

    if let Ok(mut title_text) = text_query.get_mut(*title_entity) {
        title_text.0 = format!(
            "Test Assertions: {} | {}",
            spec.name,
            if overall_ok { "OK" } else { "NG" }
        );
    }

    clear_children_recursive(&mut commands, *list_entity, &children_query);
    commands.entity(*list_entity).with_children(|parent| {
        parent.spawn((
            Text::new(format!(
                "step: {} / {}",
                replay_state.current_step, spec.step_count
            )),
            TextFont::from_font_size(12.0),
            TextColor(Color::srgba(0.90, 0.92, 0.95, 0.95)),
        ));
        parent.spawn((
            Text::new(format!(
                "mass: particles {} / cells {}",
                metrics.particle_count,
                count_solid_cells(&terrain)
            )),
            TextFont::from_font_size(12.0),
            TextColor(Color::srgba(0.90, 0.92, 0.95, 0.95)),
        ));
        for row in assertions {
            spawn_assertion_line(
                parent,
                row.ok,
                row.active,
                format!(
                    "{} expected {} actual {}",
                    row.label, row.expected, row.actual
                ),
            );
        }
    });
}

fn update_save_load_dialog(
    mut commands: Commands,
    mut root_node: Single<&mut Node, With<SaveLoadDialogRoot>>,
    title_text_entity: Single<Entity, With<SaveLoadDialogTitleText>>,
    name_input_text_entity: Single<Entity, With<SaveLoadDialogNameInputText>>,
    status_text_entity: Single<Entity, With<SaveLoadDialogStatusText>>,
    confirm_text_entity: Single<Entity, With<SaveLoadDialogConfirmButtonText>>,
    slot_list_entity: Single<Entity, With<SaveLoadDialogSlotList>>,
    mut text_query: Query<&mut Text>,
    children_query: Query<&Children>,
    save_load_ui_state: ResMut<SaveLoadUiState>,
) {
    let mut save_load_ui_state = save_load_ui_state;
    let Some(mode) = save_load_ui_state.mode else {
        root_node.display = Display::None;
        return;
    };

    root_node.display = Display::Flex;
    let mode_label = match mode {
        SaveLoadDialogMode::Save => "Save",
        SaveLoadDialogMode::Load => "Load",
    };
    let name_label = match mode {
        SaveLoadDialogMode::Save => {
            let mut value = if save_load_ui_state.input_name.is_empty() {
                "Type save name".to_string()
            } else {
                save_load_ui_state.input_name.clone()
            };
            if !save_load_ui_state.ime_preedit.is_empty() {
                value.push_str(&save_load_ui_state.ime_preedit);
            }
            value
        }
        SaveLoadDialogMode::Load => {
            let selected = save_load_ui_state
                .selected_slot
                .as_deref()
                .unwrap_or("(none)");
            let source = match save_load_ui_state.selected_slot_source {
                Some(SaveLoadSlotSource::DefaultWorld) => "Default",
                Some(SaveLoadSlotSource::Save) => "Save",
                Some(SaveLoadSlotSource::TestCase) => "Test",
                None => "-",
            };
            format!("Selected: {selected} ({source})")
        }
    };
    let status_label = save_load_ui_state.status_message.clone();

    if let Ok(mut title_text) = text_query.get_mut(*title_text_entity) {
        title_text.0 = format!("{mode_label} World");
    }
    if let Ok(mut confirm_text) = text_query.get_mut(*confirm_text_entity) {
        confirm_text.0 = mode_label.to_string();
    }
    if let Ok(mut name_text) = text_query.get_mut(*name_input_text_entity) {
        name_text.0 = name_label;
    }
    if let Ok(mut status_text) = text_query.get_mut(*status_text_entity) {
        status_text.0 = status_label;
    }

    if save_load_ui_state.refresh_requested {
        match save_load::list_save_slots() {
            Ok(slots) => {
                save_load_ui_state.slots = slots;
                save_load_ui_state.scenario_slots = default_scenario_names();
                save_load_ui_state.status_message.clear();
            }
            Err(error) => {
                save_load_ui_state.slots.clear();
                save_load_ui_state.scenario_slots = default_scenario_names();
                save_load_ui_state.status_message = error;
            }
        }
        save_load_ui_state.refresh_requested = false;
    }

    clear_children_recursive(&mut commands, *slot_list_entity, &children_query);
    commands.entity(*slot_list_entity).with_children(|parent| {
        if save_load_ui_state.mode == Some(SaveLoadDialogMode::Load) {
            let default_world_label = "Default World".to_string();
            parent
                .spawn((
                    Button,
                    Node {
                        width: percent(100.0),
                        height: px(DIALOG_SLOT_BUTTON_HEIGHT_PX),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::FlexStart,
                        padding: UiRect::horizontal(px(8.0)),
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(BUTTON_BG_OFF),
                    BorderColor::all(BUTTON_BORDER_OFF),
                    SaveLoadSlotButton {
                        slot_name: default_world_label.clone(),
                        source: SaveLoadSlotSource::DefaultWorld,
                    },
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new(default_world_label),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::WHITE),
                    ));
                });

            parent.spawn((
                Text::new("Save Slots"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgba(0.90, 0.92, 0.97, 0.95)),
            ));
        }

        for slot_name in &save_load_ui_state.slots {
            parent
                .spawn((
                    Button,
                    Node {
                        width: percent(100.0),
                        height: px(DIALOG_SLOT_BUTTON_HEIGHT_PX),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::FlexStart,
                        padding: UiRect::horizontal(px(8.0)),
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(BUTTON_BG_OFF),
                    BorderColor::all(BUTTON_BORDER_OFF),
                    SaveLoadSlotButton {
                        slot_name: slot_name.clone(),
                        source: SaveLoadSlotSource::Save,
                    },
                ))
                .with_children(|button| {
                    button.spawn((
                        Text::new(slot_name.clone()),
                        TextFont::from_font_size(13.0),
                        TextColor(Color::WHITE),
                    ));
                });
        }

        if save_load_ui_state.mode == Some(SaveLoadDialogMode::Load) {
            parent.spawn((
                Text::new("Test Cases"),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgba(0.90, 0.92, 0.97, 0.95)),
            ));

            for scenario_name in &save_load_ui_state.scenario_slots {
                parent
                    .spawn((
                        Button,
                        Node {
                            width: percent(100.0),
                            height: px(DIALOG_SLOT_BUTTON_HEIGHT_PX),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::FlexStart,
                            padding: UiRect::horizontal(px(8.0)),
                            border: UiRect::all(px(2.0)),
                            ..default()
                        },
                        BackgroundColor(BUTTON_BG_OFF),
                        BorderColor::all(BUTTON_BORDER_OFF),
                        SaveLoadSlotButton {
                            slot_name: scenario_name.clone(),
                            source: SaveLoadSlotSource::TestCase,
                        },
                    ))
                    .with_children(|button| {
                        button.spawn((
                            Text::new(format!("[Test] {scenario_name}")),
                            TextFont::from_font_size(13.0),
                            TextColor(Color::WHITE),
                        ));
                    });
            }
        }
    });
}

fn update_world_tool_tooltip(
    windows: Query<&Window, With<PrimaryWindow>>,
    buttons: Query<(&Interaction, &WorldToolButton), With<Button>>,
    mut tooltip: Single<&mut Node, With<WorldToolTooltip>>,
    mut tooltip_text: Single<&mut Text, With<WorldToolTooltipText>>,
) {
    let hovered_tool = buttons.iter().find_map(|(interaction, button)| {
        (*interaction == Interaction::Hovered).then_some(button.tool)
    });
    let Some(tool) = hovered_tool else {
        tooltip.display = Display::None;
        return;
    };

    let Some(window) = windows.iter().next() else {
        tooltip.display = Display::None;
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        tooltip.display = Display::None;
        return;
    };

    tooltip.display = Display::Flex;
    tooltip.left = px(cursor.x + TOOLTIP_CURSOR_OFFSET_X);
    tooltip.top = px(cursor.y + TOOLTIP_CURSOR_OFFSET_Y);
    tooltip_text.0 = tool.label().to_string();
}

fn handle_world_interactions(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    ui_buttons: Query<&Interaction, With<Button>>,
    mut interaction_state: ResMut<WorldInteractionState>,
    save_load_ui_state: Res<SaveLoadUiState>,
    mut particle_world: ResMut<ParticleWorld>,
    mut terrain_world: ResMut<TerrainWorld>,
    mut object_world: ResMut<ObjectWorld>,
) {
    if save_load_ui_state.mode.is_some() {
        let selected_tool = interaction_state.selected_tool;
        finalize_stone_stroke(
            &mut interaction_state.stone_stroke,
            selected_tool,
            &mut particle_world,
            &mut object_world,
        );
        interaction_state.last_drag_world = None;
        interaction_state.last_water_spawn_cell = None;
        interaction_state.last_granular_spawn_cell = None;
        return;
    }

    if keyboard.just_pressed(KeyCode::Digit1) || keyboard.just_pressed(KeyCode::Numpad1) {
        select_world_tool(&mut interaction_state, Some(WorldTool::WaterLiquid));
    } else if keyboard.just_pressed(KeyCode::Digit2) || keyboard.just_pressed(KeyCode::Numpad2) {
        select_world_tool(&mut interaction_state, Some(WorldTool::StoneSolid));
    } else if keyboard.just_pressed(KeyCode::Digit3) || keyboard.just_pressed(KeyCode::Numpad3) {
        select_world_tool(&mut interaction_state, Some(WorldTool::StoneGranular));
    } else if keyboard.just_pressed(KeyCode::Digit4) || keyboard.just_pressed(KeyCode::Numpad4) {
        select_world_tool(&mut interaction_state, Some(WorldTool::SoilSolid));
    } else if keyboard.just_pressed(KeyCode::Digit5) || keyboard.just_pressed(KeyCode::Numpad5) {
        select_world_tool(&mut interaction_state, Some(WorldTool::SoilGranular));
    } else if keyboard.just_pressed(KeyCode::Digit6) || keyboard.just_pressed(KeyCode::Numpad6) {
        select_world_tool(&mut interaction_state, Some(WorldTool::SandSolid));
    } else if keyboard.just_pressed(KeyCode::Digit7) || keyboard.just_pressed(KeyCode::Numpad7) {
        select_world_tool(&mut interaction_state, Some(WorldTool::SandGranular));
    } else if keyboard.just_pressed(KeyCode::Digit8) || keyboard.just_pressed(KeyCode::Numpad8) {
        select_world_tool(&mut interaction_state, Some(WorldTool::Break));
    } else if keyboard.just_pressed(KeyCode::Digit9) || keyboard.just_pressed(KeyCode::Numpad9) {
        select_world_tool(&mut interaction_state, Some(WorldTool::Delete));
    }

    if keyboard.just_pressed(KeyCode::Escape) {
        select_world_tool(&mut interaction_state, None);
    }

    let left_pressed = mouse_buttons.pressed(MouseButton::Left);
    let alt_pressed = keyboard.pressed(KeyCode::AltLeft) || keyboard.pressed(KeyCode::AltRight);
    let blocked_by_pan = alt_pressed || mouse_buttons.pressed(MouseButton::Middle);
    let blocked_by_ui = ui_buttons
        .iter()
        .any(|interaction| *interaction != Interaction::None);
    let cursor_world = cursor_world_position(&windows, &camera_query);
    let selected_tool = interaction_state.selected_tool;

    if !left_pressed || blocked_by_pan || blocked_by_ui || cursor_world.is_none() {
        finalize_stone_stroke(
            &mut interaction_state.stone_stroke,
            selected_tool,
            &mut particle_world,
            &mut object_world,
        );
        interaction_state.last_drag_world = None;
        interaction_state.last_water_spawn_cell = None;
        interaction_state.last_granular_spawn_cell = None;
        return;
    }

    let cursor_world = cursor_world.unwrap_or(Vec2::ZERO);

    let previous_world = interaction_state.last_drag_world.unwrap_or(cursor_world);
    let dt = time.delta_secs().max(1e-4);
    let stroke_velocity = (cursor_world - previous_world) / dt;
    let mut terrain_changed = false;

    match selected_tool {
        Some(WorldTool::WaterLiquid) => {
            interaction_state.last_granular_spawn_cell = None;
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                selected_tool,
                &mut particle_world,
                &mut object_world,
            );
            let mut spawn_cells = Vec::new();
            let mut last_cell = interaction_state.last_water_spawn_cell;
            stroke_cells(previous_world, cursor_world, TOOL_STROKE_STEP_M, |cell| {
                if cell_in_fixed_world(cell) && last_cell != Some(cell) {
                    spawn_cells.push(cell);
                    last_cell = Some(cell);
                }
                particle_world.wake_particles_in_radius(cell_to_world_center(cell), WAKE_RADIUS);
            });
            interaction_state.last_water_spawn_cell = last_cell;
            if !spawn_cells.is_empty() {
                let _ = particle_world.spawn_material_particles_from_cells(
                    &spawn_cells,
                    ParticleMaterial::WaterLiquid,
                    Vec2::ZERO,
                );
            }
        }
        Some(tool) if tool.terrain_material().is_some() => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.last_granular_spawn_cell = None;
            let terrain_material = tool.terrain_material().unwrap_or(TerrainMaterial::Stone);
            interaction_state.stone_stroke.active = true;
            stroke_cells(previous_world, cursor_world, TOOL_STROKE_STEP_M, |cell| {
                if cell_in_fixed_world(cell) {
                    interaction_state.stone_stroke.generated_cells.insert(cell);
                }
                particle_world.wake_particles_in_radius(cell_to_world_center(cell), WAKE_RADIUS);
            });
            terrain_changed |= update_stone_stroke_partition(
                &mut interaction_state.stone_stroke,
                &mut terrain_world,
                terrain_material,
            );
        }
        Some(tool) if tool.is_granular() => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.stone_stroke.active = true;
            if let Some(material) = tool.material() {
                let mut spawn_cells = Vec::new();
                let mut last_cell = interaction_state.last_granular_spawn_cell;
                stroke_cells(previous_world, cursor_world, TOOL_STROKE_STEP_M, |cell| {
                    if cell_in_fixed_world(cell) && last_cell != Some(cell) {
                        spawn_cells.push(cell);
                        last_cell = Some(cell);
                    }
                    particle_world
                        .wake_particles_in_radius(cell_to_world_center(cell), WAKE_RADIUS);
                });
                interaction_state.last_granular_spawn_cell = last_cell;
                if !spawn_cells.is_empty() {
                    let _ = particle_world.spawn_material_particles_from_cells(
                        &spawn_cells,
                        material,
                        Vec2::ZERO,
                    );
                }
            }
        }
        Some(WorldTool::Delete) => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.last_granular_spawn_cell = None;
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                selected_tool,
                &mut particle_world,
                &mut object_world,
            );
            let mut had_particle_removal = false;
            let mut removed_terrain_cells = HashSet::new();
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                let removal = particle_world
                    .remove_particles_in_radius_with_map(point, TOOL_DELETE_BRUSH_RADIUS_M);
                if removal.removed_count > 0 {
                    had_particle_removal = true;
                    object_world.apply_particle_remap(&removal.old_to_new, particle_world.masses());
                }
                removed_terrain_cells.extend(remove_terrain_cells_in_radius(
                    &mut terrain_world,
                    point,
                    TOOL_DELETE_BRUSH_RADIUS_M,
                ));
                particle_world.wake_particles_in_radius(point, WAKE_RADIUS);
            });
            terrain_changed |= !removed_terrain_cells.is_empty();
            if !removed_terrain_cells.is_empty() {
                terrain_changed |= particle_world.detach_terrain_components_after_cell_removal(
                    &mut terrain_world,
                    &mut object_world,
                    &removed_terrain_cells,
                );
            }
            if had_particle_removal {
                particle_world.postprocess_objects_after_topology_edit(&mut object_world);
            }
        }
        Some(WorldTool::Break) => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.last_granular_spawn_cell = None;
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                selected_tool,
                &mut particle_world,
                &mut object_world,
            );
            let mut detached_particles = HashSet::new();
            let mut removed_terrain_cells = HashSet::new();
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                let fractured = particle_world
                    .fracture_solid_particles_in_radius(point, TOOL_BREAK_BRUSH_RADIUS_M);
                detached_particles.extend(fractured);
                removed_terrain_cells.extend(break_terrain_solids_in_radius(
                    &mut terrain_world,
                    &mut particle_world,
                    point,
                    TOOL_BREAK_BRUSH_RADIUS_M,
                ));
                particle_world.wake_particles_in_radius(point, WAKE_RADIUS);
            });
            terrain_changed |= !removed_terrain_cells.is_empty();
            if !detached_particles.is_empty() {
                particle_world
                    .detach_and_postprocess_objects(&mut object_world, &detached_particles);
            }
            if !removed_terrain_cells.is_empty() {
                terrain_changed |= particle_world.detach_terrain_components_after_cell_removal(
                    &mut terrain_world,
                    &mut object_world,
                    &removed_terrain_cells,
                );
            }
        }
        Some(_) => {}
        None => {
            interaction_state.last_water_spawn_cell = None;
            interaction_state.last_granular_spawn_cell = None;
            finalize_stone_stroke(
                &mut interaction_state.stone_stroke,
                selected_tool,
                &mut particle_world,
                &mut object_world,
            );
            let velocity_delta = stroke_velocity * DRAG_VELOCITY_GAIN;
            stroke_points(previous_world, cursor_world, TOOL_STROKE_STEP_M, |point| {
                particle_world.add_velocity_in_radius(
                    point,
                    DRAG_VELOCITY_BRUSH_RADIUS_M,
                    velocity_delta,
                    PARTICLE_SPEED_LIMIT_MPS,
                );
            });
        }
    }

    if terrain_changed {
        terrain_world.rebuild_static_particles_if_dirty(TERRAIN_BOUNDARY_RADIUS_M);
    }

    interaction_state.last_drag_world = Some(cursor_world);
}

fn update_stone_stroke_partition(
    stroke: &mut StoneStrokeState,
    terrain_world: &mut TerrainWorld,
    terrain_material: TerrainMaterial,
) -> bool {
    if stroke.generated_cells.is_empty() {
        stroke.candidate_cells.clear();
        return false;
    }

    let terrain_connected = collect_terrain_connected_stroke_cells(stroke, terrain_world);
    let mut terrain_changed = false;
    for &cell in &terrain_connected {
        terrain_changed |= terrain_world.set_cell(cell, TerrainCell::solid(terrain_material));
    }

    stroke.candidate_cells.clear();
    for &cell in &stroke.generated_cells {
        if !terrain_connected.contains(&cell) {
            stroke.candidate_cells.insert(cell);
        }
    }
    terrain_changed
}

fn collect_terrain_connected_stroke_cells(
    stroke: &StoneStrokeState,
    terrain_world: &TerrainWorld,
) -> HashSet<IVec2> {
    let mut connected = HashSet::new();
    let mut queue = VecDeque::new();

    for &cell in &stroke.generated_cells {
        if is_stroke_cell_connected_to_frozen_terrain(cell, stroke, terrain_world) {
            connected.insert(cell);
            queue.push_back(cell);
        }
    }

    while let Some(cell) = queue.pop_front() {
        for offset in STONE_STROKE_NEIGHBOR_OFFSETS {
            let next = cell + offset;
            if !stroke.generated_cells.contains(&next) || connected.contains(&next) {
                continue;
            }
            connected.insert(next);
            queue.push_back(next);
        }
    }

    connected
}

fn is_stroke_cell_connected_to_frozen_terrain(
    cell: IVec2,
    stroke: &StoneStrokeState,
    terrain_world: &TerrainWorld,
) -> bool {
    if matches!(
        terrain_world.get_loaded_cell_or_empty(cell),
        TerrainCell::Solid { .. }
    ) {
        return true;
    }

    for offset in STONE_STROKE_NEIGHBOR_OFFSETS {
        let neighbor = cell + offset;
        if stroke.generated_cells.contains(&neighbor) {
            continue;
        }
        if matches!(
            terrain_world.get_loaded_cell_or_empty(neighbor),
            TerrainCell::Solid { .. }
        ) {
            return true;
        }
    }
    false
}

fn finalize_stone_stroke(
    stroke: &mut StoneStrokeState,
    selected_tool: Option<WorldTool>,
    particle_world: &mut ParticleWorld,
    object_world: &mut ObjectWorld,
) {
    if !stroke.active {
        stroke.generated_cells.clear();
        stroke.candidate_cells.clear();
        return;
    }

    if !stroke.candidate_cells.is_empty() {
        let mut cells: Vec<_> = stroke.candidate_cells.iter().copied().collect();
        cells.sort_by_key(|cell| (cell.y, cell.x));
        if let Some(tool) = selected_tool {
            let Some(material) = tool.material() else {
                stroke.active = false;
                stroke.generated_cells.clear();
                stroke.candidate_cells.clear();
                return;
            };
            let particle_indices =
                particle_world.spawn_material_particles_from_cells(&cells, material, Vec2::ZERO);

            if tool.terrain_material().is_some() {
                let _ = object_world.create_object(
                    particle_indices,
                    particle_world.positions(),
                    particle_world.masses(),
                    OBJECT_SHAPE_STIFFNESS_ALPHA,
                    OBJECT_SHAPE_ITERS,
                );
                particle_world.postprocess_objects_after_topology_edit(object_world);
            }
        }
    }

    stroke.active = false;
    stroke.generated_cells.clear();
    stroke.candidate_cells.clear();
}

fn update_simulation_hud(
    diagnostics: Res<DiagnosticsStore>,
    sim_state: Res<SimulationState>,
    particles: Res<ParticleWorld>,
    mut hud_texts: ParamSet<(
        Single<&mut Text, With<SimulationHudFpsText>>,
        Single<&mut Text, With<SimulationHudStatsText>>,
    )>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|diag| diag.smoothed())
        .unwrap_or(0.0);
    hud_texts.p0().0 = format!("FPS: {fps:.1}");

    let sim_status = if sim_state.running { "Running" } else { "Paused" };
    let water_count = particles
        .materials()
        .iter()
        .filter(|&&m| matches!(m, ParticleMaterial::WaterLiquid))
        .count();
    let stone_solid_count = particles
        .materials()
        .iter()
        .filter(|&&m| matches!(m, ParticleMaterial::StoneSolid))
        .count();
    let stone_granular_count = particles
        .materials()
        .iter()
        .filter(|&&m| matches!(m, ParticleMaterial::StoneGranular))
        .count();
    let soil_solid_count = particles
        .materials()
        .iter()
        .filter(|&&m| matches!(m, ParticleMaterial::SoilSolid))
        .count();
    let soil_granular_count = particles
        .materials()
        .iter()
        .filter(|&&m| matches!(m, ParticleMaterial::SoilGranular))
        .count();
    let sand_solid_count = particles
        .materials()
        .iter()
        .filter(|&&m| matches!(m, ParticleMaterial::SandSolid))
        .count();
    let sand_granular_count = particles
        .materials()
        .iter()
        .filter(|&&m| matches!(m, ParticleMaterial::SandGranular))
        .count();
    hud_texts.p1().0 = format!(
        "Sim: {sim_status}\nWater(L): {water_count}\nStone(S): {stone_solid_count}\nStone(G): {stone_granular_count}\nSoil(S): {soil_solid_count}\nSoil(G): {soil_granular_count}\nSand(S): {sand_solid_count}\nSand(G): {sand_granular_count}"
    );
}

fn update_step_profiler_panel(
    mut commands: Commands,
    profiler: Res<PhysicsStepProfiler>,
    mut label_text: Single<&mut Text, With<StepProfilerMsText>>,
    bar_track_entity: Single<Entity, With<StepProfilerBarTrack>>,
    children_query: Query<&Children>,
) {
    if !profiler.is_changed() {
        return;
    }
    if profiler.total_duration_ms > 0.0 {
        label_text.0 = format!("Physics Step: {:.2} ms", profiler.total_duration_ms);
    } else {
        label_text.0 = "Physics Step: -- ms".to_string();
    }

    clear_children_recursive(&mut commands, *bar_track_entity, &children_query);
    let mut fluid_index = 0usize;
    let mut granular_index = 0usize;
    let mut object_index = 0usize;
    commands.entity(*bar_track_entity).with_children(|parent| {
        for segment in &profiler.segments {
            if segment.wall_duration_ms <= 0.0 {
                continue;
            }
            let category = classify_profiler_segment(&segment.name);
            let color = match category {
                StepProfilerCategory::Fluid => {
                    let color = STEP_PROFILER_FLUID_COLORS[fluid_index % STEP_PROFILER_FLUID_COLORS.len()];
                    fluid_index += 1;
                    color
                }
                StepProfilerCategory::Granular => {
                    let color = STEP_PROFILER_GRANULAR_COLORS
                        [granular_index % STEP_PROFILER_GRANULAR_COLORS.len()];
                    granular_index += 1;
                    color
                }
                StepProfilerCategory::Object => {
                    let color = STEP_PROFILER_OBJECT_COLORS[object_index % STEP_PROFILER_OBJECT_COLORS.len()];
                    object_index += 1;
                    color
                }
            };
            parent.spawn((
                Button,
                Node {
                    width: px((segment.wall_duration_ms as f32 * STEP_PROFILER_BAR_MS_TO_PX).max(1.0)),
                    height: px(
                        ((segment.cpu_duration_ms as f32
                            / segment.wall_duration_ms.max(1e-6) as f32)
                            .clamp(0.05, STEP_PROFILER_MAX_PARALLELISM_DISPLAY))
                            * STEP_PROFILER_PARALLELISM_TO_PX,
                    ),
                    ..default()
                },
                BackgroundColor(color),
                StepProfilerBarSegment {
                    step_name: segment.name.clone(),
                    wall_duration_ms: segment.wall_duration_ms,
                    cpu_duration_ms: segment.cpu_duration_ms,
                },
            ));
        }
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepProfilerCategory {
    Fluid,
    Granular,
    Object,
}

fn classify_profiler_segment(step_name: &str) -> StepProfilerCategory {
    if step_name.starts_with("particle_step::granular_")
        || step_name.starts_with("particle_step::contact_velocity_response")
    {
        return StepProfilerCategory::Granular;
    }
    if step_name.starts_with("object_field_")
        || step_name.starts_with("terrain_")
        || step_name.starts_with("particle_step::shape_")
        || step_name.starts_with("particle_step::apply_object_reaction")
        || step_name.starts_with("particle_step::fracture_")
        || step_name == "step_overhead"
    {
        return StepProfilerCategory::Object;
    }
    StepProfilerCategory::Fluid
}

fn update_step_profiler_tooltip(
    window: Single<&Window, With<PrimaryWindow>>,
    hovered: Query<(&Interaction, &StepProfilerBarSegment), With<Button>>,
    mut tooltip_node: Single<&mut Node, With<StepProfilerTooltip>>,
    mut tooltip_text: Single<&mut Text, With<StepProfilerTooltipText>>,
) {
    let hovered_segment = hovered
        .iter()
        .find(|(interaction, _)| **interaction == Interaction::Hovered)
        .map(|(_, segment)| segment.clone());
    let Some(segment) = hovered_segment else {
        tooltip_node.display = Display::None;
        return;
    };

    let Some(cursor) = window.cursor_position() else {
        tooltip_node.display = Display::None;
        return;
    };
    tooltip_node.display = Display::Flex;
    tooltip_node.left = px(cursor.x + STEP_PROFILER_TOOLTIP_OFFSET_X);
    tooltip_node.top = px(cursor.y + STEP_PROFILER_TOOLTIP_OFFSET_Y);
    let parallelism = segment.cpu_duration_ms / segment.wall_duration_ms.max(1e-6);
    tooltip_text.0 = format!(
        "{}\nwall: {:.2} ms\ncpu: {:.2} ms\ncpu/wall: {:.2}",
        segment.step_name, segment.wall_duration_ms, segment.cpu_duration_ms, parallelism
    );
}

fn cursor_world_position(
    windows: &Query<&Window, With<PrimaryWindow>>,
    cameras: &Query<(&Camera, &GlobalTransform), With<Camera2d>>,
) -> Option<Vec2> {
    let window = windows.iter().next()?;
    let cursor = window.cursor_position()?;
    let (camera, camera_transform) = cameras.iter().next()?;
    camera.viewport_to_world_2d(camera_transform, cursor).ok()
}

fn stroke_points(from: Vec2, to: Vec2, spacing: f32, mut paint: impl FnMut(Vec2)) {
    if spacing <= 0.0 {
        paint(to);
        return;
    }

    let delta = to - from;
    let distance = delta.length();
    if distance <= spacing {
        paint(to);
        return;
    }

    let steps = (distance / spacing).ceil() as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        paint(from.lerp(to, t));
    }
}

fn stroke_cells(from: Vec2, to: Vec2, spacing: f32, mut paint_cell: impl FnMut(IVec2)) {
    let mut previous_cell = None;
    stroke_points(from, to, spacing, |point| {
        let current_cell = world_to_cell(point);
        if let Some(prev_cell) = previous_cell {
            append_4_connected_segment(prev_cell, current_cell, |cell| paint_cell(cell));
        } else {
            paint_cell(current_cell);
        }
        previous_cell = Some(current_cell);
    });
}

fn append_4_connected_segment(from: IVec2, to: IVec2, mut emit: impl FnMut(IVec2)) {
    let mut current = from;
    emit(current);
    while current != to {
        let dx = to.x - current.x;
        let dy = to.y - current.y;
        if dx.abs() >= dy.abs() {
            if dx != 0 {
                current.x += dx.signum();
                emit(current);
            }
            if dy != 0 {
                current.y += dy.signum();
                emit(current);
            }
        } else {
            if dy != 0 {
                current.y += dy.signum();
                emit(current);
            }
            if dx != 0 {
                current.x += dx.signum();
                emit(current);
            }
        }
    }
}

fn remove_terrain_cells_in_radius(
    terrain_world: &mut TerrainWorld,
    center: Vec2,
    radius: f32,
) -> HashSet<IVec2> {
    let min_cell = world_to_cell(center - Vec2::splat(radius));
    let max_cell = world_to_cell(center + Vec2::splat(radius));
    let radius2 = radius * radius;
    let mut removed = HashSet::new();

    for y in min_cell.y..=max_cell.y {
        for x in min_cell.x..=max_cell.x {
            let cell = IVec2::new(x, y);
            if !cell_in_fixed_world(cell) {
                continue;
            }
            if cell_to_world_center(cell).distance_squared(center) > radius2 {
                continue;
            }
            if terrain_world.set_cell(cell, TerrainCell::Empty) {
                removed.insert(cell);
            }
        }
    }

    removed
}

fn break_terrain_solids_in_radius(
    terrain_world: &mut TerrainWorld,
    particle_world: &mut ParticleWorld,
    center: Vec2,
    radius: f32,
) -> HashSet<IVec2> {
    let min_cell = world_to_cell(center - Vec2::splat(radius));
    let max_cell = world_to_cell(center + Vec2::splat(radius));
    let radius2 = radius * radius;
    let mut removed = HashSet::new();
    let mut spawn_cells_by_material = HashMap::<ParticleMaterial, Vec<IVec2>>::new();

    for y in min_cell.y..=max_cell.y {
        for x in min_cell.x..=max_cell.x {
            let cell = IVec2::new(x, y);
            if !cell_in_fixed_world(cell) {
                continue;
            }
            if cell_to_world_center(cell).distance_squared(center) > radius2 {
                continue;
            }
            let TerrainCell::Solid { material, .. } = terrain_world.get_loaded_cell_or_empty(cell)
            else {
                continue;
            };
            let Some(target_particle) = terrain_fracture_particle(material) else {
                continue;
            };
            if terrain_world.set_cell(cell, TerrainCell::Empty) {
                removed.insert(cell);
                spawn_cells_by_material
                    .entry(target_particle)
                    .or_default()
                    .push(cell);
            }
        }
    }

    for (material, cells) in spawn_cells_by_material {
        let _ = particle_world.spawn_material_particles_from_cells(&cells, material, Vec2::ZERO);
    }
    removed
}

fn cell_in_fixed_world(cell: IVec2) -> bool {
    let min_cell_x = WORLD_MIN_CHUNK_X * CHUNK_SIZE_I32;
    let max_cell_x = (WORLD_MAX_CHUNK_X + 1) * CHUNK_SIZE_I32 - 1;
    let min_cell_y = WORLD_MIN_CHUNK_Y * CHUNK_SIZE_I32;
    let max_cell_y = (WORLD_MAX_CHUNK_Y + 1) * CHUNK_SIZE_I32 - 1;

    (min_cell_x..=max_cell_x).contains(&cell.x) && (min_cell_y..=max_cell_y).contains(&cell.y)
}

fn toggle_button_bg(enabled: bool) -> BackgroundColor {
    if enabled {
        BUTTON_BG_ON.into()
    } else {
        BUTTON_BG_OFF.into()
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

impl WorldToolIconSet {
    fn icon_for(&self, tool: WorldTool) -> Handle<Image> {
        match tool {
            WorldTool::WaterLiquid => self.water_liquid.clone(),
            WorldTool::StoneSolid => self.stone_solid.clone(),
            WorldTool::StoneGranular => self.stone_granular.clone(),
            WorldTool::SoilSolid => self.soil_solid.clone(),
            WorldTool::SoilGranular => self.soil_granular.clone(),
            WorldTool::SandSolid => self.sand_solid.clone(),
            WorldTool::SandGranular => self.sand_granular.clone(),
            WorldTool::Break => self.break_icon.clone(),
            WorldTool::Delete => self.delete.clone(),
        }
    }
}

fn create_world_tool_icon_set(images: &mut Assets<Image>) -> WorldToolIconSet {
    let water_liquid = images.add(build_material_icon_image(
        MATERIAL_PALETTE_WATER,
        &MATERIAL_PATTERN_LIQUID,
        0x4e23_1f91,
    ));
    let stone_solid = images.add(build_material_icon_image(
        MATERIAL_PALETTE_STONE,
        &MATERIAL_PATTERN_SOLID,
        0x8a52_d9b7,
    ));
    let stone_granular = images.add(build_material_icon_image(
        MATERIAL_PALETTE_STONE,
        &MATERIAL_PATTERN_GRANULAR,
        0x77bc_26f1,
    ));
    let soil_solid = images.add(build_material_icon_image(
        MATERIAL_PALETTE_SOIL,
        &MATERIAL_PATTERN_SOLID,
        0x68a2_1bd4,
    ));
    let soil_granular = images.add(build_material_icon_image(
        MATERIAL_PALETTE_SOIL,
        &MATERIAL_PATTERN_GRANULAR,
        0xb86a_c921,
    ));
    let sand_solid = images.add(build_material_icon_image(
        MATERIAL_PALETTE_SAND,
        &MATERIAL_PATTERN_SOLID,
        0x2f9a_43ce,
    ));
    let sand_granular = images.add(build_material_icon_image(
        MATERIAL_PALETTE_SAND,
        &MATERIAL_PATTERN_GRANULAR,
        0x9133_257e,
    ));
    let break_icon = images.add(build_break_icon_image());
    let delete = images.add(build_delete_icon_image());
    WorldToolIconSet {
        water_liquid,
        stone_solid,
        stone_granular,
        soil_solid,
        soil_granular,
        sand_solid,
        sand_granular,
        break_icon,
        delete,
    }
}

fn build_material_icon_image(
    palette: [[u8; 4]; 4],
    pattern: &MaterialPattern8,
    seed: u32,
) -> Image {
    let width = TOOLBAR_ICON_SIZE_PX;
    let height = TOOLBAR_ICON_SIZE_PX;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for (gy, row) in pattern.iter().enumerate() {
        for (gx, enabled) in row.iter().copied().enumerate() {
            if !enabled {
                continue;
            }
            let palette_index = material_icon_palette_index(gx as u32, gy as u32, seed);
            let color = palette[palette_index];
            let start_x = gx as u32 * TOOLBAR_ICON_DOT_PX;
            let start_y = gy as u32 * TOOLBAR_ICON_DOT_PX;
            for py in start_y..(start_y + TOOLBAR_ICON_DOT_PX) {
                for px in start_x..(start_x + TOOLBAR_ICON_DOT_PX) {
                    if px >= width || py >= height {
                        continue;
                    }
                    let idx = ((py * width + px) * 4) as usize;
                    pixels[idx..idx + 4].copy_from_slice(&color);
                }
            }
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    image
}

fn material_icon_palette_index(x: u32, y: u32, seed: u32) -> usize {
    let mut state = x.wrapping_mul(0x45d9f3b);
    state ^= y.wrapping_mul(0x27d4eb2d);
    state ^= seed;
    state ^= state >> 16;
    state = state.wrapping_mul(0x7feb_352d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x846c_a68b);
    state ^= state >> 16;
    (state & 0b11) as usize
}

fn build_delete_icon_image() -> Image {
    let width = TOOLBAR_ICON_SIZE_PX;
    let height = TOOLBAR_ICON_SIZE_PX;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let red = [220u8, 52u8, 52u8, 255u8];
    let thickness = 3i32;
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let d1 = (x - y).abs();
            let d2 = ((width as i32 - 1 - x) - y).abs();
            if d1 <= thickness || d2 <= thickness {
                let idx = (((y as u32) * width + (x as u32)) * 4) as usize;
                pixels[idx..idx + 4].copy_from_slice(&red);
            }
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    image
}

fn build_break_icon_image() -> Image {
    let width = TOOLBAR_ICON_SIZE_PX;
    let height = TOOLBAR_ICON_SIZE_PX;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let amber = [246u8, 170u8, 38u8, 255u8];
    let dark = [115u8, 63u8, 8u8, 255u8];
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            let border = x < 2 || y < 2 || x >= width - 2 || y >= height - 2;
            if border {
                pixels[idx..idx + 4].copy_from_slice(&dark);
                continue;
            }
            let diagonal =
                (x as i32 - y as i32).abs() <= 1 || ((width - 1 - x) as i32 - y as i32).abs() <= 1;
            if diagonal {
                pixels[idx..idx + 4].copy_from_slice(&amber);
            }
        }
    }
    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    image
}
