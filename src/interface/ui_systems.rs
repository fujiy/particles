use bevy::prelude::*;
use bevy::ui::{ComputedNode, UiGlobalTransform};
use bevy::window::PrimaryWindow;

use super::*;
use crate::camera_controller::MainCamera;
use crate::overlay::{MassOverlayState, PhysicsAreaOverlayState, SdfOverlayState, TileOverlayState};
use crate::params::ActiveInterfaceParams;
use crate::params::ActivePhysicsParams;
use crate::physics::gpu_mpm::gpu_resources::world_grid_layout;
use crate::physics::profiler::cpu_profile_span;
use crate::physics::gpu_mpm::sync::{
    MpmChunkResidencyState, MpmStatisticsSnapshot, MpmStatisticsStatus,
};
use crate::physics::material::ParticleMaterial;
use crate::physics::scenario::{ScenarioStatisticsInput, evaluate_scenario_state_from_statistics};

fn phase_id_for_material(material: ParticleMaterial) -> Option<u32> {
    match material {
        ParticleMaterial::WaterLiquid => Some(0),
        ParticleMaterial::StoneGranular => Some(1),
        ParticleMaterial::SoilGranular => Some(1),
        ParticleMaterial::GrassGranular => Some(1),
        ParticleMaterial::SandGranular => Some(2),
        _ => None,
    }
}

pub(super) fn update_world_tool_button_visuals(
    interaction_state: Res<WorldInteractionState>,
    interface_params: Res<ActiveInterfaceParams>,
    mut buttons: Query<(
        &Interaction,
        &WorldToolButton,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    let colors = &interface_params.0.colors;
    for (interaction, button, mut bg, mut border_color) in &mut buttons {
        let selected = interaction_state.selected_tool == Some(button.tool);
        *bg = match *interaction {
            Interaction::Pressed => colors.button_bg_press.to_color().into(),
            Interaction::Hovered => colors.button_bg_hover.to_color().into(),
            Interaction::None => toggle_button_bg(selected, colors),
        };
        *border_color = if selected {
            BorderColor::all(colors.button_border_on.to_color())
        } else {
            BorderColor::all(colors.button_border_off.to_color())
        };
    }
}

pub(super) fn update_save_load_open_button_visuals(
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
    interface_params: Res<ActiveInterfaceParams>,
) {
    let colors = &interface_params.0.colors;
    for (interaction, mut bg, mut border_color, button) in &mut buttons {
        let selected = save_load_ui_state.mode == Some(button.mode);
        *bg = match *interaction {
            Interaction::Pressed => colors.button_bg_press.to_color().into(),
            Interaction::Hovered => colors.button_bg_hover.to_color().into(),
            Interaction::None => toggle_button_bg(selected, colors),
        };
        *border_color = if selected {
            BorderColor::all(colors.button_border_on.to_color())
        } else {
            BorderColor::all(colors.button_border_off.to_color())
        };
    }
}

pub(super) fn update_save_load_reset_button_visuals(
    interface_params: Res<ActiveInterfaceParams>,
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        (With<Button>, With<SaveLoadResetButton>),
    >,
) {
    let colors = &interface_params.0.colors;
    for (interaction, mut bg, mut border_color) in &mut buttons {
        *bg = match *interaction {
            Interaction::Pressed => colors.button_bg_press.to_color().into(),
            Interaction::Hovered => colors.button_bg_hover.to_color().into(),
            Interaction::None => colors.button_bg_off.to_color().into(),
        };
        *border_color = BorderColor::all(colors.button_border_off.to_color());
    }
}

pub(super) fn update_sim_play_pause_button_visuals(
    sim_state: Res<SimulationState>,
    interface_params: Res<ActiveInterfaceParams>,
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        (With<Button>, With<SimPlayPauseButton>),
    >,
) {
    let colors = &interface_params.0.colors;
    for (interaction, mut bg, mut border_color) in &mut buttons {
        let selected = sim_state.running;
        *bg = match *interaction {
            Interaction::Pressed => colors.button_bg_press.to_color().into(),
            Interaction::Hovered => colors.button_bg_hover.to_color().into(),
            Interaction::None => toggle_button_bg(selected, colors),
        };
        *border_color = if selected {
            BorderColor::all(colors.button_border_on.to_color())
        } else {
            BorderColor::all(colors.button_border_off.to_color())
        };
    }
}

pub(super) fn update_sim_step_button_visuals(
    interface_params: Res<ActiveInterfaceParams>,
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        (With<Button>, With<SimStepButton>),
    >,
) {
    let colors = &interface_params.0.colors;
    for (interaction, mut bg, mut border_color) in &mut buttons {
        *bg = match *interaction {
            Interaction::Pressed => colors.button_bg_press.to_color().into(),
            Interaction::Hovered => colors.button_bg_hover.to_color().into(),
            Interaction::None => colors.button_bg_off.to_color().into(),
        };
        *border_color = BorderColor::all(colors.button_border_off.to_color());
    }
}

pub(super) fn update_sim_play_pause_button_label(
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

pub(super) fn update_save_load_name_input_button_visuals(
    save_load_ui_state: Res<SaveLoadUiState>,
    interface_params: Res<ActiveInterfaceParams>,
    mut inputs: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        With<SaveLoadDialogNameInputButton>,
    >,
) {
    let colors = &interface_params.0.colors;
    for (interaction, mut bg, mut border_color) in &mut inputs {
        let selected = save_load_ui_state.input_focused;
        *bg = match *interaction {
            Interaction::Pressed => colors.button_bg_press.to_color().into(),
            Interaction::Hovered => colors.button_bg_hover.to_color().into(),
            Interaction::None => toggle_button_bg(selected, colors),
        };
        *border_color = if selected {
            BorderColor::all(colors.button_border_on.to_color())
        } else {
            BorderColor::all(colors.button_border_off.to_color())
        };
    }
}

pub(super) fn update_save_load_slot_button_visuals(
    save_load_ui_state: Res<SaveLoadUiState>,
    interface_params: Res<ActiveInterfaceParams>,
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
    let colors = &interface_params.0.colors;
    for (interaction, button, mut bg, mut border_color) in &mut buttons {
        let selected = save_load_ui_state.selected_slot.as_deref()
            == Some(button.slot_name.as_str())
            && save_load_ui_state.selected_slot_source == Some(button.source);
        *bg = match *interaction {
            Interaction::Pressed => colors.button_bg_press.to_color().into(),
            Interaction::Hovered => colors.button_bg_hover.to_color().into(),
            Interaction::None => toggle_button_bg(selected, colors),
        };
        *border_color = if selected {
            BorderColor::all(colors.button_border_on.to_color())
        } else {
            BorderColor::all(colors.button_border_off.to_color())
        };
    }
}

pub(super) fn update_test_assert_panel(
    mut commands: Commands,
    replay_state: Res<ReplayState>,
    active_physics_params: Res<ActivePhysicsParams>,
    terrain: Res<TerrainWorld>,
    stats_snapshot: Res<MpmStatisticsSnapshot>,
    mut stats_status: ResMut<MpmStatisticsStatus>,
    mut panel_node: Single<&mut Node, With<TestAssertPanelRoot>>,
    title_entity: Single<Entity, With<TestAssertTitleText>>,
    list_entity: Single<Entity, With<TestAssertList>>,
    mut text_query: Query<&mut Text>,
    children_query: Query<&Children>,
) {
    stats_status.total_particles = true;
    stats_status.phase_counts = true;
    stats_status.penetration = true;
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
    stats_status.max_speed |= spec.thresholds.max_max_speed_mps.is_some();
    stats_status.water_surface_p95 |= spec.water_surface_assertion.is_some();
    stats_status.granular_repose |= spec.granular_repose_assertion.is_some();
    stats_status.material_interaction |= spec.material_interaction_assertion.is_some();
    if let Some(repose) = spec.granular_repose_assertion {
        stats_status.repose_phase_id = phase_id_for_material(repose.material).unwrap_or(1);
    }
    if let Some(interaction) = spec.material_interaction_assertion {
        stats_status.interaction_primary_phase_id =
            phase_id_for_material(interaction.primary_material).unwrap_or(2);
        stats_status.interaction_secondary_phase_id =
            phase_id_for_material(interaction.secondary_material).unwrap_or(0);
    }
    panel_node.display = Display::Flex;

    let stats_input = ScenarioStatisticsInput {
        particle_count: stats_snapshot.total() as usize,
        sleeping_ratio: 0.0,
        max_speed_mps: stats_snapshot.max_speed_mps,
        terrain_penetration_rate: stats_snapshot.all_penetration_ratio(),
        water_surface_p95_cell: stats_snapshot.water_surface_p95_cell,
        granular_repose_angle_deg: stats_snapshot.granular_repose_angle_deg,
        granular_repose_base_span_cells: stats_snapshot.granular_repose_base_span_cells,
        material_interaction_contact_ratio: Some(stats_snapshot.material_interaction_contact_ratio),
        material_interaction_primary_centroid_y: stats_snapshot
            .material_interaction_primary_centroid_y_m,
        material_interaction_secondary_centroid_y: stats_snapshot
            .material_interaction_secondary_centroid_y_m,
    };

    let (metrics, assertions) = evaluate_scenario_state_from_statistics(
        &spec,
        replay_state.current_step,
        active_physics_params.0.fixed_dt,
        replay_state.baseline_particle_count,
        replay_state.baseline_solid_cell_count,
        &terrain,
        stats_input,
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
            let condition_suffix = if row.condition == "always" {
                String::new()
            } else {
                format!(" (when: {})", row.condition)
            };
            spawn_assertion_line(
                parent,
                row.ok,
                row.active,
                format!(
                    "{} expected {} actual {}{}",
                    row.label, row.expected, row.actual, condition_suffix
                ),
            );
        }
    });
}

pub(super) fn update_save_load_dialog(
    mut commands: Commands,
    interface_params: Res<ActiveInterfaceParams>,
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
    let colors = &interface_params.0.colors;
    let layout = &interface_params.0.layout;
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
                        height: px(layout.dialog_slot_button_height_px),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::FlexStart,
                        padding: UiRect::horizontal(px(8.0)),
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(colors.button_bg_off.to_color()),
                    BorderColor::all(colors.button_border_off.to_color()),
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
                        height: px(layout.dialog_slot_button_height_px),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::FlexStart,
                        padding: UiRect::horizontal(px(8.0)),
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(colors.button_bg_off.to_color()),
                    BorderColor::all(colors.button_border_off.to_color()),
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
                            height: px(layout.dialog_slot_button_height_px),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::FlexStart,
                            padding: UiRect::horizontal(px(8.0)),
                            border: UiRect::all(px(2.0)),
                            ..default()
                        },
                        BackgroundColor(colors.button_bg_off.to_color()),
                        BorderColor::all(colors.button_border_off.to_color()),
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

pub(super) fn update_world_tool_tooltip(
    windows: Query<&Window, With<PrimaryWindow>>,
    buttons: Query<(&Interaction, &WorldToolButton), With<Button>>,
    interface_params: Res<ActiveInterfaceParams>,
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
    tooltip.left = px(cursor.x + interface_params.0.layout.tooltip_cursor_offset_x_px);
    tooltip.top = px(cursor.y + interface_params.0.layout.tooltip_cursor_offset_y_px);
    tooltip_text.0 = tool.label().to_string();
}

pub(super) fn update_fps_hud_stats(
    time: Res<Time>,
    interface_params: Res<ActiveInterfaceParams>,
    mut fps_stats: ResMut<FpsHudStats>,
) {
    let dt = time.delta_secs();
    if dt <= f32::EPSILON {
        return;
    }

    let fps = (1.0 / dt).clamp(0.0, 10_000.0);
    fps_stats
        .frame_samples
        .push_back(FpsFrameSample { dt, fps });
    fps_stats.window_elapsed += dt;

    while fps_stats.window_elapsed > interface_params.0.behavior.hud_fps_window_sec
        && fps_stats.frame_samples.len() > 1
    {
        let Some(sample) = fps_stats.frame_samples.pop_front() else {
            break;
        };
        fps_stats.window_elapsed -= sample.dt;
    }

    if fps_stats.frame_samples.is_empty() {
        fps_stats.avg_fps = 0.0;
        fps_stats.min_fps = 0.0;
        return;
    }

    fps_stats.avg_fps = fps_stats.frame_samples.len() as f32 / fps_stats.window_elapsed.max(1e-5);
    fps_stats.min_fps = fps_stats
        .frame_samples
        .iter()
        .fold(f32::INFINITY, |acc, sample| acc.min(sample.fps));
}

pub(super) fn update_simulation_hud(
    interface_params: Res<ActiveInterfaceParams>,
    fps_stats: Res<FpsHudStats>,
    sim_state: Res<SimulationState>,
    phase_counts: Res<MpmStatisticsSnapshot>,
    chunk_overlay_state: Res<TileOverlayState>,
    physics_overlay_state: Res<PhysicsAreaOverlayState>,
    sdf_overlay_state: Res<SdfOverlayState>,
    mass_overlay_state: Res<MassOverlayState>,
    chunk_residency: Res<MpmChunkResidencyState>,
    terrain_render_diagnostics: Res<TerrainRenderDiagnostics>,
    terrain_world: Res<TerrainWorld>,
    mut hud_texts: ParamSet<(
        Single<&mut Text, With<SimulationHudFpsText>>,
        Single<&mut Text, With<SimulationHudStatsText>>,
    )>,
) {
    let _profile_span = cpu_profile_span("ui", "update_simulation_hud").entered();
    hud_texts.p0().0 = format!(
        "FPS({:.1}s avg/min): {:.1}/{:.1}",
        interface_params.0.behavior.hud_fps_window_sec, fps_stats.avg_fps, fps_stats.min_fps
    );

    let sim_status = if sim_state.running {
        "Running"
    } else {
        "Paused"
    };
    let water_count = phase_counts.water_liquid as usize;
    let granular_soil_like_count = phase_counts.soil_granular as usize;
    let granular_sand_count = phase_counts.sand_granular as usize;
    let unknown_phase_count = phase_counts.unknown as usize;
    let gpu_total_count = phase_counts.total() as usize;
    let overlay_enabled = chunk_overlay_state.enabled || physics_overlay_state.enabled;
    let layout = if chunk_residency.initialized {
        chunk_residency.grid_layout
    } else {
        world_grid_layout()
    };
    let grid_cells = UVec2::new(
        layout.dims.x.saturating_sub(1),
        layout.dims.y.saturating_sub(1),
    );
    let loaded_chunks = terrain_world.loaded_chunk_coords().len();
    let modified_chunks = terrain_world.override_chunk_coords().len();
    hud_texts.p1().0 = format!(
        "Sim: {sim_status}\nTerrainGen/frame: {:>7} (d={:>4},{:>4} full={} r=0x{:02X})\nTerrainOvr/frame: runs={:>5} cells={:>6} pending={:>5} done={:>5.1}%\nTerrainOvr/total: runs={:>7} cells={:>8}\nGPU Total: {gpu_total_count}\nWater(L): {water_count}\nGranular(Soil+Stone): {granular_soil_like_count}\nGranular(Sand): {granular_sand_count}\nGPU Unknown: {unknown_phase_count}\nChunk/Grid Overlay: {} | resident {} chunks | active tiles {}/{} skip {:>5.1}% | nodes {}x{} cells {}x{}\nRender Chunks: {} (Modified: {})\nSDF Overlay: {} | Mass Overlay: {}",
        terrain_render_diagnostics.terrain_generation_eval_count_frame,
        terrain_render_diagnostics.terrain_generation_origin_delta_x_frame,
        terrain_render_diagnostics.terrain_generation_origin_delta_y_frame,
        u8::from(terrain_render_diagnostics.terrain_generation_full_refresh_frame),
        terrain_render_diagnostics.terrain_generation_full_refresh_reason_bits,
        terrain_render_diagnostics.terrain_override_runs_frame,
        terrain_render_diagnostics.terrain_override_cells_frame,
        terrain_render_diagnostics.terrain_override_pending_runs,
        terrain_render_diagnostics.terrain_override_budget_completion_frame * 100.0,
        terrain_render_diagnostics.terrain_override_runs_total,
        terrain_render_diagnostics.terrain_override_cells_total,
        if overlay_enabled { "ON" } else { "OFF" },
        chunk_residency.resident_chunk_count,
        chunk_residency.active_tile_count,
        chunk_residency.active_tile_capacity,
        chunk_residency.inactive_skip_rate * 100.0,
        layout.dims.x,
        layout.dims.y,
        grid_cells.x,
        grid_cells.y,
        loaded_chunks,
        modified_chunks,
        if sdf_overlay_state.enabled {
            "ON"
        } else {
            "OFF"
        },
        if mass_overlay_state.enabled {
            "ON"
        } else {
            "OFF"
        },
    );
}

pub(super) fn sync_runtime_profile_hud(
    mut commands: Commands,
    interface_params: Res<ActiveInterfaceParams>,
    snapshot: Res<RuntimeProfileSnapshot>,
    mut hud_state: ResMut<RuntimeProfileHudState>,
    mut lane_bars: Query<(&RuntimeProfileLaneBar, &mut Node, &mut BackgroundColor, Entity)>,
) {
    let _profile_span = cpu_profile_span("ui", "sync_runtime_profile_hud").entered();
    if !snapshot.is_changed() && !interface_params.is_changed() {
        return;
    }

    let profiler = &interface_params.0.profiler;
    for (lane_bar, mut node, mut bg, entity) in &mut lane_bars {
        node.width = px(profiler.bar_width_px);
        node.height = px(profiler.bar_height_px);
        *bg = profiler.colors.bar_bg.to_color().into();

        commands.entity(entity).despawn_children();

        let segments = match lane_bar.lane {
            RuntimeProfileLane::Cpu => &snapshot.cpu,
            RuntimeProfileLane::Gpu => &snapshot.gpu,
        };
        let bar_width_px = profiler.bar_width_px.max(1.0);
        let scale_ms_per_sec = snapshot.scale_ms_per_sec.max(1.0);
        commands.entity(entity).with_children(|parent| {
            for (index, segment) in segments.iter().enumerate() {
                let width_px = (segment.rate_ms_per_sec / scale_ms_per_sec) * bar_width_px;
                if width_px <= 0.0 {
                    continue;
                }
                parent.spawn((
                    Node {
                        width: px(width_px),
                        height: percent(100.0),
                        ..default()
                    },
                    BackgroundColor(profile_segment_color(
                        profiler.colors.clone(),
                        &segment.category,
                        &segment.detail,
                    )),
                    RuntimeProfileSegmentNode {
                        lane: lane_bar.lane,
                        index,
                    },
                ));
            }
        });
    }

    hud_state.rendered_sequence = snapshot.sequence;
}

pub(super) fn update_runtime_profile_tooltip(
    windows: Query<&Window, With<PrimaryWindow>>,
    interface_params: Res<ActiveInterfaceParams>,
    snapshot: Res<RuntimeProfileSnapshot>,
    mut tooltip: Single<&mut Node, With<RuntimeProfileTooltip>>,
    mut tooltip_text: Single<&mut Text, With<RuntimeProfileTooltipText>>,
    lane_bars: Query<(&RuntimeProfileLaneBar, &ComputedNode, &UiGlobalTransform)>,
    mut segment_nodes: Query<(&RuntimeProfileSegmentNode, &mut BackgroundColor)>,
) {
    let _profile_span = cpu_profile_span("ui", "update_runtime_profile_tooltip").entered();
    let Some(window) = windows.iter().next() else {
        tooltip.display = Display::None;
        apply_runtime_profile_hover_colors(&snapshot, &interface_params.0.profiler.colors, None, &mut segment_nodes);
        return;
    };
    let Some(cursor_physical) = window.physical_cursor_position() else {
        tooltip.display = Display::None;
        apply_runtime_profile_hover_colors(&snapshot, &interface_params.0.profiler.colors, None, &mut segment_nodes);
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        tooltip.display = Display::None;
        apply_runtime_profile_hover_colors(&snapshot, &interface_params.0.profiler.colors, None, &mut segment_nodes);
        return;
    };
    let hovered_segment = lane_bars
        .iter()
        .filter_map(|(lane_bar, node, transform)| {
            if !node.contains_point(*transform, cursor_physical) {
                return None;
            }
            let normalized = node.normalize_point(*transform, cursor_physical)?;
            let x = normalized.x + 0.5;
            if !(0.0..=1.0).contains(&x) {
                return None;
            }
            let segments = match lane_bar.lane {
                RuntimeProfileLane::Cpu => &snapshot.cpu,
                RuntimeProfileLane::Gpu => &snapshot.gpu,
            };
            segment_at_bar_position(segments, snapshot.scale_ms_per_sec, x)
                .map(|(index, segment)| (lane_bar.lane, index, segment))
        })
        .next();
    apply_runtime_profile_hover_colors(
        &snapshot,
        &interface_params.0.profiler.colors,
        hovered_segment.map(|(lane, index, _)| (lane, index)),
        &mut segment_nodes,
    );
    let Some((_, _, segment)) = hovered_segment else {
        tooltip.display = Display::None;
        return;
    };

    tooltip.display = Display::Flex;
    tooltip.left = px(cursor.x + interface_params.0.layout.tooltip_cursor_offset_x_px);
    tooltip.top = px(cursor.y + interface_params.0.layout.tooltip_cursor_offset_y_px);
    tooltip_text.0 = format!(
        "{} {}\n{:.1} ms/s\ninterval total: {:.2} ms",
        segment.lane.label(),
        segment.label,
        segment.rate_ms_per_sec,
        segment.total_ms,
    );
}

pub(super) fn update_scale_bar(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<&Projection, With<MainCamera>>,
    interface_params: Res<ActiveInterfaceParams>,
    mut bar_line: Single<&mut Node, With<ScaleBarLine>>,
    mut bar_text: Single<&mut Text, With<ScaleBarLabelText>>,
) {
    let Some(window) = windows.iter().next() else {
        return;
    };
    let Some(projection) = camera_q.iter().next() else {
        return;
    };
    let Projection::Orthographic(ortho) = projection else {
        return;
    };

    let viewport_world_w_m = (ortho.area.max.x - ortho.area.min.x).abs().max(1e-6);
    let meters_per_pixel = viewport_world_w_m / window.width().max(1.0);
    let target_world_m = meters_per_pixel * interface_params.0.layout.scale_bar_target_width_px;
    let snapped_world_m = snap_scale_bar_world_length(target_world_m);
    let bar_width_px =
        (snapped_world_m / meters_per_pixel).max(interface_params.0.layout.scale_bar_min_width_px);

    bar_line.width = px(bar_width_px);
    bar_text.0 = if snapped_world_m >= 1000.0 {
        format!("{:.0} km", snapped_world_m / 1000.0)
    } else {
        format!("{:.0} m", snapped_world_m)
    };
}

fn snap_scale_bar_world_length(target_world_m: f32) -> f32 {
    let target = target_world_m.max(1.0);
    let exp = target.log10().floor() as i32;
    let base = 10.0f32.powi(exp);
    let candidates = [1.0 * base, 2.0 * base, 5.0 * base, 10.0 * base];
    candidates
        .into_iter()
        .min_by(|a, b| {
            (a - target)
                .abs()
                .partial_cmp(&(b - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(1.0)
        .max(1.0)
}

fn profile_segment_color(
    colors: crate::params::interface::InterfaceProfilerColorParams,
    category: &str,
    detail: &str,
) -> Color {
    let base = match normalized_profile_category(category) {
        "physics" => colors.physics,
        "render" => colors.render,
        _ => colors.others,
    };
    let mix = profile_detail_mix(detail);
    let lift = mix.max(0.0);
    let shade = (-mix).max(0.0);
    Color::srgba(
        lerp(lerp(base.r, 1.0, lift), 0.0, shade),
        lerp(lerp(base.g, 1.0, lift), 0.0, shade),
        lerp(lerp(base.b, 1.0, lift), 0.0, shade),
        base.a,
    )
}

fn profile_detail_mix(detail: &str) -> f32 {
    let mut hash = 0u32;
    for byte in detail.bytes() {
        hash = hash.wrapping_mul(16777619).wrapping_add(byte as u32);
    }
    ((hash & 0xFF) as f32 / 255.0 - 0.5) * 0.45
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn normalized_profile_category(category: &str) -> &'static str {
    match category {
        "physics" => "physics",
        "terrain" | "water" | "overlay" | "ui" | "render" => "render",
        _ => "others",
    }
}

fn segment_at_bar_position(
    segments: &[RuntimeProfileSegment],
    scale_ms_per_sec: f32,
    x01: f32,
) -> Option<(usize, &RuntimeProfileSegment)> {
    let mut cursor = x01.clamp(0.0, 1.0);
    let mut accum = 0.0;
    for (index, segment) in segments.iter().enumerate() {
        let width = (segment.rate_ms_per_sec / scale_ms_per_sec.max(1.0)).max(0.0);
        let end = (accum + width).min(1.0);
        if cursor >= accum && cursor <= end && end > accum {
            return Some((index, segment));
        }
        accum = end;
        if accum >= 1.0 {
            break;
        }
        if cursor < accum {
            cursor = accum;
        }
    }
    None
}

fn apply_runtime_profile_hover_colors(
    snapshot: &RuntimeProfileSnapshot,
    colors: &crate::params::interface::InterfaceProfilerColorParams,
    hovered: Option<(RuntimeProfileLane, usize)>,
    segment_nodes: &mut Query<(&RuntimeProfileSegmentNode, &mut BackgroundColor)>,
) {
    for (segment_node, mut bg) in segment_nodes.iter_mut() {
        let color = if hovered == Some((segment_node.lane, segment_node.index)) {
            Color::WHITE
        } else {
            runtime_profile_segment_color(snapshot, colors.clone(), segment_node)
        };
        *bg = color.into();
    }
}

fn runtime_profile_segment_color(
    snapshot: &RuntimeProfileSnapshot,
    colors: crate::params::interface::InterfaceProfilerColorParams,
    segment_node: &RuntimeProfileSegmentNode,
) -> Color {
    let segment = match segment_node.lane {
        RuntimeProfileLane::Cpu => snapshot.cpu.get(segment_node.index),
        RuntimeProfileLane::Gpu => snapshot.gpu.get(segment_node.index),
    };
    let Some(segment) = segment else {
        return colors.bar_bg.to_color();
    };
    profile_segment_color(colors, &segment.category, &segment.detail)
}
