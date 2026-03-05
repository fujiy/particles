use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use super::*;
use crate::camera_controller::MainCamera;

pub(super) fn update_world_tool_button_visuals(
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

pub(super) fn update_save_load_reset_button_visuals(
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

pub(super) fn update_sim_play_pause_button_visuals(
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

pub(super) fn update_sim_step_button_visuals(
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

pub(super) fn update_sim_parallel_button_visuals(
    parallel_settings: Res<SimulationParallelSettings>,
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor, &mut BorderColor),
        (With<Button>, With<SimParallelButton>),
    >,
) {
    for (interaction, mut bg, mut border_color) in &mut buttons {
        let selected = parallel_settings.enabled;
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

pub(super) fn update_sim_parallel_button_label(
    parallel_settings: Res<SimulationParallelSettings>,
    mut labels: Query<&mut Text, With<SimParallelButtonText>>,
) {
    if !parallel_settings.is_changed() {
        return;
    }
    for mut label in &mut labels {
        label.0 = if parallel_settings.enabled {
            "Parallel: ON".to_string()
        } else {
            "Parallel: OFF".to_string()
        };
    }
}

pub(super) fn update_save_load_name_input_button_visuals(
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

pub(super) fn update_save_load_slot_button_visuals(
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

pub(super) fn update_test_assert_panel(
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

pub(super) fn update_world_tool_tooltip(
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

pub(super) fn update_fps_hud_stats(time: Res<Time>, mut fps_stats: ResMut<FpsHudStats>) {
    let dt = time.delta_secs();
    if dt <= f32::EPSILON {
        return;
    }

    let fps = (1.0 / dt).clamp(0.0, 10_000.0);
    fps_stats
        .frame_samples
        .push_back(FpsFrameSample { dt, fps });
    fps_stats.window_elapsed += dt;

    while fps_stats.window_elapsed > HUD_FPS_WINDOW_SEC && fps_stats.frame_samples.len() > 1 {
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
    fps_stats: Res<FpsHudStats>,
    sim_state: Res<SimulationState>,
    particles: Res<ParticleWorld>,
    terrain_render_diagnostics: Res<TerrainRenderDiagnostics>,
    mut hud_texts: ParamSet<(
        Single<&mut Text, With<SimulationHudFpsText>>,
        Single<&mut Text, With<SimulationHudStatsText>>,
    )>,
) {
    hud_texts.p0().0 = format!(
        "FPS(1s avg/min): {:.1}/{:.1}",
        fps_stats.avg_fps, fps_stats.min_fps
    );

    let sim_status = if sim_state.running {
        "Running"
    } else {
        "Paused"
    };
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
        "Sim: {sim_status}\nTerrainGen/frame: {:>7} (d={:>4},{:>4} full={} r=0x{:02X})\nTerrainOvr/frame: runs={:>5} cells={:>6} pending={:>5} done={:>5.1}%\nTerrainOvr/total: runs={:>7} cells={:>8}\nWater(L): {water_count}\nStone(S): {stone_solid_count}\nStone(G): {stone_granular_count}\nSoil(S): {soil_solid_count}\nSoil(G): {soil_granular_count}\nSand(S): {sand_solid_count}\nSand(G): {sand_granular_count}",
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
    );
}

pub(super) fn update_scale_bar(
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_q: Query<&Projection, With<MainCamera>>,
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
    let target_world_m = meters_per_pixel * SCALE_BAR_TARGET_WIDTH_PX;
    let snapped_world_m = snap_scale_bar_world_length(target_world_m);
    let bar_width_px = (snapped_world_m / meters_per_pixel).max(SCALE_BAR_MIN_WIDTH_PX);

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

pub(super) fn update_step_profiler_panel(
    mut commands: Commands,
    profiler: Res<PhysicsStepProfiler>,
    solver_params: Res<SolverParams>,
    mut label_text: Single<&mut Text, With<StepProfilerMsText>>,
    bar_track_entity: Single<Entity, With<StepProfilerBarTrack>>,
    children_query: Query<&Children>,
) {
    if !profiler.is_changed() {
        return;
    }
    if profiler.total_duration_ms > 0.0 {
        let baseline_frame_ms = solver_params.fixed_dt as f64 * 1000.0;
        let physics_load_percent = profiler.total_duration_ms / baseline_frame_ms * 100.0;
        label_text.0 = format!(
            "Physics Step: {:.2} ms ({:.1}%)",
            profiler.total_duration_ms, physics_load_percent
        );
    } else {
        label_text.0 = "Physics Step: -- ms (--%)".to_string();
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
                    let color =
                        STEP_PROFILER_FLUID_COLORS[fluid_index % STEP_PROFILER_FLUID_COLORS.len()];
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
                    let color = STEP_PROFILER_OBJECT_COLORS
                        [object_index % STEP_PROFILER_OBJECT_COLORS.len()];
                    object_index += 1;
                    color
                }
            };
            parent.spawn((
                Button,
                Node {
                    width: px(
                        (segment.wall_duration_ms as f32 * STEP_PROFILER_BAR_MS_TO_PX).max(1.0),
                    ),
                    height: px(((segment.cpu_duration_ms as f32
                        / segment.wall_duration_ms.max(1e-6) as f32)
                        .clamp(0.05, STEP_PROFILER_MAX_PARALLELISM_DISPLAY))
                        * STEP_PROFILER_PARALLELISM_TO_PX),
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

pub(super) fn update_step_profiler_ideal_parallel(
    grid_hierarchy: Res<GridHierarchy>,
    mpm_block_index_table: Res<MpmBlockIndexTable>,
    mut label_text: Single<&mut Text, With<StepProfilerIdealParallelText>>,
) {
    let block_count = mpm_block_index_table
        .block_count()
        .min(grid_hierarchy.block_count());
    let mut active_block_count = 0usize;
    let mut color_seen = std::collections::BTreeSet::<u16>::new();
    for block_index in 0..block_count {
        if mpm_block_index_table
            .support_indices(block_index)
            .is_empty()
        {
            continue;
        }
        active_block_count += 1;
        if let Some(block) = grid_hierarchy.blocks().get(block_index) {
            color_seen.insert(block.color_class());
        }
    }
    if active_block_count == 0 {
        label_text.0 = "Ideal cpu/wall(block/color): --".to_string();
        return;
    }
    let color_count = color_seen.len().max(1);
    let indicator = active_block_count as f64 / color_count as f64;
    label_text.0 = format!(
        "Ideal cpu/wall(block/color): {:.4} ({} / {})",
        indicator, active_block_count, color_count
    );
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

pub(super) fn update_step_profiler_tooltip(
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
