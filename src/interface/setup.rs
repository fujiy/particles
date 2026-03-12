use bevy::prelude::*;

use super::*;
use crate::params::ActiveInterfaceParams;

pub(super) fn setup_simulation_ui(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    interface_params: Res<ActiveInterfaceParams>,
) {
    let colors = &interface_params.0.colors;
    let layout = &interface_params.0.layout;
    let icon_palette = &interface_params.0.icon_palette;
    let icon_set = create_world_tool_icon_set(&mut images, icon_palette);
    commands.insert_resource(icon_set.clone());

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(10.0),
                top: px(10.0),
                width: px(layout.hud_panel_width_px),
                padding: UiRect::axes(px(10.0), px(6.0)),
                row_gap: px(6.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(colors.hud_bg.to_color()),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(format!(
                    "FPS({:.1}s avg/min): --/--",
                    interface_params.0.behavior.hud_fps_window_sec
                )),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                SimulationHudFpsText,
            ));
            parent
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: px(interface_params.0.profiler.lane_gap_px),
                        ..default()
                    },
                    RuntimeProfileHudRoot,
                ))
                .with_children(|profile| {
                    for lane in [RuntimeProfileLane::Cpu, RuntimeProfileLane::Gpu] {
                        profile
                            .spawn((
                                Node {
                                    align_items: AlignItems::Center,
                                    column_gap: px(6.0),
                                    ..default()
                                },
                            ))
                            .with_children(|row| {
                                row.spawn((
                                    Text::new(lane.label()),
                                    TextFont::from_font_size(
                                        interface_params.0.profiler.lane_label_font_px,
                                    ),
                                    TextColor(Color::srgba(0.9, 0.92, 0.97, 0.95)),
                                ));
                                row.spawn((
                                    Node {
                                        width: px(interface_params.0.profiler.bar_width_px),
                                        height: px(interface_params.0.profiler.bar_height_px),
                                        overflow: Overflow::clip(),
                                        ..default()
                                    },
                                    BackgroundColor(
                                        interface_params.0.profiler.colors.bar_bg.to_color(),
                                    ),
                                    RuntimeProfileLaneBar { lane },
                                ));
                            });
                    }
                });
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
        .spawn((Node {
            position_type: PositionType::Absolute,
            left: px(layout.scale_bar_left_px),
            bottom: px(layout.scale_bar_bottom_px),
            flex_direction: FlexDirection::Column,
            row_gap: px(6.0),
            ..default()
        },))
        .with_children(|parent| {
            parent.spawn((
                Text::new("10 m"),
                TextFont::from_font_size(layout.scale_bar_label_font_px),
                TextColor(Color::srgba(0.94, 0.96, 0.99, 0.96)),
                ScaleBarLabelText,
            ));
            parent.spawn((
                Node {
                    width: px(layout.scale_bar_target_width_px),
                    height: px(layout.scale_bar_height_px),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.94, 0.96, 0.99, 0.96)),
                ScaleBarLine,
            ));
        });

    commands
        .spawn((Node {
            position_type: PositionType::Absolute,
            right: px(layout.save_load_bar_right_px),
            top: px(layout.save_load_bar_top_px),
            column_gap: px(8.0),
            ..default()
        },))
        .with_children(|parent| {
            parent
                .spawn((
                    Button,
                    Node {
                        width: px(layout.save_load_button_width_px),
                        height: px(layout.save_load_button_height_px),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(colors.button_bg_off.to_color()),
                    BorderColor::all(colors.button_border_off.to_color()),
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
                        width: px(layout.save_load_button_width_px),
                        height: px(layout.save_load_button_height_px),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(colors.button_bg_off.to_color()),
                    BorderColor::all(colors.button_border_off.to_color()),
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
                        width: px(layout.save_load_button_width_px),
                        height: px(layout.save_load_button_height_px),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(colors.button_bg_off.to_color()),
                    BorderColor::all(colors.button_border_off.to_color()),
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
                        width: px(layout.save_load_button_width_px),
                        height: px(layout.save_load_button_height_px),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(colors.button_bg_off.to_color()),
                    BorderColor::all(colors.button_border_off.to_color()),
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
                        width: px(layout.save_load_button_width_px),
                        height: px(layout.save_load_button_height_px),
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Center,
                        border: UiRect::all(px(2.0)),
                        ..default()
                    },
                    BackgroundColor(colors.button_bg_off.to_color()),
                    BorderColor::all(colors.button_border_off.to_color()),
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
                right: px(layout.test_assert_panel_right_px),
                top: px(layout.test_assert_panel_top_px),
                width: px(layout.test_assert_panel_width_px),
                display: Display::None,
                padding: UiRect::all(px(8.0)),
                row_gap: px(6.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(colors.hud_bg.to_color()),
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
            bottom: px(layout.toolbar_bottom_px),
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
                    BackgroundColor(colors.toolbar_bg.to_color()),
                ))
                .with_children(|toolbar| {
                    for tool in WORLD_TOOLBAR_TOOLS {
                        toolbar
                            .spawn((
                                Button,
                                Node {
                                    width: px(layout.toolbar_button_size_px),
                                    height: px(layout.toolbar_button_size_px),
                                    align_items: AlignItems::Center,
                                    justify_content: JustifyContent::Center,
                                    border: UiRect::all(px(2.0)),
                                    ..default()
                                },
                                BackgroundColor(colors.button_bg_off.to_color()),
                                BorderColor::all(colors.button_border_off.to_color()),
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
            BackgroundColor(colors.tooltip_bg.to_color()),
            GlobalZIndex(layout.tooltip_global_z_index),
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
                display: Display::None,
                left: px(0.0),
                top: px(0.0),
                padding: UiRect::axes(px(8.0), px(4.0)),
                ..default()
            },
            BackgroundColor(colors.tooltip_bg.to_color()),
            GlobalZIndex(layout.tooltip_global_z_index),
            RuntimeProfileTooltip,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(13.0),
                TextColor(Color::WHITE),
                RuntimeProfileTooltipText,
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
            GlobalZIndex(layout.tooltip_global_z_index - 1),
            SaveLoadDialogRoot,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Node {
                        width: px(layout.dialog_width_px),
                        padding: UiRect::all(px(12.0)),
                        row_gap: px(8.0),
                        flex_direction: FlexDirection::Column,
                        ..default()
                    },
                    BackgroundColor(colors.dialog_bg.to_color()),
                    BorderColor::all(colors.button_border_on.to_color()),
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
                                height: px(layout.dialog_name_input_height_px),
                                align_items: AlignItems::Center,
                                justify_content: JustifyContent::FlexStart,
                                padding: UiRect::horizontal(px(8.0)),
                                border: UiRect::all(px(2.0)),
                                ..default()
                            },
                            BackgroundColor(colors.button_bg_off.to_color()),
                            BorderColor::all(colors.button_border_off.to_color()),
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
                                max_height: px(layout.dialog_slot_list_max_height_px),
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
                                    BackgroundColor(colors.button_bg_off.to_color()),
                                    BorderColor::all(colors.button_border_off.to_color()),
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
                                    BackgroundColor(colors.button_bg_off.to_color()),
                                    BorderColor::all(colors.button_border_off.to_color()),
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
