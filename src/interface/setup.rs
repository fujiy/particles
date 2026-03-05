use bevy::prelude::*;

use super::*;

pub(super) fn setup_simulation_ui(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
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
                Text::new("FPS(1s avg/min): --/--"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                SimulationHudFpsText,
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
        .spawn((Node {
            position_type: PositionType::Absolute,
            left: px(SCALE_BAR_LEFT_PX),
            bottom: px(SCALE_BAR_BOTTOM_PX),
            flex_direction: FlexDirection::Column,
            row_gap: px(6.0),
            ..default()
        },))
        .with_children(|parent| {
            parent.spawn((
                Text::new("10 m"),
                TextFont::from_font_size(SCALE_BAR_LABEL_FONT_PX),
                TextColor(Color::srgba(0.94, 0.96, 0.99, 0.96)),
                ScaleBarLabelText,
            ));
            parent.spawn((
                Node {
                    width: px(SCALE_BAR_TARGET_WIDTH_PX),
                    height: px(SCALE_BAR_HEIGHT_PX),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.94, 0.96, 0.99, 0.96)),
                ScaleBarLine,
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
