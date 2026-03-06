use bevy::prelude::*;

use super::*;
use crate::params::ActiveOverlayParams;
use crate::params::overlay::OverlayColorParams;
use crate::physics::gpu_mpm::gpu_resources::world_grid_layout;
use crate::physics::gpu_mpm::sync::MpmChunkResidencyState;

pub(super) fn setup_overlay_ui(mut commands: Commands, overlay_params: Res<ActiveOverlayParams>) {
    let colors = &overlay_params.0.colors;
    let ui = &overlay_params.0.ui;
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(ui.right_px),
                bottom: px(ui.tile_button_bottom_px),
                padding: UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px)),
                ..default()
            },
            BackgroundColor(colors.button_bg_off.to_color()),
            TileOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Chunk Overlay: OFF"),
                TextFont::from_font_size(ui.button_font_size_px),
                TextColor(Color::WHITE),
                TileOverlayToggleButtonLabel,
            ));
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(ui.right_px),
                bottom: px(ui.sdf_button_bottom_px),
                padding: UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px)),
                ..default()
            },
            BackgroundColor(colors.button_bg_off.to_color()),
            SdfOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("SDF Overlay: OFF"),
                TextFont::from_font_size(ui.button_font_size_px),
                TextColor(Color::WHITE),
                SdfOverlayToggleButtonLabel,
            ));
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(ui.right_px),
                bottom: px(ui.physics_area_button_bottom_px),
                padding: UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px)),
                ..default()
            },
            BackgroundColor(colors.button_bg_off.to_color()),
            PhysicsAreaOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Physics Area Overlay: OFF"),
                TextFont::from_font_size(ui.button_font_size_px),
                TextColor(Color::WHITE),
                PhysicsAreaOverlayToggleButtonLabel,
            ));
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(ui.right_px),
                bottom: px(ui.particle_button_bottom_px),
                padding: UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px)),
                ..default()
            },
            BackgroundColor(colors.button_bg_off.to_color()),
            ParticleOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Particle Overlay: OFF"),
                TextFont::from_font_size(ui.button_font_size_px),
                TextColor(Color::WHITE),
                ParticleOverlayToggleButtonLabel,
            ));
        });

    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(ui.info_font_size_px),
        TextColor(Color::srgba(0.96, 0.96, 0.98, 0.95)),
        Node {
            position_type: PositionType::Absolute,
            left: px(ui.info_left_px),
            top: px(ui.info_top_px),
            ..default()
        },
        OverlayInfoText,
    ));
}

pub(super) fn apply_overlay_ui_params(
    overlay_params: Res<ActiveOverlayParams>,
    mut nodes: ParamSet<(
        Single<&mut Node, With<TileOverlayToggleButton>>,
        Single<&mut Node, With<SdfOverlayToggleButton>>,
        Single<&mut Node, With<PhysicsAreaOverlayToggleButton>>,
        Single<&mut Node, With<ParticleOverlayToggleButton>>,
        Single<&mut Node, With<OverlayInfoText>>,
    )>,
    mut fonts: ParamSet<(
        Single<&mut TextFont, With<TileOverlayToggleButtonLabel>>,
        Single<&mut TextFont, With<SdfOverlayToggleButtonLabel>>,
        Single<&mut TextFont, With<PhysicsAreaOverlayToggleButtonLabel>>,
        Single<&mut TextFont, With<ParticleOverlayToggleButtonLabel>>,
        Single<&mut TextFont, With<OverlayInfoText>>,
    )>,
) {
    if !overlay_params.is_changed() {
        return;
    }
    let ui = &overlay_params.0.ui;

    let mut tile_node = nodes.p0();
    tile_node.right = px(ui.right_px);
    tile_node.bottom = px(ui.tile_button_bottom_px);
    tile_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    let mut sdf_node = nodes.p1();
    sdf_node.right = px(ui.right_px);
    sdf_node.bottom = px(ui.sdf_button_bottom_px);
    sdf_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    let mut physics_node = nodes.p2();
    physics_node.right = px(ui.right_px);
    physics_node.bottom = px(ui.physics_area_button_bottom_px);
    physics_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    let mut particle_node = nodes.p3();
    particle_node.right = px(ui.right_px);
    particle_node.bottom = px(ui.particle_button_bottom_px);
    particle_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    let mut info_node = nodes.p4();
    info_node.left = px(ui.info_left_px);
    info_node.top = px(ui.info_top_px);

    fonts.p0().font_size = ui.button_font_size_px;
    fonts.p1().font_size = ui.button_font_size_px;
    fonts.p2().font_size = ui.button_font_size_px;
    fonts.p3().font_size = ui.button_font_size_px;
    fonts.p4().font_size = ui.info_font_size_px;
}

pub(super) fn handle_particle_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<ParticleOverlayToggleButton>),
    >,
    overlay_params: Res<ActiveOverlayParams>,
    mut overlay_state: ResMut<ParticleOverlayState>,
) {
    let colors = &overlay_params.0.colors;
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                overlay_state.enabled = !overlay_state.enabled;
                *bg = colors.button_bg_press.to_color().into();
            }
            Interaction::Hovered => {
                *bg = colors.button_bg_hover.to_color().into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(overlay_state.enabled, colors);
            }
        }
    }
}

pub(super) fn handle_tile_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<TileOverlayToggleButton>),
    >,
    overlay_params: Res<ActiveOverlayParams>,
    mut overlay_state: ResMut<TileOverlayState>,
    mut physics_overlay_state: ResMut<PhysicsAreaOverlayState>,
) {
    let colors = &overlay_params.0.colors;
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                let merged_enabled = overlay_state.enabled || physics_overlay_state.enabled;
                let next = !merged_enabled;
                overlay_state.enabled = next;
                physics_overlay_state.enabled = next;
                *bg = colors.button_bg_press.to_color().into();
            }
            Interaction::Hovered => {
                *bg = colors.button_bg_hover.to_color().into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(
                    overlay_state.enabled || physics_overlay_state.enabled,
                    colors,
                );
            }
        }
    }
}

pub(super) fn handle_physics_area_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<PhysicsAreaOverlayToggleButton>),
    >,
    overlay_params: Res<ActiveOverlayParams>,
    mut overlay_state: ResMut<PhysicsAreaOverlayState>,
    mut tile_overlay_state: ResMut<TileOverlayState>,
) {
    let colors = &overlay_params.0.colors;
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                let merged_enabled = overlay_state.enabled || tile_overlay_state.enabled;
                let next = !merged_enabled;
                overlay_state.enabled = next;
                tile_overlay_state.enabled = next;
                *bg = colors.button_bg_press.to_color().into();
            }
            Interaction::Hovered => {
                *bg = colors.button_bg_hover.to_color().into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(
                    overlay_state.enabled || tile_overlay_state.enabled,
                    colors,
                );
            }
        }
    }
}

pub(super) fn handle_sdf_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<SdfOverlayToggleButton>),
    >,
    overlay_params: Res<ActiveOverlayParams>,
    mut overlay_state: ResMut<SdfOverlayState>,
) {
    let colors = &overlay_params.0.colors;
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                overlay_state.enabled = !overlay_state.enabled;
                *bg = colors.button_bg_press.to_color().into();
            }
            Interaction::Hovered => {
                *bg = colors.button_bg_hover.to_color().into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(overlay_state.enabled, colors);
            }
        }
    }
}

pub(super) fn update_tile_overlay_button_label(
    overlay_state: Res<TileOverlayState>,
    physics_overlay_state: Res<PhysicsAreaOverlayState>,
    chunk_residency: Res<MpmChunkResidencyState>,
    mut labels: Query<&mut Text, With<TileOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() && !physics_overlay_state.is_changed() && !chunk_residency.is_changed() {
        return;
    }

    let enabled = overlay_state.enabled || physics_overlay_state.enabled;
    let layout = if chunk_residency.initialized {
        chunk_residency.grid_layout
    } else {
        world_grid_layout()
    };

    for mut label in &mut labels {
        label.0 = if enabled {
            format!(
                "Chunk Overlay: ON (GPU resident:{} grid:{}x{})",
                chunk_residency.resident_chunk_count, layout.dims.x, layout.dims.y,
            )
        } else {
            "Chunk Overlay: OFF".to_string()
        };
    }
}

pub(super) fn sync_overlay_button_backgrounds(
    overlay_params: Res<ActiveOverlayParams>,
    tile_overlay_state: Res<TileOverlayState>,
    sdf_overlay_state: Res<SdfOverlayState>,
    physics_overlay_state: Res<PhysicsAreaOverlayState>,
    particle_overlay_state: Res<ParticleOverlayState>,
    mut buttons: ParamSet<(
        Query<&mut BackgroundColor, With<TileOverlayToggleButton>>,
        Query<&mut BackgroundColor, With<SdfOverlayToggleButton>>,
        Query<&mut BackgroundColor, With<PhysicsAreaOverlayToggleButton>>,
        Query<&mut BackgroundColor, With<ParticleOverlayToggleButton>>,
    )>,
) {
    if !overlay_params.is_changed()
        && !tile_overlay_state.is_changed()
        && !sdf_overlay_state.is_changed()
        && !physics_overlay_state.is_changed()
        && !particle_overlay_state.is_changed()
    {
        return;
    }

    let colors = &overlay_params.0.colors;
    let merged_chunk_enabled = tile_overlay_state.enabled || physics_overlay_state.enabled;
    for mut bg in &mut buttons.p0() {
        *bg = toggle_button_bg(merged_chunk_enabled, colors);
    }
    for mut bg in &mut buttons.p2() {
        *bg = toggle_button_bg(merged_chunk_enabled, colors);
    }
    for mut bg in &mut buttons.p1() {
        *bg = toggle_button_bg(sdf_overlay_state.enabled, colors);
    }
    for mut bg in &mut buttons.p3() {
        *bg = toggle_button_bg(particle_overlay_state.enabled, colors);
    }
}

pub(super) fn update_sdf_overlay_button_label(
    overlay_state: Res<SdfOverlayState>,
    mut labels: Query<&mut Text, With<SdfOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() {
        return;
    }
    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            "SDF Overlay: ON".to_string()
        } else {
            "SDF Overlay: OFF".to_string()
        };
    }
}

pub(super) fn update_physics_area_overlay_button_label(
    tile_overlay_state: Res<TileOverlayState>,
    overlay_state: Res<PhysicsAreaOverlayState>,
    chunk_residency: Res<MpmChunkResidencyState>,
    mut labels: Query<&mut Text, With<PhysicsAreaOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() && !tile_overlay_state.is_changed() && !chunk_residency.is_changed() {
        return;
    }

    let enabled = overlay_state.enabled || tile_overlay_state.enabled;
    let gpu_layout = if chunk_residency.initialized {
        chunk_residency.grid_layout
    } else {
        world_grid_layout()
    };
    let gpu_cells = UVec2::new(
        gpu_layout.dims.x.saturating_sub(1),
        gpu_layout.dims.y.saturating_sub(1),
    );
    for mut label in &mut labels {
        label.0 = if enabled {
            format!(
                "Physics Area Overlay: ON (Merged, resident:{} GPU:{}x{} nodes {}x{} cells)",
                chunk_residency.resident_chunk_count,
                gpu_layout.dims.x,
                gpu_layout.dims.y,
                gpu_cells.x,
                gpu_cells.y,
            )
        } else {
            "Physics Area Overlay: OFF".to_string()
        };
    }
}

pub(super) fn update_particle_overlay_button_label(
    overlay_state: Res<ParticleOverlayState>,
    mut labels: Query<&mut Text, With<ParticleOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() {
        return;
    }

    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            "Particle Overlay: ON".to_string()
        } else {
            "Particle Overlay: OFF".to_string()
        };
    }
}

pub(super) fn update_overlay_info_text(
    tile_overlay_state: Res<TileOverlayState>,
    sdf_overlay_state: Res<SdfOverlayState>,
    physics_overlay_state: Res<PhysicsAreaOverlayState>,
    chunk_residency: Res<MpmChunkResidencyState>,
    terrain_world: Res<TerrainWorld>,
    mut labels: Query<&mut Text, With<OverlayInfoText>>,
) {
    if !tile_overlay_state.is_changed()
        && !sdf_overlay_state.is_changed()
        && !physics_overlay_state.is_changed()
        && !chunk_residency.is_changed()
        && !terrain_world.is_changed()
    {
        return;
    }
    let overlay_gpu_enabled = tile_overlay_state.enabled || physics_overlay_state.enabled;
    let gpu_layout = if chunk_residency.initialized {
        chunk_residency.grid_layout
    } else {
        world_grid_layout()
    };
    let gpu_cells = UVec2::new(
        gpu_layout.dims.x.saturating_sub(1),
        gpu_layout.dims.y.saturating_sub(1),
    );
    for mut label in &mut labels {
        let mut lines = Vec::new();
        if overlay_gpu_enabled {
            lines.push(format!(
                "GPU Chunk/Physics Overlay: resident {} chunks, nodes {}x{} cells {}x{}",
                chunk_residency.resident_chunk_count,
                gpu_layout.dims.x,
                gpu_layout.dims.y,
                gpu_cells.x,
                gpu_cells.y
            ));
        }
        if tile_overlay_state.enabled {
            let loaded_chunks = terrain_world.loaded_chunk_coords().len();
            let modified_chunks = terrain_world.override_chunk_coords().len();
            lines.push(format!(
                "Render Chunks: {} (Modified: {})",
                loaded_chunks, modified_chunks
            ));
        }
        if sdf_overlay_state.enabled {
            lines.push("SDF Overlay: per-cell signed distance".to_string());
        }
        label.0 = lines.join("\n");
    }
}
fn toggle_button_bg(enabled: bool, colors: &OverlayColorParams) -> BackgroundColor {
    if enabled {
        colors.button_bg_on.to_color().into()
    } else {
        colors.button_bg_off.to_color().into()
    }
}
