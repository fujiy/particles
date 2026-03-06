use bevy::prelude::*;

use super::*;
use crate::params::ActiveOverlayParams;
use crate::params::overlay::OverlayColorParams;
use crate::physics::gpu_mpm::gpu_resources::world_grid_layout;

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
    mut tile_node: Single<&mut Node, With<TileOverlayToggleButton>>,
    mut sdf_node: Single<&mut Node, With<SdfOverlayToggleButton>>,
    mut physics_node: Single<&mut Node, With<PhysicsAreaOverlayToggleButton>>,
    mut particle_node: Single<&mut Node, With<ParticleOverlayToggleButton>>,
    mut info_node: Single<&mut Node, With<OverlayInfoText>>,
    mut tile_font: Single<&mut TextFont, With<TileOverlayToggleButtonLabel>>,
    mut sdf_font: Single<&mut TextFont, With<SdfOverlayToggleButtonLabel>>,
    mut physics_font: Single<&mut TextFont, With<PhysicsAreaOverlayToggleButtonLabel>>,
    mut particle_font: Single<&mut TextFont, With<ParticleOverlayToggleButtonLabel>>,
    mut info_font: Single<&mut TextFont, With<OverlayInfoText>>,
) {
    if !overlay_params.is_changed() {
        return;
    }
    let ui = &overlay_params.0.ui;

    tile_node.right = px(ui.right_px);
    tile_node.bottom = px(ui.tile_button_bottom_px);
    tile_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    sdf_node.right = px(ui.right_px);
    sdf_node.bottom = px(ui.sdf_button_bottom_px);
    sdf_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    physics_node.right = px(ui.right_px);
    physics_node.bottom = px(ui.physics_area_button_bottom_px);
    physics_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    particle_node.right = px(ui.right_px);
    particle_node.bottom = px(ui.particle_button_bottom_px);
    particle_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    info_node.left = px(ui.info_left_px);
    info_node.top = px(ui.info_top_px);

    tile_font.font_size = ui.button_font_size_px;
    sdf_font.font_size = ui.button_font_size_px;
    physics_font.font_size = ui.button_font_size_px;
    particle_font.font_size = ui.button_font_size_px;
    info_font.font_size = ui.info_font_size_px;
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

pub(super) fn handle_physics_area_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<PhysicsAreaOverlayToggleButton>),
    >,
    overlay_params: Res<ActiveOverlayParams>,
    mut overlay_state: ResMut<PhysicsAreaOverlayState>,
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
    terrain_world: Res<TerrainWorld>,
    generated_chunk_cache: Res<TerrainGeneratedChunkCache>,
    mut labels: Query<&mut Text, With<TileOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed()
        && !terrain_world.is_changed()
        && !generated_chunk_cache.is_changed()
    {
        return;
    }

    let loaded_chunks = terrain_world.loaded_chunk_coords().len();
    let cached_chunks = generated_chunk_cache.cached_chunk_coords().len();
    let modified_chunks = terrain_world.override_chunk_coords().len();

    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            format!(
                "Chunk Overlay: ON (Loaded:{} Cached:{} Modified:{})",
                loaded_chunks, cached_chunks, modified_chunks,
            )
        } else {
            "Chunk Overlay: OFF".to_string()
        };
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
    overlay_state: Res<PhysicsAreaOverlayState>,
    render_diagnostics: Res<TerrainRenderDiagnostics>,
    mut labels: Query<&mut Text, With<PhysicsAreaOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() && !render_diagnostics.is_changed() {
        return;
    }

    let gpu_layout = world_grid_layout();
    let gpu_cells = UVec2::new(
        gpu_layout.dims.x.saturating_sub(1),
        gpu_layout.dims.y.saturating_sub(1),
    );
    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            format!(
                "Physics Area Overlay: ON (T:{} P:{} GPU:{}x{} nodes {}x{} cells)",
                render_diagnostics
                    .terrain_updated_chunk_highlight_frames
                    .len(),
                render_diagnostics
                    .particle_updated_chunk_highlight_frames
                    .len(),
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
    terrain_world: Res<TerrainWorld>,
    mut labels: Query<&mut Text, With<OverlayInfoText>>,
) {
    if !tile_overlay_state.is_changed()
        && !sdf_overlay_state.is_changed()
        && !physics_overlay_state.is_changed()
        && !terrain_world.is_changed()
    {
        return;
    }
    let gpu_layout = world_grid_layout();
    let gpu_cells = UVec2::new(
        gpu_layout.dims.x.saturating_sub(1),
        gpu_layout.dims.y.saturating_sub(1),
    );
    for mut label in &mut labels {
        let mut lines = Vec::new();
        if physics_overlay_state.enabled {
            lines.push(format!(
                "GPU Grid: nodes {}x{} cells {}x{} (single uniform grid, no tiles)",
                gpu_layout.dims.x, gpu_layout.dims.y, gpu_cells.x, gpu_cells.y
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
