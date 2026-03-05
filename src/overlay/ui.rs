use bevy::prelude::*;

use super::*;
use crate::physics::gpu_mpm::gpu_resources::world_grid_layout;

pub(super) fn setup_overlay_ui(mut commands: Commands) {
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(12.0),
                bottom: px(TILE_BUTTON_BOTTOM_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(BUTTON_BG_OFF),
            TileOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Chunk Overlay: OFF"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                TileOverlayToggleButtonLabel,
            ));
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(12.0),
                bottom: px(SDF_BUTTON_BOTTOM_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(BUTTON_BG_OFF),
            SdfOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("SDF Overlay: OFF"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                SdfOverlayToggleButtonLabel,
            ));
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(12.0),
                bottom: px(PHYSICS_AREA_BUTTON_BOTTOM_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(BUTTON_BG_OFF),
            PhysicsAreaOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Physics Area Overlay: OFF"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                PhysicsAreaOverlayToggleButtonLabel,
            ));
        });

    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(12.0),
                bottom: px(PARTICLE_BUTTON_BOTTOM_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(BUTTON_BG_OFF),
            ParticleOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Particle Overlay: OFF"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                ParticleOverlayToggleButtonLabel,
            ));
        });

    commands.spawn((
        Text::new(""),
        TextFont::from_font_size(14.0),
        TextColor(Color::srgba(0.96, 0.96, 0.98, 0.95)),
        Node {
            position_type: PositionType::Absolute,
            left: px(12.0),
            top: px(12.0),
            ..default()
        },
        OverlayInfoText,
    ));
}

pub(super) fn handle_particle_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<ParticleOverlayToggleButton>),
    >,
    mut overlay_state: ResMut<ParticleOverlayState>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                overlay_state.enabled = !overlay_state.enabled;
                *bg = BUTTON_BG_PRESS.into();
            }
            Interaction::Hovered => {
                *bg = BUTTON_BG_HOVER.into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(overlay_state.enabled);
            }
        }
    }
}

pub(super) fn handle_tile_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<TileOverlayToggleButton>),
    >,
    mut overlay_state: ResMut<TileOverlayState>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                overlay_state.enabled = !overlay_state.enabled;
                *bg = BUTTON_BG_PRESS.into();
            }
            Interaction::Hovered => {
                *bg = BUTTON_BG_HOVER.into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(overlay_state.enabled);
            }
        }
    }
}

pub(super) fn handle_physics_area_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<PhysicsAreaOverlayToggleButton>),
    >,
    mut overlay_state: ResMut<PhysicsAreaOverlayState>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                overlay_state.enabled = !overlay_state.enabled;
                *bg = BUTTON_BG_PRESS.into();
            }
            Interaction::Hovered => {
                *bg = BUTTON_BG_HOVER.into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(overlay_state.enabled);
            }
        }
    }
}

pub(super) fn handle_sdf_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<SdfOverlayToggleButton>),
    >,
    mut overlay_state: ResMut<SdfOverlayState>,
) {
    for (interaction, mut bg) in &mut interactions {
        match *interaction {
            Interaction::Pressed => {
                overlay_state.enabled = !overlay_state.enabled;
                *bg = BUTTON_BG_PRESS.into();
            }
            Interaction::Hovered => {
                *bg = BUTTON_BG_HOVER.into();
            }
            Interaction::None => {
                *bg = toggle_button_bg(overlay_state.enabled);
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
    if !overlay_state.is_changed()
        && !render_diagnostics.is_changed()
    {
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
fn toggle_button_bg(enabled: bool) -> BackgroundColor {
    if enabled {
        BUTTON_BG_ON.into()
    } else {
        BUTTON_BG_OFF.into()
    }
}
