use bevy::prelude::*;

use super::*;

pub(super) fn setup_overlay_ui(mut commands: Commands) {
    commands
        .spawn((
            Button,
            Node {
                position_type: PositionType::Absolute,
                right: px(12.0),
                bottom: px(GRID_BUTTON_BOTTOM_PX),
                padding: UiRect::axes(px(10.0), px(6.0)),
                ..default()
            },
            BackgroundColor(BUTTON_BG_OFF),
            GridOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Grid Overlay: OFF"),
                TextFont::from_font_size(14.0),
                TextColor(Color::WHITE),
                GridOverlayToggleButtonLabel,
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
        GridOverlayInfoText,
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

pub(super) fn handle_grid_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<GridOverlayToggleButton>),
    >,
    mut overlay_state: ResMut<GridOverlayState>,
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

pub(super) fn update_grid_overlay_button_label(
    overlay_state: Res<GridOverlayState>,
    render_diagnostics: Res<TerrainRenderDiagnostics>,
    mut labels: Query<&mut Text, With<GridOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() && !render_diagnostics.is_changed() {
        return;
    }

    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            format!(
                "Grid Overlay: ON (T:{} P:{} L:{})",
                render_diagnostics
                    .terrain_updated_chunk_highlight_frames
                    .len(),
                render_diagnostics
                    .particle_updated_chunk_highlight_frames
                    .len(),
                render_diagnostics.lod_visible_tiles.len(),
            )
        } else {
            "Grid Overlay: OFF".to_string()
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

pub(super) fn update_grid_overlay_info_text(
    overlay_state: Res<GridOverlayState>,
    active_region: Res<PhysicsActiveRegion>,
    mut labels: Query<&mut Text, With<GridOverlayInfoText>>,
) {
    if !overlay_state.is_changed() && !active_region.is_changed() {
        return;
    }
    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            format!("Physics Chunks: {}", active_region.active_chunks.len())
        } else {
            String::new()
        };
    }
}
fn toggle_button_bg(enabled: bool) -> BackgroundColor {
    if enabled {
        BUTTON_BG_ON.into()
    } else {
        BUTTON_BG_OFF.into()
    }
}
