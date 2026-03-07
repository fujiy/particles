use bevy::prelude::*;

use super::*;
use crate::params::ActiveOverlayParams;
use crate::params::overlay::OverlayColorParams;

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
                Text::new("Chunk/Grid Overlay: OFF"),
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
}

pub(super) fn apply_overlay_ui_params(
    overlay_params: Res<ActiveOverlayParams>,
    mut nodes: ParamSet<(
        Single<&mut Node, With<TileOverlayToggleButton>>,
        Single<&mut Node, With<SdfOverlayToggleButton>>,
        Single<&mut Node, With<ParticleOverlayToggleButton>>,
    )>,
    mut fonts: ParamSet<(
        Single<&mut TextFont, With<TileOverlayToggleButtonLabel>>,
        Single<&mut TextFont, With<SdfOverlayToggleButtonLabel>>,
        Single<&mut TextFont, With<ParticleOverlayToggleButtonLabel>>,
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

    let mut particle_node = nodes.p2();
    particle_node.right = px(ui.right_px);
    particle_node.bottom = px(ui.particle_button_bottom_px);
    particle_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    fonts.p0().font_size = ui.button_font_size_px;
    fonts.p1().font_size = ui.button_font_size_px;
    fonts.p2().font_size = ui.button_font_size_px;
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
    mut labels: Query<&mut Text, With<TileOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() && !physics_overlay_state.is_changed() {
        return;
    }

    let enabled = overlay_state.enabled || physics_overlay_state.enabled;
    for mut label in &mut labels {
        label.0 = if enabled {
            "Chunk/Grid Overlay: ON".to_string()
        } else {
            "Chunk/Grid Overlay: OFF".to_string()
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
    for mut bg in &mut buttons.p1() {
        *bg = toggle_button_bg(sdf_overlay_state.enabled, colors);
    }
    for mut bg in &mut buttons.p2() {
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
fn toggle_button_bg(enabled: bool, colors: &OverlayColorParams) -> BackgroundColor {
    if enabled {
        colors.button_bg_on.to_color().into()
    } else {
        colors.button_bg_off.to_color().into()
    }
}
