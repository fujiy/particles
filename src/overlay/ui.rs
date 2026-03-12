use bevy::prelude::*;

use super::*;
use crate::params::overlay::{OverlayColor, OverlayColorParams};
use crate::params::{ActiveOverlayParams, ActivePhysicsParams};
use crate::physics::gpu_mpm::gpu_resources::MPM_NODE_SPACING_M;

const MASS_COLORBAR_SEGMENTS: u8 = 16;
const MASS_COLORBAR_PANEL_PADDING_PX: f32 = 8.0;
const MASS_COLORBAR_PANEL_EXTRA_WIDTH_PX: f32 = 44.0;

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
                bottom: px(ui.mass_button_bottom_px),
                padding: UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px)),
                ..default()
            },
            BackgroundColor(colors.button_bg_off.to_color()),
            MassOverlayToggleButton,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Mass Overlay: OFF"),
                TextFont::from_font_size(ui.button_font_size_px),
                TextColor(Color::WHITE),
                MassOverlayToggleButtonLabel,
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

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                right: px(ui.mass_colorbar_right_px),
                bottom: px(ui.mass_colorbar_bottom_px),
                width: px(ui.mass_colorbar_width_px + MASS_COLORBAR_PANEL_EXTRA_WIDTH_PX),
                padding: UiRect::all(px(MASS_COLORBAR_PANEL_PADDING_PX)),
                flex_direction: FlexDirection::Column,
                row_gap: px(4.0),
                align_items: AlignItems::Stretch,
                ..default()
            },
            BackgroundColor(colors.button_bg_off.to_color()),
            Visibility::Hidden,
            MassOverlayColorbarRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Cell Mass [kg]"),
                TextFont::from_font_size(ui.mass_colorbar_font_size_px),
                TextColor(Color::WHITE),
                MassOverlayColorbarTitle,
            ));
            parent.spawn((
                Text::new("0 kg"),
                TextFont::from_font_size(ui.mass_colorbar_font_size_px),
                TextColor(Color::WHITE),
                MassOverlayColorbarTopLabel,
            ));
            parent
                .spawn((
                    Node {
                        width: px(ui.mass_colorbar_width_px),
                        height: px(ui.mass_colorbar_height_px),
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Stretch,
                        ..default()
                    },
                    BackgroundColor(Color::BLACK.with_alpha(0.15)),
                    MassOverlayColorbarBar,
                ))
                .with_children(|bar| {
                    for index in 0..MASS_COLORBAR_SEGMENTS {
                        bar.spawn((
                            Node {
                                width: percent(100.0),
                                flex_grow: 1.0,
                                ..default()
                            },
                            BackgroundColor(Color::NONE),
                            MassOverlayColorbarSegment { index },
                        ));
                    }
                });
            parent.spawn((
                Text::new("0 kg"),
                TextFont::from_font_size(ui.mass_colorbar_font_size_px),
                TextColor(Color::WHITE),
                MassOverlayColorbarBottomLabel,
            ));
        });
}

pub(super) fn apply_overlay_ui_params(
    overlay_params: Res<ActiveOverlayParams>,
    mut nodes: ParamSet<(
        Single<&mut Node, With<TileOverlayToggleButton>>,
        Single<&mut Node, With<SdfOverlayToggleButton>>,
        Single<&mut Node, With<MassOverlayToggleButton>>,
        Single<&mut Node, With<ParticleOverlayToggleButton>>,
    )>,
    mut fonts: ParamSet<(
        Single<&mut TextFont, With<TileOverlayToggleButtonLabel>>,
        Single<&mut TextFont, With<SdfOverlayToggleButtonLabel>>,
        Single<&mut TextFont, With<MassOverlayToggleButtonLabel>>,
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

    let mut mass_node = nodes.p2();
    mass_node.right = px(ui.right_px);
    mass_node.bottom = px(ui.mass_button_bottom_px);
    mass_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    let mut particle_node = nodes.p3();
    particle_node.right = px(ui.right_px);
    particle_node.bottom = px(ui.particle_button_bottom_px);
    particle_node.padding = UiRect::axes(px(ui.button_padding_x_px), px(ui.button_padding_y_px));

    fonts.p0().font_size = ui.button_font_size_px;
    fonts.p1().font_size = ui.button_font_size_px;
    fonts.p2().font_size = ui.button_font_size_px;
    fonts.p3().font_size = ui.button_font_size_px;
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

pub(super) fn handle_mass_overlay_button(
    mut interactions: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<MassOverlayToggleButton>),
    >,
    overlay_params: Res<ActiveOverlayParams>,
    mut overlay_state: ResMut<MassOverlayState>,
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
    mass_overlay_state: Res<MassOverlayState>,
    physics_overlay_state: Res<PhysicsAreaOverlayState>,
    particle_overlay_state: Res<ParticleOverlayState>,
    mut buttons: ParamSet<(
        Query<&mut BackgroundColor, With<TileOverlayToggleButton>>,
        Query<&mut BackgroundColor, With<SdfOverlayToggleButton>>,
        Query<&mut BackgroundColor, With<MassOverlayToggleButton>>,
        Query<&mut BackgroundColor, With<ParticleOverlayToggleButton>>,
    )>,
) {
    if !overlay_params.is_changed()
        && !tile_overlay_state.is_changed()
        && !sdf_overlay_state.is_changed()
        && !mass_overlay_state.is_changed()
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
        *bg = toggle_button_bg(mass_overlay_state.enabled, colors);
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

pub(super) fn update_mass_overlay_button_label(
    overlay_state: Res<MassOverlayState>,
    mut labels: Query<&mut Text, With<MassOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() {
        return;
    }
    for mut label in &mut labels {
        label.0 = if overlay_state.enabled {
            "Mass Overlay: ON".to_string()
        } else {
            "Mass Overlay: OFF".to_string()
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

pub(super) fn sync_mass_overlay_colorbar(
    overlay_params: Res<ActiveOverlayParams>,
    physics_params: Res<ActivePhysicsParams>,
    overlay_state: Res<MassOverlayState>,
    root_query: Single<
        (&mut Node, &mut BackgroundColor, &mut Visibility),
        (With<MassOverlayColorbarRoot>, Without<MassOverlayColorbarSegment>),
    >,
    bar_query: Single<
        &mut Node,
        (With<MassOverlayColorbarBar>, Without<MassOverlayColorbarRoot>),
    >,
    title_query: Single<
        (&mut Text, &mut TextFont),
        (
            With<MassOverlayColorbarTitle>,
            Without<MassOverlayColorbarTopLabel>,
            Without<MassOverlayColorbarBottomLabel>,
        ),
    >,
    top_label_query: Single<
        (&mut Text, &mut TextFont),
        (
            With<MassOverlayColorbarTopLabel>,
            Without<MassOverlayColorbarTitle>,
            Without<MassOverlayColorbarBottomLabel>,
        ),
    >,
    bottom_label_query: Single<
        (&mut Text, &mut TextFont),
        (
            With<MassOverlayColorbarBottomLabel>,
            Without<MassOverlayColorbarTitle>,
            Without<MassOverlayColorbarTopLabel>,
        ),
    >,
    mut segments: Query<
        (&MassOverlayColorbarSegment, &mut BackgroundColor),
        Without<MassOverlayColorbarRoot>,
    >,
) {
    if !overlay_params.is_changed() && !physics_params.is_changed() && !overlay_state.is_changed() {
        return;
    }

    let ui = &overlay_params.0.ui;
    let colors = &overlay_params.0.colors;

    let (mut root_node, mut root_bg, mut visibility) = root_query.into_inner();
    root_node.right = px(ui.mass_colorbar_right_px);
    root_node.bottom = px(ui.mass_colorbar_bottom_px);
    root_node.width = px(ui.mass_colorbar_width_px + MASS_COLORBAR_PANEL_EXTRA_WIDTH_PX);
    root_node.padding = UiRect::all(px(MASS_COLORBAR_PANEL_PADDING_PX));
    root_bg.0 = colors.button_bg_off.to_color();
    *visibility = if overlay_state.enabled {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };

    let mut bar_node = bar_query.into_inner();
    bar_node.width = px(ui.mass_colorbar_width_px);
    bar_node.height = px(ui.mass_colorbar_height_px);

    let max_cell_mass_kg = mass_overlay_max_cell_mass_kg(&overlay_params, &physics_params);

    let (mut title_text, mut title_font) = title_query.into_inner();
    title_text.0 = "Cell Mass [kg]".to_string();
    title_font.font_size = ui.mass_colorbar_font_size_px;

    let (mut top_text, mut top_font) = top_label_query.into_inner();
    top_text.0 = format_mass_label(max_cell_mass_kg);
    top_font.font_size = ui.mass_colorbar_font_size_px;

    let (mut bottom_text, mut bottom_font) = bottom_label_query.into_inner();
    bottom_text.0 = "0 kg".to_string();
    bottom_font.font_size = ui.mass_colorbar_font_size_px;

    for (segment, mut bg) in &mut segments {
        let t = if MASS_COLORBAR_SEGMENTS <= 1 {
            1.0
        } else {
            1.0 - segment.index as f32 / (MASS_COLORBAR_SEGMENTS - 1) as f32
        };
        *bg = mass_overlay_gradient_color(t, colors).to_color().into();
    }
}

fn toggle_button_bg(enabled: bool, colors: &OverlayColorParams) -> BackgroundColor {
    if enabled {
        colors.button_bg_on.to_color().into()
    } else {
        colors.button_bg_off.to_color().into()
    }
}

fn mass_overlay_gradient_color(t: f32, colors: &OverlayColorParams) -> OverlayColor {
    let t = t.clamp(0.0, 1.0);
    if t <= 0.5 {
        lerp_overlay_color(colors.mass_overlay_low, colors.mass_overlay_mid, t * 2.0)
    } else {
        lerp_overlay_color(colors.mass_overlay_mid, colors.mass_overlay_high, (t - 0.5) * 2.0)
    }
}

fn lerp_overlay_color(a: OverlayColor, b: OverlayColor, t: f32) -> OverlayColor {
    let t = t.clamp(0.0, 1.0);
    OverlayColor {
        r: a.r + (b.r - a.r) * t,
        g: a.g + (b.g - a.g) * t,
        b: a.b + (b.b - a.b) * t,
        a: a.a + (b.a - a.a) * t,
    }
}

fn mass_overlay_max_cell_mass_kg(
    overlay_params: &ActiveOverlayParams,
    physics_params: &ActivePhysicsParams,
) -> f32 {
    let cell_size_m = MPM_NODE_SPACING_M * 2.0;
    physics_params.0.water.rho0
        * cell_size_m
        * cell_size_m
        * overlay_params.0.mass.max_ref_cell_mass_scale
}

fn format_mass_label(value_kg: f32) -> String {
    if value_kg >= 100.0 {
        format!("{value_kg:.0} kg")
    } else if value_kg >= 10.0 {
        format!("{value_kg:.1} kg")
    } else {
        format!("{value_kg:.2} kg")
    }
}
