use bevy::prelude::*;

use super::*;
use crate::physics::world::grid::GridHierarchy;

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
                Text::new("Tile Overlay: OFF"),
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
    render_diagnostics: Res<TerrainRenderDiagnostics>,
    mut labels: Query<&mut Text, With<TileOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed() && !render_diagnostics.is_changed() {
        return;
    }

    for mut label in &mut labels {
        let lod_tile_count = render_diagnostics
            .visible_tiles
            .iter()
            .filter(|tile| tile.span_chunks > 1)
            .count();
        label.0 = if overlay_state.enabled {
            format!(
                "Tile Overlay: ON (Tiles:{} LOD:{})",
                render_diagnostics.visible_tiles.len(),
                lod_tile_count,
            )
        } else {
            "Tile Overlay: OFF".to_string()
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
    active_region: Res<PhysicsActiveRegion>,
    particle_world: Res<ParticleWorld>,
    grid_hierarchy: Res<GridHierarchy>,
    mut labels: Query<&mut Text, With<PhysicsAreaOverlayToggleButtonLabel>>,
) {
    if !overlay_state.is_changed()
        && !render_diagnostics.is_changed()
        && !active_region.is_changed()
        && !particle_world.is_changed()
        && !grid_hierarchy.is_changed()
    {
        return;
    }

    for mut label in &mut labels {
        let sub_block_count = particle_world.sub_block_overlay_samples().len();
        let max_debt = particle_world
            .sub_block_overlay_samples()
            .iter()
            .map(|sample| sample.debt_ratio)
            .fold(0.0, f32::max);
        let mpm_block_count = grid_hierarchy.block_count();
        let mpm_active_nodes: usize = grid_hierarchy
            .blocks()
            .iter()
            .map(|block| block.active_node_count())
            .sum();
        let mpm_color_count = grid_hierarchy.block_color_count();
        label.0 = if overlay_state.enabled {
            format!(
                "Physics Area Overlay: ON (A:{} T:{} P:{} SB:{} D:{:.2} MPM:{} AN:{} C:{})",
                active_region.active_chunks.len(),
                render_diagnostics
                    .terrain_updated_chunk_highlight_frames
                    .len(),
                render_diagnostics
                    .particle_updated_chunk_highlight_frames
                    .len(),
                sub_block_count,
                max_debt,
                mpm_block_count,
                mpm_active_nodes,
                mpm_color_count,
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
    active_region: Res<PhysicsActiveRegion>,
    render_diagnostics: Res<TerrainRenderDiagnostics>,
    particle_world: Res<ParticleWorld>,
    grid_hierarchy: Res<GridHierarchy>,
    mut labels: Query<&mut Text, With<OverlayInfoText>>,
) {
    if !tile_overlay_state.is_changed()
        && !sdf_overlay_state.is_changed()
        && !physics_overlay_state.is_changed()
        && !active_region.is_changed()
        && !render_diagnostics.is_changed()
        && !particle_world.is_changed()
        && !grid_hierarchy.is_changed()
    {
        return;
    }
    for mut label in &mut labels {
        let mut lines = Vec::new();
        if physics_overlay_state.enabled {
            lines.push(format!(
                "Physics Chunks: {}",
                active_region.active_chunks.len()
            ));
            let max_debt = particle_world
                .sub_block_overlay_samples()
                .iter()
                .map(|sample| sample.debt_ratio)
                .fold(0.0, f32::max);
            lines.push(format!(
                "Sub-blocks: {} (Debt max {:.2})",
                particle_world.sub_block_overlay_samples().len(),
                max_debt,
            ));
            let mpm_block_count = grid_hierarchy.block_count();
            let mpm_active_nodes: usize = grid_hierarchy
                .blocks()
                .iter()
                .map(|block| block.active_node_count())
                .sum();
            let mpm_color_count = grid_hierarchy.block_color_count();
            lines.push(format!(
                "MPM Grid: blocks {} active nodes {} colors {}",
                mpm_block_count, mpm_active_nodes, mpm_color_count
            ));
        }
        if tile_overlay_state.enabled {
            let lod_tile_count = render_diagnostics
                .visible_tiles
                .iter()
                .filter(|tile| tile.span_chunks > 1)
                .count();
            lines.push(format!(
                "Render Tiles: {} (LOD: {})",
                render_diagnostics.visible_tiles.len(),
                lod_tile_count
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
