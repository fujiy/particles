use std::collections::HashSet;

use bevy::prelude::*;

use super::*;

pub fn sync_object_sprites_to_render(
    mut commands: Commands,
    mut object_world: ResMut<ObjectWorld>,
    particles: Res<ParticleWorld>,
    mut render_state: ResMut<ObjectRenderState>,
    mut render_diagnostics: ResMut<TerrainRenderDiagnostics>,
    mut images: ResMut<Assets<Image>>,
) {
    let _span = tracing::info_span!("render::sync_object_sprites_to_render").entered();
    if !particles.is_changed() && !object_world.is_changed() {
        return;
    }

    let positions = particles.positions();
    let masses = particles.masses();
    let materials = particles.materials();
    let mut live_ids = Vec::new();
    let mut updated_chunks = HashSet::new();

    {
        let _span = tracing::info_span!("render::object_update_live").entered();
        for object in object_world.objects_mut() {
            let Some((center, theta)) = object_pose(object, positions, masses) else {
                continue;
            };
            live_ids.push(object.id);
            for &index in &object.particle_indices {
                if index < positions.len() {
                    updated_chunks.insert(chunk_coord_from_world(positions[index]));
                }
            }

            let transform = Transform::from_xyz(center.x, center.y, OBJECT_RENDER_Z)
                .with_rotation(Quat::from_rotation_z(theta));
            if let Some(entry) = render_state.object_sprites.get_mut(&object.id) {
                if object.shape_dirty {
                    let (image, world_size) = rasterize_object_image(object, materials);
                    if images.insert(entry.image.id(), image).is_err() {
                        let (new_image, _) = rasterize_object_image(object, materials);
                        entry.image = images.add(new_image);
                    }
                    entry.world_size = world_size;
                    let mut sprite = Sprite::from_image(entry.image.clone());
                    sprite.custom_size = Some(entry.world_size);
                    commands.entity(entry.entity).insert(sprite);
                    object.shape_dirty = false;
                }
                commands.entity(entry.entity).insert(transform);
                continue;
            }

            let (image, world_size) = rasterize_object_image(object, materials);
            let image_handle = images.add(image);
            let mut sprite = Sprite::from_image(image_handle.clone());
            sprite.custom_size = Some(world_size);
            let entity = commands
                .spawn((
                    sprite,
                    transform,
                    ObjectRenderHandle {
                        object_id: object.id,
                    },
                ))
                .id();
            render_state.object_sprites.insert(
                object.id,
                ObjectSprite {
                    entity,
                    image: image_handle,
                    world_size,
                },
            );
            object.shape_dirty = false;
        }
    }

    {
        let _span = tracing::info_span!("render::object_remove_stale").entered();
        let stale_ids: Vec<_> = render_state
            .object_sprites
            .keys()
            .copied()
            .filter(|id| !live_ids.contains(id))
            .collect();
        for object_id in stale_ids {
            if let Some(entry) = render_state.object_sprites.remove(&object_id) {
                commands.entity(entry.entity).despawn();
                images.remove(entry.image.id());
            }
        }
    }
    {
        let _span = tracing::info_span!("render::object_mark_chunks").entered();
        for chunk in updated_chunks {
            mark_particle_chunk_updated(&mut render_diagnostics, chunk);
        }
    }
}

fn object_pose(object: &ObjectData, positions: &[Vec2], masses: &[f32]) -> Option<(Vec2, f32)> {
    if object.particle_indices.is_empty() || object.rest_local.is_empty() {
        return None;
    }

    let mut mass_sum = 0.0;
    let mut center = Vec2::ZERO;
    for &index in &object.particle_indices {
        if index >= positions.len() || index >= masses.len() {
            continue;
        }
        let mass = masses[index];
        mass_sum += mass;
        center += positions[index] * mass;
    }
    if mass_sum <= 1e-6 {
        return None;
    }
    center /= mass_sum;

    let mut a00 = 0.0;
    let mut a01 = 0.0;
    let mut a10 = 0.0;
    let mut a11 = 0.0;
    for (slot, &index) in object.particle_indices.iter().enumerate() {
        if slot >= object.rest_local.len() || index >= positions.len() || index >= masses.len() {
            continue;
        }
        let mass = masses[index];
        let p = positions[index] - center;
        let q = object.rest_local[slot];
        a00 += mass * p.x * q.x;
        a01 += mass * p.x * q.y;
        a10 += mass * p.y * q.x;
        a11 += mass * p.y * q.y;
    }
    let theta = (a10 - a01).atan2(a00 + a11);
    Some((center, theta))
}

fn rasterize_object_image(object: &ObjectData, materials: &[ParticleMaterial]) -> (Image, Vec2) {
    let half_cell = CELL_SIZE_M * 0.5;
    let mut extent = Vec2::splat(half_cell);
    for q in &object.rest_local {
        extent.x = extent.x.max(q.x.abs() + half_cell);
        extent.y = extent.y.max(q.y.abs() + half_cell);
    }

    let world_size = extent * 2.0;
    let width = ((world_size.x * OBJECT_DOT_SCALE).ceil() as u32 + OBJECT_PADDING_PX * 2).max(2);
    let height = ((world_size.y * OBJECT_DOT_SCALE).ceil() as u32 + OBJECT_PADDING_PX * 2).max(2);
    let mut pixels = vec![0_u8; (width * height * 4) as usize];
    let pad = OBJECT_PADDING_PX as f32;
    let seed_material = object
        .particle_indices
        .first()
        .and_then(|&index| materials.get(index))
        .copied()
        .unwrap_or(ParticleMaterial::StoneSolid);
    let palette = cell_palette_for_particle(seed_material);

    for q in &object.rest_local {
        let min_world = *q - Vec2::splat(half_cell);
        let max_world = *q + Vec2::splat(half_cell);
        let min_px = Vec2::new(
            (min_world.x + extent.x) * OBJECT_DOT_SCALE + pad,
            (min_world.y + extent.y) * OBJECT_DOT_SCALE + pad,
        );
        let max_px = Vec2::new(
            (max_world.x + extent.x) * OBJECT_DOT_SCALE + pad,
            (max_world.y + extent.y) * OBJECT_DOT_SCALE + pad,
        );
        let start_x = min_px.x.floor() as i32;
        let end_x = max_px.x.ceil() as i32 - 1;
        let start_y = min_px.y.floor() as i32;
        let end_y = max_px.y.ceil() as i32 - 1;
        let cell_index_x = (q.x / CELL_SIZE_M).round() as i32;
        let cell_index_y = (q.y / CELL_SIZE_M).round() as i32;

        for py in start_y..=end_y {
            if py < 0 || py >= height as i32 {
                continue;
            }
            for px in start_x..=end_x {
                if px < 0 || px >= width as i32 {
                    continue;
                }
                let iy = height as i32 - 1 - py;
                let offset = ((iy as u32 * width + px as u32) * 4) as usize;
                let pattern_x = cell_index_x * CELL_PIXEL_SIZE as i32 + (px - start_x);
                let pattern_y = cell_index_y * CELL_PIXEL_SIZE as i32 + (py - start_y);
                let palette_index = deterministic_palette_index(pattern_x, pattern_y);
                pixels[offset..offset + 4].copy_from_slice(&palette[palette_index]);
            }
        }
    }

    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = Some(pixels);
    let texture_world_size = Vec2::new(
        width as f32 / OBJECT_DOT_SCALE,
        height as f32 / OBJECT_DOT_SCALE,
    );
    (image, texture_world_size)
}
