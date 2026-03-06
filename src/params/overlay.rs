// assets/params/overlay.ron の Asset 型定義とローダー。
//
// Overlay UI 配置と SDF overlay 描画スケールを実行時に調整する。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OverlayColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl OverlayColor {
    pub fn to_color(self) -> Color {
        Color::srgba(self.r, self.g, self.b, self.a)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlaySdfParams {
    /// SDF 等値線サンプル間隔 [セル倍率]. 許容: 0.1 – 8.0
    pub step_cell_scale: f32,
    /// SDF 点描半径 [セル倍率]. 許容: 0.05 – 2.0
    pub cell_radius_scale: f32,
    /// SDF 表示レンジ [セル倍率]. 許容: 0.5 – 30.0
    pub range_cell_scale: f32,
    /// 地形内部塗り潰し Z. 許容: 0.0 – 2.0
    pub negative_fill_z: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlayUiParams {
    /// ボタン共通の右余白 [px]. 許容: 0.0 – 200.0
    pub right_px: f32,
    /// Chunk ボタン下端 [px]. 許容: 0.0 – 400.0
    pub tile_button_bottom_px: f32,
    /// SDF ボタン下端 [px]. 許容: 0.0 – 400.0
    pub sdf_button_bottom_px: f32,
    /// Physics Area ボタン下端 [px]. 許容: 0.0 – 400.0
    pub physics_area_button_bottom_px: f32,
    /// Particle ボタン下端 [px]. 許容: 0.0 – 400.0
    pub particle_button_bottom_px: f32,
    /// ボタン横パディング [px]. 許容: 0.0 – 40.0
    pub button_padding_x_px: f32,
    /// ボタン縦パディング [px]. 許容: 0.0 – 24.0
    pub button_padding_y_px: f32,
    /// ボタンラベルフォントサイズ [px]. 許容: 8.0 – 40.0
    pub button_font_size_px: f32,
    /// 情報ラベル左余白 [px]. 許容: 0.0 – 400.0
    pub info_left_px: f32,
    /// 情報ラベル上余白 [px]. 許容: 0.0 – 400.0
    pub info_top_px: f32,
    /// 情報ラベルフォントサイズ [px]. 許容: 8.0 – 40.0
    pub info_font_size_px: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlayColorParams {
    pub button_bg_off: OverlayColor,
    pub button_bg_on: OverlayColor,
    pub button_bg_hover: OverlayColor,
    pub button_bg_press: OverlayColor,
    pub grid_chunk_boundary: OverlayColor,
    pub grid_cached_chunk: OverlayColor,
    pub grid_modified_chunk: OverlayColor,
    pub grid_terrain_updated: OverlayColor,
    pub grid_particle_updated: OverlayColor,
}

#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct OverlayParams {
    pub sdf: OverlaySdfParams,
    pub ui: OverlayUiParams,
    pub colors: OverlayColorParams,
}

impl Default for OverlayParams {
    fn default() -> Self {
        Self {
            sdf: OverlaySdfParams {
                step_cell_scale: 1.0,
                cell_radius_scale: 0.45,
                range_cell_scale: 6.0,
                negative_fill_z: 0.2,
            },
            ui: OverlayUiParams {
                right_px: 12.0,
                tile_button_bottom_px: 88.0,
                sdf_button_bottom_px: 126.0,
                physics_area_button_bottom_px: 50.0,
                particle_button_bottom_px: 12.0,
                button_padding_x_px: 10.0,
                button_padding_y_px: 6.0,
                button_font_size_px: 14.0,
                info_left_px: 12.0,
                info_top_px: 12.0,
                info_font_size_px: 14.0,
            },
            colors: OverlayColorParams {
                button_bg_off: OverlayColor {
                    r: 0.17,
                    g: 0.18,
                    b: 0.22,
                    a: 0.95,
                },
                button_bg_on: OverlayColor {
                    r: 0.16,
                    g: 0.30,
                    b: 0.46,
                    a: 0.95,
                },
                button_bg_hover: OverlayColor {
                    r: 0.24,
                    g: 0.25,
                    b: 0.30,
                    a: 0.98,
                },
                button_bg_press: OverlayColor {
                    r: 0.38,
                    g: 0.40,
                    b: 0.48,
                    a: 0.98,
                },
                grid_chunk_boundary: OverlayColor {
                    r: 0.80,
                    g: 0.86,
                    b: 0.93,
                    a: 0.22,
                },
                grid_cached_chunk: OverlayColor {
                    r: 0.32,
                    g: 0.86,
                    b: 0.98,
                    a: 0.62,
                },
                grid_modified_chunk: OverlayColor {
                    r: 0.99,
                    g: 0.43,
                    b: 0.22,
                    a: 0.95,
                },
                grid_terrain_updated: OverlayColor {
                    r: 0.13,
                    g: 0.85,
                    b: 0.92,
                    a: 1.00,
                },
                grid_particle_updated: OverlayColor {
                    r: 0.76,
                    g: 0.56,
                    b: 0.98,
                    a: 1.00,
                },
            },
        }
    }
}

impl OverlayParams {
    pub fn validate(&self) -> Result<(), String> {
        macro_rules! check {
            ($val:expr, $name:expr, $lo:expr, $hi:expr) => {
                if !($val >= $lo && $val <= $hi) {
                    return Err(format!(
                        "{}: {} は [{}, {}] の範囲外",
                        $name, $val, $lo, $hi
                    ));
                }
            };
        }
        check!(self.sdf.step_cell_scale, "sdf.step_cell_scale", 0.1, 8.0);
        check!(
            self.sdf.cell_radius_scale,
            "sdf.cell_radius_scale",
            0.05,
            2.0
        );
        check!(self.sdf.range_cell_scale, "sdf.range_cell_scale", 0.5, 30.0);
        check!(self.sdf.negative_fill_z, "sdf.negative_fill_z", 0.0, 2.0);

        check!(self.ui.right_px, "ui.right_px", 0.0, 200.0);
        check!(
            self.ui.tile_button_bottom_px,
            "ui.tile_button_bottom_px",
            0.0,
            400.0
        );
        check!(
            self.ui.sdf_button_bottom_px,
            "ui.sdf_button_bottom_px",
            0.0,
            400.0
        );
        check!(
            self.ui.physics_area_button_bottom_px,
            "ui.physics_area_button_bottom_px",
            0.0,
            400.0
        );
        check!(
            self.ui.particle_button_bottom_px,
            "ui.particle_button_bottom_px",
            0.0,
            400.0
        );
        check!(
            self.ui.button_padding_x_px,
            "ui.button_padding_x_px",
            0.0,
            40.0
        );
        check!(
            self.ui.button_padding_y_px,
            "ui.button_padding_y_px",
            0.0,
            24.0
        );
        check!(
            self.ui.button_font_size_px,
            "ui.button_font_size_px",
            8.0,
            40.0
        );
        check!(self.ui.info_left_px, "ui.info_left_px", 0.0, 400.0);
        check!(self.ui.info_top_px, "ui.info_top_px", 0.0, 400.0);
        check!(self.ui.info_font_size_px, "ui.info_font_size_px", 8.0, 40.0);
        let color_values = [
            ("colors.button_bg_off", self.colors.button_bg_off),
            ("colors.button_bg_on", self.colors.button_bg_on),
            ("colors.button_bg_hover", self.colors.button_bg_hover),
            ("colors.button_bg_press", self.colors.button_bg_press),
            (
                "colors.grid_chunk_boundary",
                self.colors.grid_chunk_boundary,
            ),
            ("colors.grid_cached_chunk", self.colors.grid_cached_chunk),
            (
                "colors.grid_modified_chunk",
                self.colors.grid_modified_chunk,
            ),
            (
                "colors.grid_terrain_updated",
                self.colors.grid_terrain_updated,
            ),
            (
                "colors.grid_particle_updated",
                self.colors.grid_particle_updated,
            ),
        ];
        for (name, color) in color_values {
            for (channel, value) in [
                ("r", color.r),
                ("g", color.g),
                ("b", color.b),
                ("a", color.a),
            ] {
                if !(0.0..=1.0).contains(&value) {
                    return Err(format!("{name}.{channel}: {value} は [0, 1] の範囲外"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Default, bevy::reflect::TypePath)]
pub struct OverlayParamsLoader;

impl AssetLoader for OverlayParamsLoader {
    type Asset = OverlayParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<OverlayParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: OverlayParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
