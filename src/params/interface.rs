// assets/params/interface.ron の Asset 型定義とローダー。
//
// Interface の色・レイアウト・ツール操作パラメータを実行時調整する。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use crate::params::color_serde::{
    Rgba8Color, RgbaColor, deserialize_rgba_color_from_ron, deserialize_u8_color_from_ron,
};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct UiColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl From<RgbaColor> for UiColor {
    fn from(color: RgbaColor) -> Self {
        Self {
            r: color.r,
            g: color.g,
            b: color.b,
            a: color.a,
        }
    }
}

impl<'de> Deserialize<'de> for UiColor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(deserialize_rgba_color_from_ron(deserializer)?.into())
    }
}

impl UiColor {
    pub fn to_color(self) -> Color {
        Color::srgba(self.r, self.g, self.b, self.a)
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct UiColor8 {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl From<Rgba8Color> for UiColor8 {
    fn from(color: Rgba8Color) -> Self {
        Self {
            r: color.r,
            g: color.g,
            b: color.b,
            a: color.a,
        }
    }
}

impl<'de> Deserialize<'de> for UiColor8 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(deserialize_u8_color_from_ron(deserializer)?.into())
    }
}

impl UiColor8 {
    pub fn as_rgba(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceColorParams {
    pub hud_bg: UiColor,
    pub button_bg_off: UiColor,
    pub button_bg_on: UiColor,
    pub button_bg_hover: UiColor,
    pub button_bg_press: UiColor,
    pub button_border_off: UiColor,
    pub button_border_on: UiColor,
    pub toolbar_bg: UiColor,
    pub tooltip_bg: UiColor,
    pub dialog_bg: UiColor,
    pub hover_highlight_terrain: UiColor,
    pub hover_highlight_break: UiColor,
    pub hover_highlight_delete: UiColor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceLayoutParams {
    pub toolbar_bottom_px: f32,
    pub toolbar_button_size_px: f32,
    pub tooltip_cursor_offset_x_px: f32,
    pub tooltip_cursor_offset_y_px: f32,
    pub tooltip_global_z_index: i32,
    pub save_load_bar_top_px: f32,
    pub save_load_bar_right_px: f32,
    pub save_load_button_width_px: f32,
    pub save_load_button_height_px: f32,
    pub dialog_width_px: f32,
    pub dialog_name_input_height_px: f32,
    pub dialog_slot_list_max_height_px: f32,
    pub dialog_slot_button_height_px: f32,
    pub test_assert_panel_top_px: f32,
    pub test_assert_panel_right_px: f32,
    pub test_assert_panel_width_px: f32,
    pub hud_panel_width_px: f32,
    pub scale_bar_left_px: f32,
    pub scale_bar_bottom_px: f32,
    pub scale_bar_target_width_px: f32,
    pub scale_bar_height_px: f32,
    pub scale_bar_min_width_px: f32,
    pub scale_bar_label_font_px: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceBehaviorParams {
    pub tool_stroke_step_cell_scale: f32,
    pub tool_break_brush_radius_cell_scale: f32,
    pub hud_fps_window_sec: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceProfilerParams {
    pub update_hz: f32,
    pub bar_width_px: f32,
    pub bar_height_px: f32,
    pub lane_gap_px: f32,
    pub label_width_px: f32,
    pub lane_label_font_px: f32,
    pub min_segment_ms_per_sec: f32,
    pub max_segments_per_lane: u32,
    pub over_budget_headroom: f32,
    pub colors: InterfaceProfilerColorParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceProfilerColorParams {
    pub bar_bg: UiColor,
    pub physics: UiColor,
    pub render: UiColor,
    pub others: UiColor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceIconPaletteParams {
    pub water: [UiColor8; 4],
    pub stone: [UiColor8; 4],
    pub sand: [UiColor8; 4],
    pub soil: [UiColor8; 4],
}

#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceParams {
    pub colors: InterfaceColorParams,
    pub layout: InterfaceLayoutParams,
    pub behavior: InterfaceBehaviorParams,
    pub profiler: InterfaceProfilerParams,
    pub icon_palette: InterfaceIconPaletteParams,
}

impl Default for InterfaceParams {
    fn default() -> Self {
        Self {
            colors: InterfaceColorParams {
                hud_bg: UiColor {
                    r: 0.05,
                    g: 0.06,
                    b: 0.09,
                    a: 0.82,
                },
                button_bg_off: UiColor {
                    r: 0.17,
                    g: 0.18,
                    b: 0.22,
                    a: 0.95,
                },
                button_bg_on: UiColor {
                    r: 0.16,
                    g: 0.30,
                    b: 0.46,
                    a: 0.95,
                },
                button_bg_hover: UiColor {
                    r: 0.24,
                    g: 0.25,
                    b: 0.30,
                    a: 0.98,
                },
                button_bg_press: UiColor {
                    r: 0.38,
                    g: 0.40,
                    b: 0.48,
                    a: 0.98,
                },
                button_border_off: UiColor {
                    r: 0.08,
                    g: 0.10,
                    b: 0.14,
                    a: 1.0,
                },
                button_border_on: UiColor {
                    r: 0.80,
                    g: 0.92,
                    b: 1.00,
                    a: 1.0,
                },
                toolbar_bg: UiColor {
                    r: 0.05,
                    g: 0.06,
                    b: 0.09,
                    a: 0.88,
                },
                tooltip_bg: UiColor {
                    r: 0.04,
                    g: 0.05,
                    b: 0.08,
                    a: 0.96,
                },
                dialog_bg: UiColor {
                    r: 0.06,
                    g: 0.07,
                    b: 0.10,
                    a: 0.97,
                },
                hover_highlight_terrain: UiColor {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 0.96,
                },
                hover_highlight_break: UiColor {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 0.96,
                },
                hover_highlight_delete: UiColor {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 0.96,
                },
            },
            layout: InterfaceLayoutParams {
                toolbar_bottom_px: 12.0,
                toolbar_button_size_px: 44.0,
                tooltip_cursor_offset_x_px: 14.0,
                tooltip_cursor_offset_y_px: 20.0,
                tooltip_global_z_index: 10_000,
                save_load_bar_top_px: 10.0,
                save_load_bar_right_px: 10.0,
                save_load_button_width_px: 84.0,
                save_load_button_height_px: 34.0,
                dialog_width_px: 380.0,
                dialog_name_input_height_px: 34.0,
                dialog_slot_list_max_height_px: 240.0,
                dialog_slot_button_height_px: 30.0,
                test_assert_panel_top_px: 260.0,
                test_assert_panel_right_px: 10.0,
                test_assert_panel_width_px: 360.0,
                hud_panel_width_px: 360.0,
                scale_bar_left_px: 14.0,
                scale_bar_bottom_px: 18.0,
                scale_bar_target_width_px: 140.0,
                scale_bar_height_px: 4.0,
                scale_bar_min_width_px: 24.0,
                scale_bar_label_font_px: 13.0,
            },
            behavior: InterfaceBehaviorParams {
                tool_stroke_step_cell_scale: 0.5,
                tool_break_brush_radius_cell_scale: 0.5,
                hud_fps_window_sec: 1.0,
            },
            profiler: InterfaceProfilerParams {
                update_hz: 10.0,
                bar_width_px: 240.0,
                bar_height_px: 12.0,
                lane_gap_px: 6.0,
                label_width_px: 30.0,
                lane_label_font_px: 12.0,
                min_segment_ms_per_sec: 2.0,
                max_segments_per_lane: 10,
                over_budget_headroom: 1.1,
                colors: InterfaceProfilerColorParams {
                    bar_bg: UiColor {
                        r: 0.10,
                        g: 0.12,
                        b: 0.16,
                        a: 0.90,
                    },
                    physics: UiColor {
                        r: 0.24,
                        g: 0.55,
                        b: 0.92,
                        a: 0.95,
                    },
                    render: UiColor {
                        r: 0.93,
                        g: 0.76,
                        b: 0.18,
                        a: 0.95,
                    },
                    others: UiColor {
                        r: 0.62,
                        g: 0.66,
                        b: 0.73,
                        a: 0.95,
                    },
                },
            },
            icon_palette: InterfaceIconPaletteParams {
                water: [
                    UiColor8 {
                        r: 42,
                        g: 120,
                        b: 202,
                        a: 235,
                    },
                    UiColor8 {
                        r: 52,
                        g: 136,
                        b: 218,
                        a: 240,
                    },
                    UiColor8 {
                        r: 65,
                        g: 152,
                        b: 228,
                        a: 245,
                    },
                    UiColor8 {
                        r: 78,
                        g: 167,
                        b: 238,
                        a: 250,
                    },
                ],
                stone: [
                    UiColor8 {
                        r: 70,
                        g: 67,
                        b: 63,
                        a: 255,
                    },
                    UiColor8 {
                        r: 83,
                        g: 79,
                        b: 74,
                        a: 255,
                    },
                    UiColor8 {
                        r: 95,
                        g: 90,
                        b: 84,
                        a: 255,
                    },
                    UiColor8 {
                        r: 108,
                        g: 103,
                        b: 96,
                        a: 255,
                    },
                ],
                sand: [
                    UiColor8 {
                        r: 172,
                        g: 149,
                        b: 111,
                        a: 255,
                    },
                    UiColor8 {
                        r: 185,
                        g: 162,
                        b: 124,
                        a: 255,
                    },
                    UiColor8 {
                        r: 198,
                        g: 175,
                        b: 136,
                        a: 255,
                    },
                    UiColor8 {
                        r: 210,
                        g: 188,
                        b: 148,
                        a: 255,
                    },
                ],
                soil: [
                    UiColor8 {
                        r: 105,
                        g: 79,
                        b: 56,
                        a: 255,
                    },
                    UiColor8 {
                        r: 119,
                        g: 91,
                        b: 67,
                        a: 255,
                    },
                    UiColor8 {
                        r: 133,
                        g: 103,
                        b: 78,
                        a: 255,
                    },
                    UiColor8 {
                        r: 147,
                        g: 115,
                        b: 88,
                        a: 255,
                    },
                ],
            },
        }
    }
}

impl InterfaceParams {
    pub fn validate(&self) -> Result<(), String> {
        fn check_color(name: &str, c: UiColor) -> Result<(), String> {
            let vals = [("r", c.r), ("g", c.g), ("b", c.b), ("a", c.a)];
            for (label, value) in vals {
                if !(0.0..=1.0).contains(&value) {
                    return Err(format!("{name}.{label}: {value} は [0, 1] の範囲外"));
                }
            }
            Ok(())
        }
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

        let colors = &self.colors;
        check_color("colors.hud_bg", colors.hud_bg)?;
        check_color("colors.button_bg_off", colors.button_bg_off)?;
        check_color("colors.button_bg_on", colors.button_bg_on)?;
        check_color("colors.button_bg_hover", colors.button_bg_hover)?;
        check_color("colors.button_bg_press", colors.button_bg_press)?;
        check_color("colors.button_border_off", colors.button_border_off)?;
        check_color("colors.button_border_on", colors.button_border_on)?;
        check_color("colors.toolbar_bg", colors.toolbar_bg)?;
        check_color("colors.tooltip_bg", colors.tooltip_bg)?;
        check_color("colors.dialog_bg", colors.dialog_bg)?;
        check_color(
            "colors.hover_highlight_terrain",
            colors.hover_highlight_terrain,
        )?;
        check_color("colors.hover_highlight_break", colors.hover_highlight_break)?;
        check_color(
            "colors.hover_highlight_delete",
            colors.hover_highlight_delete,
        )?;

        let l = &self.layout;
        check!(l.toolbar_bottom_px, "layout.toolbar_bottom_px", 0.0, 400.0);
        check!(
            l.toolbar_button_size_px,
            "layout.toolbar_button_size_px",
            16.0,
            200.0
        );
        check!(
            l.tooltip_cursor_offset_x_px,
            "layout.tooltip_cursor_offset_x_px",
            -200.0,
            200.0
        );
        check!(
            l.tooltip_cursor_offset_y_px,
            "layout.tooltip_cursor_offset_y_px",
            -200.0,
            200.0
        );
        if !(0..=200_000).contains(&l.tooltip_global_z_index) {
            return Err(format!(
                "layout.tooltip_global_z_index: {} は [0, 200000] の範囲外",
                l.tooltip_global_z_index
            ));
        }
        check!(
            l.save_load_bar_top_px,
            "layout.save_load_bar_top_px",
            0.0,
            400.0
        );
        check!(
            l.save_load_bar_right_px,
            "layout.save_load_bar_right_px",
            0.0,
            400.0
        );
        check!(
            l.save_load_button_width_px,
            "layout.save_load_button_width_px",
            24.0,
            400.0
        );
        check!(
            l.save_load_button_height_px,
            "layout.save_load_button_height_px",
            20.0,
            200.0
        );
        check!(l.dialog_width_px, "layout.dialog_width_px", 120.0, 1000.0);
        check!(
            l.dialog_name_input_height_px,
            "layout.dialog_name_input_height_px",
            20.0,
            120.0
        );
        check!(
            l.dialog_slot_list_max_height_px,
            "layout.dialog_slot_list_max_height_px",
            60.0,
            800.0
        );
        check!(
            l.dialog_slot_button_height_px,
            "layout.dialog_slot_button_height_px",
            18.0,
            120.0
        );
        check!(
            l.test_assert_panel_top_px,
            "layout.test_assert_panel_top_px",
            0.0,
            1200.0
        );
        check!(
            l.test_assert_panel_right_px,
            "layout.test_assert_panel_right_px",
            0.0,
            400.0
        );
        check!(
            l.test_assert_panel_width_px,
            "layout.test_assert_panel_width_px",
            120.0,
            1000.0
        );
        check!(
            l.hud_panel_width_px,
            "layout.hud_panel_width_px",
            120.0,
            1000.0
        );
        check!(l.scale_bar_left_px, "layout.scale_bar_left_px", 0.0, 400.0);
        check!(
            l.scale_bar_bottom_px,
            "layout.scale_bar_bottom_px",
            0.0,
            400.0
        );
        check!(
            l.scale_bar_target_width_px,
            "layout.scale_bar_target_width_px",
            20.0,
            600.0
        );
        check!(
            l.scale_bar_height_px,
            "layout.scale_bar_height_px",
            1.0,
            40.0
        );
        check!(
            l.scale_bar_min_width_px,
            "layout.scale_bar_min_width_px",
            1.0,
            200.0
        );
        check!(
            l.scale_bar_label_font_px,
            "layout.scale_bar_label_font_px",
            6.0,
            48.0
        );

        let b = &self.behavior;
        check!(
            b.tool_stroke_step_cell_scale,
            "behavior.tool_stroke_step_cell_scale",
            0.05,
            5.0
        );
        check!(
            b.tool_break_brush_radius_cell_scale,
            "behavior.tool_break_brush_radius_cell_scale",
            0.05,
            10.0
        );
        check!(
            b.hud_fps_window_sec,
            "behavior.hud_fps_window_sec",
            0.1,
            10.0
        );

        let p = &self.profiler;
        check!(p.update_hz, "profiler.update_hz", 1.0, 60.0);
        check!(p.bar_width_px, "profiler.bar_width_px", 80.0, 800.0);
        check!(p.bar_height_px, "profiler.bar_height_px", 4.0, 48.0);
        check!(p.lane_gap_px, "profiler.lane_gap_px", 0.0, 48.0);
        check!(p.label_width_px, "profiler.label_width_px", 16.0, 120.0);
        check!(
            p.lane_label_font_px,
            "profiler.lane_label_font_px",
            6.0,
            32.0
        );
        check!(
            p.min_segment_ms_per_sec,
            "profiler.min_segment_ms_per_sec",
            0.0,
            500.0
        );
        if !(1..=32).contains(&p.max_segments_per_lane) {
            return Err(format!(
                "profiler.max_segments_per_lane: {} は [1, 32] の範囲外",
                p.max_segments_per_lane
            ));
        }
        check!(
            p.over_budget_headroom,
            "profiler.over_budget_headroom",
            1.0,
            4.0
        );
        check_color("profiler.colors.bar_bg", p.colors.bar_bg)?;
        check_color("profiler.colors.physics", p.colors.physics)?;
        check_color("profiler.colors.render", p.colors.render)?;
        check_color("profiler.colors.others", p.colors.others)?;
        Ok(())
    }
}

#[derive(Default, bevy::reflect::TypePath)]
pub struct InterfaceParamsLoader;

impl AssetLoader for InterfaceParamsLoader {
    type Asset = InterfaceParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<InterfaceParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: InterfaceParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
