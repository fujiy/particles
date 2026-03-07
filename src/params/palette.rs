// assets/params/palette.ron の Asset 型定義とローダー。
//
// 水ドット描画で使う材質別パレットを管理する。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use crate::params::color_serde::{Rgba8Color, deserialize_u8_color_from_ron};

/// RGB8カラー（sRGB）。
#[derive(Debug, Clone, Copy, Serialize)]
pub struct PaletteColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl From<Rgba8Color> for PaletteColor {
    fn from(color: Rgba8Color) -> Self {
        Self {
            r: color.r,
            g: color.g,
            b: color.b,
        }
    }
}

impl<'de> Deserialize<'de> for PaletteColor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(deserialize_u8_color_from_ron(deserializer)?.into())
    }
}

/// 4色パレット。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialPalette4 {
    pub colors: [PaletteColor; 4],
}

/// 水ドット用パレット定義。
#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct PaletteParams {
    pub water: MaterialPalette4,
    pub soil: MaterialPalette4,
    pub sand: MaterialPalette4,
}

impl Default for PaletteParams {
    fn default() -> Self {
        Self {
            water: MaterialPalette4 {
                colors: [
                    PaletteColor {
                        r: 42,
                        g: 120,
                        b: 202,
                    },
                    PaletteColor {
                        r: 52,
                        g: 136,
                        b: 218,
                    },
                    PaletteColor {
                        r: 65,
                        g: 152,
                        b: 228,
                    },
                    PaletteColor {
                        r: 78,
                        g: 167,
                        b: 238,
                    },
                ],
            },
            soil: MaterialPalette4 {
                colors: [
                    PaletteColor {
                        r: 112,
                        g: 86,
                        b: 61,
                    },
                    PaletteColor {
                        r: 126,
                        g: 97,
                        b: 69,
                    },
                    PaletteColor {
                        r: 144,
                        g: 112,
                        b: 79,
                    },
                    PaletteColor {
                        r: 163,
                        g: 128,
                        b: 92,
                    },
                ],
            },
            sand: MaterialPalette4 {
                colors: [
                    PaletteColor {
                        r: 156,
                        g: 134,
                        b: 94,
                    },
                    PaletteColor {
                        r: 172,
                        g: 148,
                        b: 104,
                    },
                    PaletteColor {
                        r: 189,
                        g: 163,
                        b: 115,
                    },
                    PaletteColor {
                        r: 205,
                        g: 178,
                        b: 126,
                    },
                ],
            },
        }
    }
}

impl PaletteParams {
    pub fn validate(&self) -> Result<(), String> {
        // u8なので範囲チェックは不要。将来拡張用にフックだけ置く。
        Ok(())
    }
}

#[derive(Default, bevy::reflect::TypePath)]
pub struct PaletteParamsLoader;

impl AssetLoader for PaletteParamsLoader {
    type Asset = PaletteParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<PaletteParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: PaletteParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
