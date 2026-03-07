use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

pub fn parse_hex_rgba(raw: &str) -> Result<(u8, u8, u8, u8), String> {
    let hex = raw
        .strip_prefix('#')
        .ok_or_else(|| "hex color must start with '#'".to_string())?;
    let (r, g, b, a) = match hex.len() {
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| "invalid hex color component in RGB".to_string())?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| "invalid hex color component in RGB".to_string())?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| "invalid hex color component in RGB".to_string())?;
            (r, g, b, 255)
        }
        8 => {
            let r = u8::from_str_radix(&hex[0..2], 16)
                .map_err(|_| "invalid hex color component in RGBA".to_string())?;
            let g = u8::from_str_radix(&hex[2..4], 16)
                .map_err(|_| "invalid hex color component in RGBA".to_string())?;
            let b = u8::from_str_radix(&hex[4..6], 16)
                .map_err(|_| "invalid hex color component in RGBA".to_string())?;
            let a = u8::from_str_radix(&hex[6..8], 16)
                .map_err(|_| "invalid hex color component in RGBA".to_string())?;
            (r, g, b, a)
        }
        _ => {
            return Err(format!(
                "invalid hex color length: expected 6 or 8 hex digits, got {}",
                hex.len()
            ));
        }
    };
    Ok((r, g, b, a))
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct Rgba8Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct RgbaColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

pub fn deserialize_u8_color_from_ron<'de, D>(deserializer: D) -> Result<Rgba8Color, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Hex(String),
        Rgb { r: u8, g: u8, b: u8 },
        Rgba { r: u8, g: u8, b: u8, a: u8 },
    }

    match Repr::deserialize(deserializer)? {
        Repr::Hex(raw) => {
            let (r, g, b, a) = parse_hex_rgba(&raw).map_err(serde::de::Error::custom)?;
            Ok(Rgba8Color { r, g, b, a })
        }
        Repr::Rgb { r, g, b } => Ok(Rgba8Color { r, g, b, a: 255 }),
        Repr::Rgba { r, g, b, a } => Ok(Rgba8Color { r, g, b, a }),
    }
}

pub fn deserialize_rgba_color_from_ron<'de, D>(deserializer: D) -> Result<RgbaColor, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Hex(String),
        Rgb { r: f32, g: f32, b: f32 },
        Rgba { r: f32, g: f32, b: f32, a: f32 },
    }

    match Repr::deserialize(deserializer)? {
        Repr::Hex(raw) => {
            let (r, g, b, a) = parse_hex_rgba(&raw).map_err(serde::de::Error::custom)?;
            Ok(RgbaColor {
                r: f32::from(r) / 255.0,
                g: f32::from(g) / 255.0,
                b: f32::from(b) / 255.0,
                a: f32::from(a) / 255.0,
            })
        }
        Repr::Rgb { r, g, b } => Ok(RgbaColor { r, g, b, a: 1.0 }),
        Repr::Rgba { r, g, b, a } => Ok(RgbaColor { r, g, b, a }),
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::{
        Rgba8Color, RgbaColor, deserialize_rgba_color_from_ron, deserialize_u8_color_from_ron,
        parse_hex_rgba,
    };

    #[derive(Deserialize)]
    struct U8ColorWrap {
        #[serde(deserialize_with = "deserialize_u8_color_from_ron")]
        c: Rgba8Color,
    }

    #[derive(Deserialize)]
    struct RgbaColorWrap {
        #[serde(deserialize_with = "deserialize_rgba_color_from_ron")]
        c: RgbaColor,
    }

    #[test]
    fn parses_u8_hex_color() {
        let value: U8ColorWrap = ron::from_str("(c: \"#2A78CA\")").unwrap();
        assert_eq!(
            value.c,
            Rgba8Color {
                r: 42,
                g: 120,
                b: 202,
                a: 255
            }
        );
    }

    #[test]
    fn parses_u8_hex_color_with_alpha() {
        let value: U8ColorWrap = ron::from_str("(c: \"#2A78CA80\")").unwrap();
        assert_eq!(
            value.c,
            Rgba8Color {
                r: 42,
                g: 120,
                b: 202,
                a: 128
            }
        );
    }

    #[test]
    fn parses_rgba_hex_color() {
        let value: RgbaColorWrap = ron::from_str("(c: \"#2A78CA80\")").unwrap();
        assert!((value.c.r - 42.0 / 255.0).abs() < 1e-6);
        assert!((value.c.a - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn parses_hex_rgb_and_alpha_without_hash() {
        let err = parse_hex_rgba("2A78CA").unwrap_err();
        assert!(err.contains("must start"));
    }
}
