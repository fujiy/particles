// assets/params/camera.ron の Asset 型定義とローダー。
//
// カメラ初期位置・操作感度を実行時調整する。

use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Asset, TypePath, Debug, Clone, Serialize, Deserialize)]
pub struct CameraParams {
    pub viewport_height_m: f32,
    pub pan_speed_mps: f32,
    pub zoom_sensitivity: f32,
    pub min_zoom: f32,
    pub max_zoom: f32,
    pub start_cell_x: i32,
    pub start_cell_y: i32,
    pub start_z: f32,
}

impl Default for CameraParams {
    fn default() -> Self {
        Self {
            viewport_height_m: 14.0,
            pan_speed_mps: 10.0,
            zoom_sensitivity: 0.06,
            min_zoom: 0.1,
            max_zoom: 1000.0,
            start_cell_x: 0,
            start_cell_y: 8,
            start_z: 100.0,
        }
    }
}

impl CameraParams {
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
        check!(self.viewport_height_m, "viewport_height_m", 0.1, 2000.0);
        check!(self.pan_speed_mps, "pan_speed_mps", 0.0, 10000.0);
        check!(self.zoom_sensitivity, "zoom_sensitivity", 0.0, 5.0);
        check!(self.min_zoom, "min_zoom", 1e-4, 1000.0);
        check!(self.max_zoom, "max_zoom", 1e-3, 100000.0);
        if self.min_zoom > self.max_zoom {
            return Err(format!(
                "min_zoom ({}) > max_zoom ({})",
                self.min_zoom, self.max_zoom
            ));
        }
        check!(self.start_z, "start_z", -10000.0, 10000.0);
        Ok(())
    }
}

#[derive(Default, bevy::reflect::TypePath)]
pub struct CameraParamsLoader;

impl AssetLoader for CameraParamsLoader {
    type Asset = CameraParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<CameraParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: CameraParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
