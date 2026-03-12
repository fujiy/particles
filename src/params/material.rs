use bevy::asset::{io::Reader, AssetLoader, LoadContext};
use bevy::prelude::*;

use crate::physics::material::MaterialParams;

#[derive(Default, bevy::reflect::TypePath)]
pub struct MaterialParamsLoader;

impl AssetLoader for MaterialParamsLoader {
    type Asset = MaterialParams;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<MaterialParams, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let text = std::str::from_utf8(&bytes)?;
        let params: MaterialParams = ron::from_str(text)?;
        Ok(params)
    }

    fn extensions(&self) -> &[&str] {
        &["ron"]
    }
}
