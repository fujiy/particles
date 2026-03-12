use crate::physics::material::ParticleMaterial;

pub const MPM_PHASE_ID_WATER: u8 = 0;
pub const MPM_PHASE_ID_GRANULAR_SOIL: u8 = 1;
pub const MPM_PHASE_ID_GRANULAR_SAND: u8 = 2;

pub fn mpm_phase_id_for_particle(material: ParticleMaterial) -> Option<u8> {
    match material {
        ParticleMaterial::WaterLiquid => Some(MPM_PHASE_ID_WATER),
        ParticleMaterial::StoneGranular => Some(MPM_PHASE_ID_GRANULAR_SOIL),
        ParticleMaterial::SoilGranular => Some(MPM_PHASE_ID_GRANULAR_SOIL),
        ParticleMaterial::GrassGranular => Some(MPM_PHASE_ID_GRANULAR_SOIL),
        ParticleMaterial::SandGranular => Some(MPM_PHASE_ID_GRANULAR_SAND),
        _ => None,
    }
}
