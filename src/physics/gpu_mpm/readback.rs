// GPU → CPU readback for particle data.
//
// Async mapping: map_async is issued each Cleanup frame. The wgpu callback sets
// an AtomicBool flag when mapping completes. The following Cleanup reads the
// mapped range, parses particles, and unmaps — no poll(Wait), no frame stall.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use super::buffers::GpuParticle;

// ---------------------------------------------------------------------------
// Final parsed result (shared render world → main world via Arc)
// ---------------------------------------------------------------------------

#[derive(Resource, Clone, Default)]
pub struct GpuReadbackResult {
    pub inner: Arc<Mutex<Option<Vec<GpuParticle>>>>,
}

impl GpuReadbackResult {
    pub fn take(&self) -> Option<Vec<GpuParticle>> {
        if let Ok(mut g) = self.inner.lock() {
            g.take()
        } else {
            None
        }
    }
    pub fn store(&self, particles: Vec<GpuParticle>) {
        if let Ok(mut g) = self.inner.lock() {
            *g = Some(particles);
        }
    }
}

// ---------------------------------------------------------------------------
// Render-world-only async readback state
// ---------------------------------------------------------------------------

/// Tracks state of the in-flight map_async request.
#[derive(Resource)]
pub struct GpuReadbackState {
    /// True while a map_async is in-flight (issued but not yet consumed).
    pub mapped: bool,
    /// Particle count for the in-flight mapping.
    pub pending_count: u32,
    /// Set to true by the wgpu callback when mapping completes.
    pub mapped_ready: Arc<AtomicBool>,
    /// Readback cadence counter (render frames).
    pub frame_counter: u64,
}

impl Default for GpuReadbackState {
    fn default() -> Self {
        Self {
            mapped: false,
            pending_count: 0,
            mapped_ready: Arc::new(AtomicBool::new(false)),
            frame_counter: 0,
        }
    }
}
