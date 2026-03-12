// GPU → CPU readback for particle data.
//
// Async mapping: map_async is issued each Cleanup frame. The wgpu callback sets
// an AtomicBool flag when mapping completes. The following Cleanup reads the
// mapped range, parses particles, and unmaps — no poll(Wait), no frame stall.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;

use super::buffers::{GpuChunkEventRecord, GpuMoverRecord, GpuParticle, GpuStatisticsScalars};

// ---------------------------------------------------------------------------
// Final parsed result (shared render world → main world via Arc)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct GpuParticleReadbackPayload {
    pub particles: Vec<GpuParticle>,
    pub particle_revision: u64,
}

#[derive(Resource, Clone, Default)]
pub struct GpuReadbackResult {
    pub inner: Arc<Mutex<Option<GpuParticleReadbackPayload>>>,
}

impl GpuReadbackResult {
    pub fn take(&self) -> Option<GpuParticleReadbackPayload> {
        if let Ok(mut g) = self.inner.lock() {
            g.take()
        } else {
            None
        }
    }
    pub fn store(&self, payload: GpuParticleReadbackPayload) {
        if let Ok(mut g) = self.inner.lock() {
            *g = Some(payload);
        }
    }
}

#[derive(Resource, Clone, Default)]
pub struct GpuStatisticsReadbackResult {
    pub inner: Arc<Mutex<Option<GpuStatisticsScalars>>>,
}

#[derive(Resource, Clone, Default)]
pub struct GpuMoverReadbackResult {
    pub inner: Arc<Mutex<Option<Vec<GpuMoverRecord>>>>,
}

#[derive(Resource, Clone, Default)]
pub struct GpuChunkEventReadbackResult {
    pub inner: Arc<Mutex<Option<Vec<GpuChunkEventRecord>>>>,
}

impl GpuMoverReadbackResult {
    pub fn take(&self) -> Option<Vec<GpuMoverRecord>> {
        if let Ok(mut g) = self.inner.lock() {
            g.take()
        } else {
            None
        }
    }
    pub fn store(&self, movers: Vec<GpuMoverRecord>) {
        if let Ok(mut g) = self.inner.lock() {
            *g = Some(movers);
        }
    }
}

impl GpuChunkEventReadbackResult {
    pub fn take(&self) -> Option<Vec<GpuChunkEventRecord>> {
        if let Ok(mut g) = self.inner.lock() {
            g.take()
        } else {
            None
        }
    }
    pub fn store(&self, events: Vec<GpuChunkEventRecord>) {
        if let Ok(mut g) = self.inner.lock() {
            *g = Some(events);
        }
    }
}

#[derive(Resource, Clone, Default)]
pub struct GpuMoverApplyAck {
    inner: Arc<AtomicU64>,
}

impl GpuMoverApplyAck {
    pub fn signal(&self) {
        self.inner.fetch_add(1, Ordering::AcqRel);
    }

    pub fn value(&self) -> u64 {
        self.inner.load(Ordering::Acquire)
    }
}

impl GpuStatisticsReadbackResult {
    pub fn take(&self) -> Option<GpuStatisticsScalars> {
        if let Ok(mut g) = self.inner.lock() {
            g.take()
        } else {
            None
        }
    }
    pub fn store(&self, counts: GpuStatisticsScalars) {
        if let Ok(mut g) = self.inner.lock() {
            *g = Some(counts);
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
    /// Particle revision captured when the current mapping was issued.
    pub pending_particle_revision: u64,
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
            pending_particle_revision: 0,
            mapped_ready: Arc::new(AtomicBool::new(false)),
            frame_counter: 0,
        }
    }
}

/// Tracks in-flight map_async state for phase-count readback.
#[derive(Resource)]
pub struct GpuStatisticsReadbackState {
    pub mapped: bool,
    pub mapped_ready: Arc<AtomicBool>,
    pub frame_counter: u64,
}

impl Default for GpuStatisticsReadbackState {
    fn default() -> Self {
        Self {
            mapped: false,
            mapped_ready: Arc::new(AtomicBool::new(false)),
            frame_counter: 0,
        }
    }
}

#[derive(Resource)]
pub struct GpuMoverReadbackState {
    pub mapped: bool,
    pub mapped_ready: Arc<AtomicBool>,
    pub frame_counter: u64,
    pub copy_pending: Arc<AtomicBool>,
}

impl Default for GpuMoverReadbackState {
    fn default() -> Self {
        Self {
            mapped: false,
            mapped_ready: Arc::new(AtomicBool::new(false)),
            frame_counter: 0,
            copy_pending: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[derive(Resource)]
pub struct GpuChunkEventReadbackState {
    pub mapped: bool,
    pub mapped_ready: Arc<AtomicBool>,
    pub frame_counter: u64,
    pub copy_pending: Arc<AtomicBool>,
}

impl Default for GpuChunkEventReadbackState {
    fn default() -> Self {
        Self {
            mapped: false,
            mapped_ready: Arc::new(AtomicBool::new(false)),
            frame_counter: 0,
            copy_pending: Arc::new(AtomicBool::new(false)),
        }
    }
}
