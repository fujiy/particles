use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bevy::app::App;
use bevy::log::tracing::field::{Field, Visit};
use bevy::log::tracing::{Id, Subscriber, info_span};
use bevy::log::tracing_subscriber::layer::{Context, Layer};
use bevy::log::tracing_subscriber::registry::LookupSpan;
use bevy::log::BoxedLayer;
use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Render, RenderApp, RenderSystems};
use wgpu_profiler::{GpuProfiler, GpuProfilerQuery, GpuProfilerSettings, GpuTimerQueryResult};

use crate::params::ActiveInterfaceParams;
use crate::physics::state::SimulationState;

thread_local! {
    static CPU_PROFILE_STACK: RefCell<Vec<ActiveCpuSpan>> = const { RefCell::new(Vec::new()) };
}

const PROFILE_SPAN_NAME: &str = "runtime_profile";
const PROFILE_LANE_CPU: &str = "cpu";
const GPU_LABEL_SEPARATOR: &str = "/";
const TARGET_REALTIME_MS_PER_SEC: f32 = 1_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeProfileLane {
    Cpu,
    Gpu,
}

impl RuntimeProfileLane {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProfileKey {
    lane: RuntimeProfileLane,
    category: String,
    detail: String,
}

impl ProfileKey {
    fn new(lane: RuntimeProfileLane, category: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            lane,
            category: category.into(),
            detail: detail.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeProfileSegment {
    pub lane: RuntimeProfileLane,
    pub category: String,
    pub detail: String,
    pub label: String,
    pub total_duration: Duration,
    pub total_ms: f32,
    pub rate_ms_per_sec: f32,
}

#[derive(Resource, Clone, Debug)]
pub struct RuntimeProfileShared(Arc<Mutex<RuntimeProfileAccum>>);

impl Default for RuntimeProfileShared {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(RuntimeProfileAccum::default())))
    }
}

impl RuntimeProfileShared {
    pub fn record_cpu_duration(
        &self,
        category: impl Into<String>,
        detail: impl Into<String>,
        duration: Duration,
    ) {
        self.record_duration(ProfileKey::new(RuntimeProfileLane::Cpu, category, detail), duration);
    }

    pub fn record_gpu_duration(
        &self,
        category: impl Into<String>,
        detail: impl Into<String>,
        duration: Duration,
    ) {
        self.record_duration(ProfileKey::new(RuntimeProfileLane::Gpu, category, detail), duration);
    }

    pub fn take_snapshot(
        &self,
        update_period: Duration,
        min_segment_ms_per_sec: f32,
        max_segments_per_lane: usize,
        over_budget_headroom: f32,
    ) -> Option<RuntimeProfileSnapshot> {
        let Ok(mut inner) = self.0.lock() else {
            return None;
        };
        let now = Instant::now();
        let elapsed = now.saturating_duration_since(inner.window_started);
        if elapsed < update_period {
            return None;
        }
        let elapsed_secs = elapsed.as_secs_f32().max(1.0e-5);
        let cpu = build_lane_segments(
            &inner.accum,
            RuntimeProfileLane::Cpu,
            elapsed_secs,
            min_segment_ms_per_sec,
            max_segments_per_lane,
        );
        let gpu = build_lane_segments(
            &inner.accum,
            RuntimeProfileLane::Gpu,
            elapsed_secs,
            min_segment_ms_per_sec,
            max_segments_per_lane,
        );
        let cpu_total_ms_per_sec = cpu.iter().map(|segment| segment.rate_ms_per_sec).sum::<f32>();
        let gpu_total_ms_per_sec = gpu.iter().map(|segment| segment.rate_ms_per_sec).sum::<f32>();
        let scale_ms_per_sec = TARGET_REALTIME_MS_PER_SEC.max(
            cpu_total_ms_per_sec
                .max(gpu_total_ms_per_sec)
                .max(0.0)
                * over_budget_headroom.max(1.0),
        );
        inner.accum.clear();
        inner.window_started = now;
        inner.sequence = inner.sequence.saturating_add(1);
        Some(RuntimeProfileSnapshot {
            elapsed_secs,
            scale_ms_per_sec,
            cpu_total_ms_per_sec,
            gpu_total_ms_per_sec,
            cpu,
            gpu,
            sequence: inner.sequence,
        })
    }

    fn record_duration(&self, key: ProfileKey, duration: Duration) {
        let Ok(mut inner) = self.0.lock() else {
            return;
        };
        *inner.accum.entry(key).or_default() += duration;
    }

    pub fn reset_window(&self) {
        let Ok(mut inner) = self.0.lock() else {
            return;
        };
        inner.accum.clear();
        inner.window_started = Instant::now();
    }
}

#[derive(Resource, Clone, Debug)]
pub struct RuntimeProfileSnapshot {
    pub elapsed_secs: f32,
    pub scale_ms_per_sec: f32,
    pub cpu_total_ms_per_sec: f32,
    pub gpu_total_ms_per_sec: f32,
    pub cpu: Vec<RuntimeProfileSegment>,
    pub gpu: Vec<RuntimeProfileSegment>,
    pub sequence: u64,
}

impl Default for RuntimeProfileSnapshot {
    fn default() -> Self {
        Self {
            elapsed_secs: 0.0,
            scale_ms_per_sec: TARGET_REALTIME_MS_PER_SEC,
            cpu_total_ms_per_sec: 0.0,
            gpu_total_ms_per_sec: 0.0,
            cpu: Vec::new(),
            gpu: Vec::new(),
            sequence: 0,
        }
    }
}

#[derive(Debug)]
struct RuntimeProfileAccum {
    window_started: Instant,
    accum: HashMap<ProfileKey, Duration>,
    sequence: u64,
}

impl Default for RuntimeProfileAccum {
    fn default() -> Self {
        Self {
            window_started: Instant::now(),
            accum: HashMap::default(),
            sequence: 0,
        }
    }
}

#[derive(Clone)]
struct CpuProfileLayer {
    shared: RuntimeProfileShared,
}

#[derive(Clone, Debug)]
struct CpuProfileSpanMetadata {
    category: String,
    detail: String,
}

#[derive(Clone, Debug)]
struct ActiveCpuSpan {
    span_id: u64,
    metadata: CpuProfileSpanMetadata,
    started_at: Instant,
    child_duration: Duration,
}

#[derive(Default)]
struct ProfileSpanFieldVisitor {
    profile_lane: Option<String>,
    profile_category: Option<String>,
    profile_detail: Option<String>,
}

impl Visit for ProfileSpanFieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_value(field.name(), value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.record_value(field.name(), format!("{value:?}").trim_matches('"').to_string());
    }
}

impl ProfileSpanFieldVisitor {
    fn into_cpu_metadata(self) -> Option<CpuProfileSpanMetadata> {
        (self.profile_lane.as_deref() == Some(PROFILE_LANE_CPU)).then_some(CpuProfileSpanMetadata {
            category: self.profile_category?,
            detail: self.profile_detail?,
        })
    }

    fn record_value(&mut self, field_name: &str, value: String) {
        match field_name {
            "profile_lane" => self.profile_lane = Some(value),
            "profile_category" => self.profile_category = Some(value),
            "profile_detail" => self.profile_detail = Some(value),
            _ => {}
        }
    }
}

impl<S> Layer<S> for CpuProfileLayer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_new_span(
        &self,
        attrs: &bevy::log::tracing::span::Attributes<'_>,
        id: &Id,
        ctx: Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let mut visitor = ProfileSpanFieldVisitor::default();
        attrs.record(&mut visitor);
        let Some(metadata) = visitor.into_cpu_metadata() else {
            return;
        };
        span.extensions_mut().insert(metadata);
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        let Some(metadata) = span.extensions().get::<CpuProfileSpanMetadata>().cloned() else {
            return;
        };
        CPU_PROFILE_STACK.with(|stack| {
            stack.borrow_mut().push(ActiveCpuSpan {
                span_id: id.into_u64(),
                metadata,
                started_at: Instant::now(),
                child_duration: Duration::ZERO,
            });
        });
    }

    fn on_exit(&self, id: &Id, _ctx: Context<'_, S>) {
        let exited_id = id.into_u64();
        CPU_PROFILE_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            let Some(mut active) = stack.pop() else {
                return;
            };
            if active.span_id != exited_id {
                if let Some(index) = stack.iter().rposition(|entry| entry.span_id == exited_id) {
                    active = stack.remove(index);
                } else {
                    return;
                }
            }
            let elapsed = active.started_at.elapsed();
            let self_duration = elapsed.saturating_sub(active.child_duration);
            self.shared.record_cpu_duration(
                active.metadata.category,
                active.metadata.detail,
                self_duration,
            );
            if let Some(parent) = stack.last_mut() {
                parent.child_duration += elapsed;
            }
        });
    }
}

#[derive(Resource)]
pub struct GpuProfilerRenderState {
    profiler: Mutex<GpuProfiler>,
    shared: RuntimeProfileShared,
}

impl GpuProfilerRenderState {
    fn new(device: &RenderDevice, shared: RuntimeProfileShared) -> Option<Self> {
        let settings = GpuProfilerSettings {
            enable_timer_queries: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            enable_debug_groups: true,
            ..Default::default()
        };
        let profiler = GpuProfiler::new(device.wgpu_device(), settings).ok()?;
        Some(Self {
            profiler: Mutex::new(profiler),
            shared,
        })
    }
}

#[derive(Default)]
pub struct RuntimeProfilerPlugin;

impl Plugin for RuntimeProfilerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RuntimeProfileShared>()
            .init_resource::<RuntimeProfileSnapshot>();

        let shared = app.world().resource::<RuntimeProfileShared>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(shared);
        render_app.add_systems(
            Render,
            finalize_gpu_profiler_frame.in_set(RenderSystems::Cleanup),
        );
    }

    fn finish(&self, app: &mut App) {
        let shared = app.world().resource::<RuntimeProfileShared>().clone();
        let render_app = app.sub_app_mut(RenderApp);
        let device = render_app.world().resource::<RenderDevice>().clone();
        if let Some(state) = GpuProfilerRenderState::new(&device, shared) {
            render_app.world_mut().insert_resource(state);
        }
    }
}

pub fn build_profiling_layer(app: &mut App) -> Option<BoxedLayer> {
    if app.world().get_resource::<RuntimeProfileShared>().is_none() {
        app.init_resource::<RuntimeProfileShared>();
    }
    let shared = app.world().resource::<RuntimeProfileShared>().clone();
    let layer = CpuProfileLayer { shared };
    #[cfg(feature = "tracy")]
    {
        Some(Box::new(layer.and_then(tracing_tracy::TracyLayer::default())))
    }
    #[cfg(not(feature = "tracy"))]
    {
        Some(Box::new(layer))
    }
}

pub fn cpu_profile_span(category: &'static str, detail: &'static str) -> bevy::log::tracing::Span {
    info_span!(
        PROFILE_SPAN_NAME,
        profile_lane = PROFILE_LANE_CPU,
        profile_category = category,
        profile_detail = detail
    )
}

pub fn gpu_profile_label(category: &str, detail: &str) -> String {
    format!("{category}{GPU_LABEL_SEPARATOR}{detail}")
}

pub fn begin_gpu_pass_query(
    world: &World,
    category: &str,
    detail: &str,
    encoder: &mut wgpu::CommandEncoder,
) -> Option<GpuProfilerQuery> {
    let state = world.get_resource::<GpuProfilerRenderState>()?;
    let Ok(profiler) = state.profiler.lock() else {
        return None;
    };
    Some(profiler.begin_pass_query(gpu_profile_label(category, detail), encoder))
}

pub fn end_gpu_pass_query(
    world: &World,
    encoder: &mut wgpu::CommandEncoder,
    query: Option<GpuProfilerQuery>,
) {
    let Some(query) = query else {
        return;
    };
    let Some(state) = world.get_resource::<GpuProfilerRenderState>() else {
        return;
    };
    let Ok(profiler) = state.profiler.lock() else {
        return;
    };
    profiler.end_query(encoder, query);
}

pub fn resolve_gpu_profiler_queries(world: &World, encoder: &mut wgpu::CommandEncoder) {
    let Some(state) = world.get_resource::<GpuProfilerRenderState>() else {
        return;
    };
    let Ok(mut profiler) = state.profiler.lock() else {
        return;
    };
    profiler.resolve_queries(encoder);
}

pub fn update_runtime_profile_snapshot(
    interface_params: Res<ActiveInterfaceParams>,
    sim_state: Res<SimulationState>,
    shared: Res<RuntimeProfileShared>,
    mut snapshot: ResMut<RuntimeProfileSnapshot>,
) {
    if !sim_state.running && !sim_state.step_once {
        shared.reset_window();
        return;
    }
    let profiler_params = &interface_params.0.profiler;
    let update_hz = profiler_params.update_hz.max(1.0);
    let update_period = Duration::from_secs_f32(1.0 / update_hz);
    let Some(new_snapshot) = shared.take_snapshot(
        update_period,
        profiler_params.min_segment_ms_per_sec.max(0.0),
        profiler_params.max_segments_per_lane.max(1) as usize,
        profiler_params.over_budget_headroom.max(1.0),
    ) else {
        return;
    };
    *snapshot = new_snapshot;
}

fn finalize_gpu_profiler_frame(
    queue: Res<RenderQueue>,
    profiler_state: Option<Res<GpuProfilerRenderState>>,
) {
    let Some(profiler_state) = profiler_state else {
        return;
    };
    let Ok(mut profiler) = profiler_state.profiler.lock() else {
        return;
    };
    let _ = profiler.end_frame();
    while let Some(results) = profiler.process_finished_frame(queue.get_timestamp_period()) {
        collect_gpu_profile_results(&profiler_state.shared, &results);
    }
}

fn collect_gpu_profile_results(shared: &RuntimeProfileShared, results: &[GpuTimerQueryResult]) {
    for result in results {
        collect_gpu_query_result_recursive(shared, result);
    }
}

fn collect_gpu_query_result_recursive(shared: &RuntimeProfileShared, result: &GpuTimerQueryResult) -> Duration {
    let child_total = result
        .nested_queries
        .iter()
        .map(|child| collect_gpu_query_result_recursive(shared, child))
        .fold(Duration::ZERO, |acc, duration| acc + duration);
    let total = result
        .time
        .as_ref()
        .map(|time| Duration::from_secs_f64((time.end - time.start).max(0.0)))
        .unwrap_or(Duration::ZERO);
    let self_duration = total.saturating_sub(child_total);
    if self_duration > Duration::ZERO {
        let (category, detail) = parse_gpu_profile_label(&result.label);
        shared.record_gpu_duration(category, detail, self_duration);
    }
    total
}

fn parse_gpu_profile_label(label: &str) -> (String, String) {
    if let Some((category, detail)) = label.split_once(GPU_LABEL_SEPARATOR) {
        (category.to_string(), detail.to_string())
    } else {
        ("others".to_string(), label.to_string())
    }
}

fn normalized_profile_category(category: &str) -> &'static str {
    match category {
        "physics" => "physics",
        "terrain" | "water" | "overlay" | "ui" | "render" => "render",
        _ => "others",
    }
}

fn profile_category_sort_key(category: &str) -> u8 {
    match normalized_profile_category(category) {
        "physics" => 0,
        "render" => 1,
        _ => 2,
    }
}

fn build_lane_segments(
    accum: &HashMap<ProfileKey, Duration>,
    lane: RuntimeProfileLane,
    elapsed_secs: f32,
    min_segment_ms_per_sec: f32,
    max_segments_per_lane: usize,
) -> Vec<RuntimeProfileSegment> {
    let mut segments = accum
        .iter()
        .filter(|(key, _)| key.lane == lane)
        .map(|(key, duration)| {
            let category = normalized_profile_category(&key.category).to_string();
            RuntimeProfileSegment {
                lane,
                label: format!("{category}/{}", key.detail),
                category,
                detail: key.detail.clone(),
                total_duration: *duration,
                total_ms: duration.as_secs_f32() * 1_000.0,
                rate_ms_per_sec: duration.as_secs_f32() * 1_000.0 / elapsed_secs,
            }
        })
        .filter(|segment| segment.rate_ms_per_sec >= min_segment_ms_per_sec)
        .collect::<Vec<_>>();

    segments.sort_by(|a, b| {
        profile_category_sort_key(&a.category)
            .cmp(&profile_category_sort_key(&b.category))
            .then_with(|| a.detail.cmp(&b.detail))
    });

    if segments.len() > max_segments_per_lane {
        let overflow = segments.split_off(max_segments_per_lane.saturating_sub(1));
        let overflow_total = overflow
            .into_iter()
            .fold(Duration::ZERO, |acc, segment| acc + segment.total_duration);
        segments.push(RuntimeProfileSegment {
            lane,
            category: "others".to_string(),
            detail: "remainder".to_string(),
            label: "others/remainder".to_string(),
            total_duration: overflow_total,
            total_ms: overflow_total.as_secs_f32() * 1_000.0,
            rate_ms_per_sec: overflow_total.as_secs_f32() * 1_000.0 / elapsed_secs,
        });
    }

    segments
}

#[cfg(unix)]
pub fn process_cpu_time_seconds() -> Option<f64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes a valid rusage into the provided pointer on success.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: rc == 0 guarantees the structure has been initialized by getrusage.
    let usage = unsafe { usage.assume_init() };
    let user = usage.ru_utime;
    let sys = usage.ru_stime;
    Some(
        user.tv_sec as f64
            + user.tv_usec as f64 * 1e-6
            + sys.tv_sec as f64
            + sys.tv_usec as f64 * 1e-6,
    )
}

#[cfg(not(unix))]
pub fn process_cpu_time_seconds() -> Option<f64> {
    None
}
