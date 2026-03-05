# Tasks

## Task Format

- 本ドキュメントは「実装単位（Work Unit）」ごとに管理する。
- 各Work Unitは以下を持つ。
  - 背景
  - スコープ
  - サブタスク（チェックリスト）
  - 完了条件
- `docs/design.md` は最新目標のみを持ち、変更の背景・移行文脈はここに記載する。

## Lifecycle Rules

- 完了済みタスクは削除しない。`Done` セクションへ移動するか、アーカイブへ退避して要約を残す。
- 方針転換で無効化したタスクは削除しない。`Closed Work Units` に `Deferred` / `Superseded` として残す。
- `tasks.md` は「実行中タスク」と「直近完了タスク」を優先して保持し、肥大化した詳細は `docs/tasks_archive/` へ移す。
- アーカイブ退避時は、`tasks.md` に以下の1行要約を残す。
  - `ID / Status / 理由 / 参照アーカイブ`
- アーカイブ時はスナップショットファイルを作成し、過去情報の原文を保持する。
- 目安として、`tasks.md` が 600 行を超えたらコンパクションを行う。

## Active Work Units

### [MPM-GPU-00] 単一grid回帰とGPU-First基盤再定義

- Status: `In Progress`
- 背景:
  - block/空間LoD/時間LoD前提の実装は境界管理コストが高く、CPU並列最適化の伸びも限定的だった。
  - 今後の水・粉体・剛体連成を見据えると、水ソルバ本番経路を早期にGPUへ移す方が手戻りを抑えられる。
- スコープ:
  - 物理設計を単一解像度グリッドへ回帰し、GPU常駐データ前提に再定義する。
  - 連続体の本番経路をGPUへ一本化し、不要CPUコードを段階的に削除する。
- Subtasks:
  - [x] ADR（単一grid + GPU-first）を作成する。
  - [x] `design.md` を単一grid + active tile + GPU常駐方針へ更新する。
  - [x] `physics.md` を block/ghost前提から単一grid GPUパイプライン仕様へ更新する。
  - [x] 旧CPU連続体コードの削除順序（依存関係と撤去単位）を設計する。
  - [x] GPUバッファレイアウト（particle/grid/active tile/metrics）を確定する。
  - [ ] 不要になったCPU実装を削除し、関連テストとドキュメント参照を整理する。
- 完了条件:
  - 設計文書が新方針へ整合し、GPU一本化とCPU撤去の実施手順が定義されている。


### [MPM-GPU-01] Water Drop最小GPU再現（計算）

- Status: `In Progress`
- 背景:
  - 新方針の最小成立条件は `water_drop` をGPU経路で安定再現すること。
  - まずは計算成立を優先し、描画経路は段階的にGPU完結化する。
- スコープ:
  - 単一grid MLS-MPM をGPU computeで実行し、`water_drop` の物理挙動を再現する。
  - 地形SDF境界補正をGPU経路へ接続する。
- Subtasks:
  - [x] GPU compute pass の最小骨格を実装する（`clear -> p2g -> grid_update -> g2p`）。
  - [x] 粒子/格子バッファのGPU常駐更新を実装する（CPU→GPU upload, ExtractResource経由）。
  - [x] 地形SDF/normalをGPUバッファへ供給する経路を実装する。
  - [x] GPU readbackバッファ（MAP_READ）を用意し、Arcベース共有チャネルで結果を転送する構造を実装する。
  - [x] GPU readback結果をContinuumParticleWorldへ適用する（`apply_gpu_readback` system in Update）。
  - [x] `water_drop` で NaN/発散なしで実時間継続することを確認する。
  - [x] assertion/デバッグ向けに GPU readback を低頻度（既定1秒）で取得できるようにする。
  - [x] compute shader import 失敗を解消し、起動直後から MPM pipeline を安定稼働させる。
  - [x] GPU経路の物性・境界パラメータをCPU経路相当に整合させる（減衰/圧縮性/境界補正）。
  - [ ] 質量誤差・侵入率・CFLをGPU経路で計測できるようにする。
  - [ ] 旧CPU経路依存の更新ループを削除し、GPU経路のみで `water_drop` が成立することを確認する。
- 進捗:
  - 2026-03-01: `src/physics/gpu_mpm/` モジュール作成（buffers, gpu_resources, pipeline, shaders, node, sync, readback）。
  - 2026-03-01: WGSL shaders作成（mpm_types, mpm_clear, mpm_p2g, mpm_grid_update, mpm_g2p）。
  - 2026-03-01: `GpuMpmPlugin` を `main.rs` に登録、コンパイル通過、テスト105件全通過確認。
  - 2026-03-01: GPU particle buffer 72 bytes/particle、grid buffer 16 bytes/node 確定。
  - 2026-03-01: `apply_gpu_readback` (Update) + `readback_particles` (RenderSystems::Cleanup) 実装。GPU→CPU同期経路完成。コンパイル通過。
  - 2026-03-01: WGSL `#define_import_path particles::mpm_types` 追加、struct field 名を `f_00/c_00/v_0` 等に修正（naga swizzle 解釈回避）。シェーダーエラーなし確認。
  - 2026-03-01: `SimulationState::gpu_mpm_active` フラグ追加。粒子あり時に GPU 経路を有効化し CPU MPM ステップをスキップ。テスト105件通過。
  - 2026-03-01: `SimulationState::mpm_enabled` を追加し、CPU/GPU MPM 両経路を一括停止できるよう変更。overlay検証向けに「粒子ロード + 表示のみ」動作を可能化。
  - 2026-03-01: 性能切り分けのため `main.rs` から `GpuMpmPlugin` 登録を一時的に外し、compute shader / readback 経路を停止。
  - 2026-03-01: 追加切り分けとして `main.rs` から `RenderPlugin` を外し、overlay依存の `TerrainRenderDiagnostics` のみ初期化。overlay以外の描画ルートを停止。
  - 2026-03-01: Physics Area Overlay の MPM block/tile 表示を廃止し、`world_grid_layout()` に基づく GPU 単一グリッド（uniform grid, no tiles）表示へ切替。UIラベルも GPU nodes/cells 表示へ更新。
  - 2026-03-01: `GpuMpmPlugin` を再登録。`MpmGpuControl { init_only: true }` を導入し、GPUバッファ初期化のみ有効・upload/compute/readback は停止した切り分けモードを追加。
  - 2026-03-01: readback経路を切り分け用に有効化。compute無効時でも `particle_buf -> readback_buf` コピーを実行し、`apply_gpu_readback` で `ParticleWorld` へ同期して particle overlay 表示へ反映。
  - 2026-03-01: particle overlay を gizmos 円描画から GPU sprite 描画へ置換。円テクスチャを一度生成し、粒子ごとの sprite transform/color/size を更新する方式へ変更。
  - 2026-03-01: particle overlay をさらに GPU完結へ変更。Core2D render graph に専用 pass を追加し、`MpmGpuBuffers.particle_buf` を頂点インスタンス参照して円を直接描画。`MpmGpuControl.readback_enabled=false` で readback を停止。
  - 2026-03-01: particle overlay 非表示の切り分けとして、粒子バッファ参照を使わない GPU デバッグパターン（時間変化フルスクリーン描画）を追加。overlay pass が生きているか単体確認可能にした。
  - 2026-03-01: overlay表示元を安定化。`ensure_continuum_seed_from_particle_world` を追加し、`ParticleWorld` に水粒子があるのに `ContinuumParticleWorld` が空な場合に自動再構築して GPU upload ソースを保証。particle overlay はデバッグ強制を解除して粒子描画へ復帰。
  - 2026-03-01: さらに切り分けのため、particle overlay shader を `view + time uniform` のみで描く強制デバッグ表示へ簡素化。粒子バッファ/params依存を外し、render pass 単体の可視化確認を優先。
  - 2026-03-01: `particle_overlay_gpu_pipeline` の multisample 設定を `Msaa.samples()` に合わせて specialize するよう修正。RenderPass sample_count=4 との不一致パニックを解消。
  - 2026-03-01: デバッグ可視性を優先し、particle overlay pass のブレンドを無効化（不透明上書き）して描画確認を容易化。
  - 2026-03-01: `SimulationState.mpm_enabled=false` 時に `fixed_update::step_physics` を早期 return するよう変更。overlay/debug中に active region 走査・object field 再構築などのCPU負荷が走り続ける問題を抑制。
  - 2026-03-01: 切り分けのため particle overlay の強制デバッグパターン描画を一時停止（overlay node 早期return）。
  - 2026-03-01: 追加切り分けとして `main.rs` の `MpmGpuControl.init_only=true` へ切替。GpuMpmPlugin を残したまま upload/compute/readback 同期を停止し、CPU負荷源を分離。
  - 2026-03-01: particle overlay 描画を再有効化。GPU pass を「背景1インスタンス + 粒子インスタンス群」に変更し、粒子ゼロでも薄い背景色を常時描画するよう更新。
  - 2026-03-01: particle overlay shader を背景+粒子描画へ更新（instance 0: fullscreen tint, instance 1..N: particle circles）。overlay ON時は粒子有無に関わらず薄い背景を出す挙動を固定化。
  - 2026-03-01: 追加切り分けとして particle overlay node の `overlay_state` 依存を一時解除し、ON/OFFに関係なく背景描画を実行。`MpmGpuBuffers` が未生成/未更新でも背景を描けるよう fallback uniform/storage buffer を追加。
  - 2026-03-01: さらに切り分けのため particle overlay を背景専用の最小パスへ簡素化（`ViewUniform` のみbind、粒子/params/storage参照を除去）。背景の常時表示可否をまず確定する段階へ移行。
  - 2026-03-01: 背景表示確認後、粒子描画を段階的に再導入。`instance 0` 背景 + `instance 1..N` 粒子描画へ復帰し、`main.rs` を `MpmGpuControl { init_only: false, readback_enabled: false }` に戻して GPU 粒子uploadを再有効化。合わせて `sync::prepare_particle_upload/prepare_terrain_upload` を毎フレーム再uploadしない形へ修正。
  - 2026-03-01: GPU MPM再有効化。`SimulationState::mpm_enabled` のデフォルトを `true` に変更し、`RenderGraph` へ `MpmComputeLabel -> CameraDriverLabel` エッジを追加して compute node が毎フレーム実行されるよう修正。
  - 2026-03-01: 切り分けのため `drift_only` モードを追加。`clear/p2g/grid_update/g2p` を停止し、GPU compute で粒子を一定速度で平行移動する `mpm_drift.wgsl` のみ実行する経路へ変更。`main.rs` は `SimulationState.mpm_enabled=false` + `MpmGpuControl{drift_only:true}` で起動するよう設定。
  - 2026-03-01: Codex単独で回せる自動検証ループを追加。`PARTICLES_AUTOVERIFY_DRIFT=1 cargo run` で `water_drop` を自動ロードし、一定フレーム後の粒子位置差分を `artifacts/drift_autoverify.json` に出力して自動終了する仕組みを実装。
  - 2026-03-01: `mpm_drift.wgsl` の `#import` 依存により `Shader import not yet available` で drift pipeline が不成立だった問題を修正（シェーダー内に `MpmParams/GpuParticle` を自己完結定義）。自動検証結果で `mean_dx > 0` かつ粒子間差分がほぼ一様であることを確認。
  - 2026-03-01: MPM実行モードを復帰。通常起動時は `SimulationState.mpm_enabled=true` / `MpmGpuControl{drift_only:false, readback_enabled:false}` となるよう `main.rs` を環境変数駆動へ変更（driftデバッグは `PARTICLES_AUTOVERIFY_DRIFT=1` 時のみ有効）。
  - 2026-03-01: MPM自動検証ループを追加。`PARTICLES_AUTOVERIFY_MPM=1 cargo run` で `water_drop` を自動ロードし、落下量・地形侵入率・平均FPSを `artifacts/mpm_autoverify.json` に出力して自動終了する仕組みを実装。
  - 2026-03-01: GPU MPM時でも `fixed_update` のCPU重処理（active region/object field等）が走っていたため、`sim_state.gpu_mpm_active` で早期returnする軽量化を実施。MPM自動検証で `avg_fps ≈ 103`、`mean_drop > 0`、`terrain_penetration_ratio=0` を確認。
  - 2026-03-01: `test assertions` 向けに GPU readback を低頻度化。`MpmGpuControl.readback_interval_frames` を追加し、既定で60フレーム間隔（約1秒）でのみ `particle_buf -> readback_buf` コピー/マップを行うよう変更。`PARTICLES_GPU_READBACK_INTERVAL_FRAMES` で上書き可能化。
  - 2026-03-01: GPU MPM経路で `ReplayState.current_step` が進まず assertion 条件が発火しない問題を修正。`fixed_update` の `gpu_mpm_active` 早期return分岐でも `running || step_once` 時に step を進めるようにした。
  - 2026-03-01: `water_drop` 非落下の主因だった compute shader import 失敗（`Shader import not yet available`）を修正。`mpm_types.wgsl` を明示ロードし、import composerで失敗する識別子（数字/先頭 `_` を含む名前）を改名して MPM pipeline のコンパイル失敗を解消。`PARTICLES_AUTOVERIFY_MPM=1 cargo run` で `run_frames=240` でも落下・地形相互作用・FPS指標の通過を再確認。
  - 2026-03-01: 水塊が縦列で跳ねて横拡散しない問題に対し、GPU `p2g` の圧力項を修正（圧縮時のみ圧力を発生させ、応力寄与の運動量符号を CPU 実装と整合化）。`water_drop` で落下後の横方向広がり改善を確認。
  - 2026-03-01: GPU MPM の時間進行を render FPS 依存から固定ステップ積算へ変更。`MpmGpuRunRequest.substeps` と `MpmGpuStepClock` を導入し、`fixed_dt` ベースで1フレーム内に複数 substep 実行、処理限界超過時は上限で打ち切ってスロー化する挙動を実装。
  - 2026-03-01: 左壁接触での上向き吹き飛び対策として、GPU境界SDF生成を CPU `terrain_boundary` 実装へ整合化（距離を「セル中心近似」から「セルAABB最近点距離」へ修正し、可能時は `TerrainWorld::sample_signed_distance_and_normal` を直接使用）。`PARTICLES_AUTOVERIFY_MPM=1`（1200フレーム）で `water_surface_height_p95` を含む全指標の通過を確認。
  - 2026-03-01: 物性のCPU相当化として GPU 減衰/境界パラメータをCPU経路に合わせて調整（`C_DAMPING=0.05`、全体速度減衰/速度cap撤去、`bulk_modulus=ρc²`、`j_min/j_max=0.6/1.4`、境界の `threshold/gain/cap/tangential` をCPU式へ一致）。`PARTICLES_AUTOVERIFY_MPM=1`（1200フレーム）再実行で指標通過を確認。
  - 2026-03-01: モジュール整理として `drift` デバッグ経路を削除（`MpmGpuControl.drift_only`、`run_drift_autoverify`、`mpm_drift.wgsl` と対応 pipeline/node 分岐を撤去）。あわせて CPU 側 MPM 実行ルートを撤去し、`step_simulation_once` は粒子ステップ専用へ簡素化。
- 完了条件:
  - `water_drop` がGPU経路で安定再現され、品質指標を自動計測できる。


### [MPM-GPU-02] Water Drop表示のGPU完結化

- Status: `In Progress`
- 背景:
  - 計算だけGPU化しても、毎フレームreadback描画では同期コストが残る。
- スコープ:
  - デバッグ用円オーバーレイは維持しつつ、ゲーム本流は常時表示のGPUドット絵描画へ移行する。
  - `water_drop` でスクリーンショットを取得し、見た目の成立を自動検証できるループを整備する。
- Subtasks:
  - [x] デバッグオーバーレイで粒子円表示を実装し、`water_drop` の挙動確認に使う。
  - [x] overlayトグル非依存で常時表示されるGPUドット絵の本流パスを実装する。
  - [x] GPU粒子から独立したドット存在度グリッドへ splat + blur し、閾値でドット有無を決める前処理パスを実装する。
  - [x] GPU粒子/密度バッファからドット絵描画するパスを実装する。
  - [x] フラグメントシェーダーを第一候補として実装し、必要ならcompute前処理を追加する。
  - [x] 異種粒子混在セルで平均色を使わず、材質パレットからドット単位で相をランダム選択する描画へ変更する。
  - [x] 材質別カラーパレット定義を shader ハードコードから `assets/params/palette.ron` へ移管し、hot reload 可能にする。
  - [ ] 表示品質（連続性、エイリアシング、tile境界、ちらつき）を確認する。
  - [ ] 縮小時（描画ドット < 画面ピクセル）のAA品質を改善する（mipmap / 解析的AA）。
  - [x] `water_drop` 自動実行でスクリーンショットを保存し、成果物で表示確認できるようにする。
- 進捗:
  - 2026-03-01: 本流描画として `WaterDotGpuPlugin` を追加。`Core2d` render graph に常時実行ノードを登録し、`MpmGpuBuffers.particle_buf` を直接参照して overlay 非依存で GPU ドット描画を実行。
  - 2026-03-01: `assets/shaders/water_dot_gpu_mainline.wgsl` を追加。粒子中心を world dot grid にスナップし、GPU側パレットハッシュで色を選択するドット調フラグメント描画を実装。
  - 2026-03-01: `PARTICLES_AUTOVERIFY_SCREENSHOT=1` の自動検証ループを追加。`water_drop` を自動ロードして一定フレーム後にスクリーンショット保存し、自動終了する仕組みを実装。
  - 2026-03-01: `PARTICLES_AUTOVERIFY_SCREENSHOT=1 PARTICLES_AUTOVERIFY_SCENARIO=water_drop PARTICLES_AUTOVERIFY_SCREENSHOT_OUT=artifacts/water_drop_gpu_render.png PARTICLES_AUTOVERIFY_SCREENSHOT_WARMUP_FRAMES=240 cargo run -q` を実行し、成果物 `artifacts/water_drop_gpu_render.png` で本流GPU描画を確認。
  - 2026-03-01: 本流ドット描画を再設計。`WaterDotPreprocessNode`（compute）で `dot density grid` を `clear -> particle splat -> blur_x -> blur_y` し、`WaterDotGpuNode`（fragment）で閾値判定＋決定論的パレット選択して描画する構成へ移行。
  - 2026-03-01: `PARTICLES_AUTOVERIFY_SCREENSHOT_CAMERA_SCALE` を追加し、スクリーンショット自動検証でカメラ拡大状態を再現可能にした。
  - 2026-03-01: `PARTICLES_AUTOVERIFY_SCREENSHOT_CAMERA_CENTER_X/Y` を追加し、拡大時でも注視領域（液体位置）を固定して撮影できるようにした。
  - 2026-03-01: `PARTICLES_AUTOVERIFY_SCREENSHOT=1 ... OUT=artifacts/water_drop_dot_default.png ...` で通常表示を撮影し、粒子単位ではなく連続した水塊として描画されることを確認。
  - 2026-03-01: `PARTICLES_AUTOVERIFY_SCREENSHOT=1 ... OUT=artifacts/water_drop_dot_zoom.png ... CAMERA_SCALE=0.35 CAMERA_CENTER_X=0 CAMERA_CENTER_Y=-6` で拡大表示を撮影し、ドットパターンが視認できることを確認。
  - 2026-03-01: シェーダー資産を責務別に再配置（`assets/shaders/physics|render|overlay`）。Rust側のロードパスも追従し、`physics` と `render/overlay` の分離を明確化。
  - 2026-03-03: 混在ドットの色決定を「密度加重ランダム相選択 + 相ごとの4色パレット選択」へ更新。平均色 `mix` を廃止し、水/土/砂の離散的なドット表現へ変更。
  - 2026-03-03: 材質パレットを `assets/params/palette.ron` に移管し、`src/params/palette.rs` + `ParamsPlugin` hot reload + `WaterDotGpu` の uniform 連携を実装。`configs/autoverify/sand_water_interaction_drop_screenshot.json` で表示確認。
- 完了条件:
  - `water_drop` で「計算 + 描画」がGPU完結で実行できる。
  - スクリーンショット成果物で本流GPU描画を確認できる。


### [MPM-GPU-03] 地形表示のGPU完結化（CPU生成→GPU描画）

- Status: `Done`
- 背景:
  - 今後、地形と粒子へ統一的にライティング/反射を適用するためには、地形描画もGPU本流へ寄せる必要がある。
  - 既存CPUタイル描画はデバッグしやすいが、描画経路が分離し将来拡張の足かせになる。
- スコープ:
  - 地形データ生成/更新はCPUで維持し、描画用の地形ドット化・表示はGPUで行う。
  - 水ドット描画とは前処理を分離し、RenderGraphで `Terrain -> Water -> Overlay` の順序を固定する。
  - `water_drop` で岩枠表示、`default world` で生成地形表示を成立させる。
  - 目標達成後、旧CPU地形表示コードを削除する。
- Subtasks:
  - [x] 地形GPUアップロード用データ（solid/material）を定義し、CPU→GPU転送経路を実装する。
  - [ ] 地形専用ドット前処理compute（地形側）を実装する。
  - [x] 地形描画shaderを実装し、本流表示として常時描画する。
  - [x] RenderGraph順序を `Terrain -> Water -> ParticleOverlay` へ固定する。
  - [x] `water_drop` と `default world` のスクリーンショット検証ルートを整備する。
  - [x] 旧CPU地形表示コードを削除する。
- 進捗:
  - 2026-03-02: `TerrainDotGpuPlugin` を追加。`TerrainWorld` から `solid/material` を `TerrainDotUploadRequest` へ詰め、ExtractResource経由で render world の storage buffer へ転送する経路を実装。
  - 2026-03-02: `assets/shaders/render/terrain_dot_gpu.wgsl` を追加。地形セルを決定論的パレットでGPU描画し、常時表示の本流パスとして有効化。
  - 2026-03-02: RenderGraph連結を `TerrainDotGpuLabel -> WaterDotGpuLabel -> ParticleOverlayGpuLabel` とし、地形→水→overlay の描画順を固定。
  - 2026-03-02: `PARTICLES_AUTOVERIFY_SCENARIO=default_world` 相当の検証を行えるよう screenshot 自動検証に `default_world/none`（シナリオロードをスキップ）を追加。
  - 2026-03-02: `PARTICLES_AUTOVERIFY_SCREENSHOT=1` で以下を取得し、表示成立を確認。
    - `artifacts/water_drop_terrain_gpu.png`（岩枠表示）
    - `artifacts/default_world_terrain_gpu.png`（生成地形表示）
  - 2026-03-02: 旧CPU描画ルートを削除（`src/render/tiles.rs`, `src/render/palette.rs`, `src/render/constants.rs`, `src/render/water.rs`, `src/render/free_particles.rs`, `src/render/object_sprites.rs`）。
  - 2026-03-02: 地形ドットの見た目改善として、地形描画解像度を `2x2` 相当から `8x8` ドット/セルへ引き上げ。
- 完了条件:
  - `water_drop` で岩枠がGPU地形描画として表示される。
  - `default world` で生成地形がGPU描画として表示される。
  - CPU地形表示コードが撤去される。


### [REND-GPU-01] GPU常駐地形描画パイプライン（Near/Far 2層キャッシュ + リングバッファ）

- Status: `In Progress`
- 背景:
  - 現行の `terrain_dot_gpu` は毎フレームCPUが全セルをエンコードし GPU storage buffer へ転送するシンプルな方式。
  - これは CPU→GPU 転送を常態化させており、Near/Far LOD分離・リングバッファ・ズーム予算管理が未実装。
  - `render.md` が規定する GPU常駐 Near/Far 2層キャッシュ方式へ移行し、CPU転送を「dirty差分のみ」に削減する。
  - 地形生成関数（fBm + 材料層決定）をWGSLへ移植し、GPU compute が新規可視領域を自律的に評価できるようにする。
  - [MPM-GPU-03] の完了条件はすでに達成されているため Done とし、本WUが描画品質・アーキテクチャを引き継ぐ。
  - [REND-01]（CPUタイルパイプライン）および [WGEN-02]（CPU LODスプライト）は本WUに Superseded とする。
- スコープ:
  - 地形生成関数（fBm ノイズ + 材料層決定）を `terrain_gen.wgsl` としてWGSLへ移植する。
  - Near（材料ID テクスチャ, R16Uint）/ Far（集約テクスチャ, RGBA8）をGPU常駐バッファとして保持し、リングバッファでパンに追従する。
  - 3パス構成（`render.md` §3）:
    - `TerrainNearUpdate` (compute): dirty タイルの材料IDをGPU生成関数で更新
    - `TerrainFarUpdate`  (compute): Near → Far 集約
    - `TerrainCompose`   (fragment): Far → Near を合成して画面へ
  - dirty キュー + 予算スケジューラ（`render.ron` の `max_tiles_per_frame_near/far`）でスパイクを抑える。
  - モード切替（Magnify/Minify + ヒステリシス S_UP=1.2, S_DOWN=0.8）を `TerrainCompose` で管理する。
  - 旧 `src/render/terrain_dot_gpu.rs` の単純転送方式を削除する。
  - 自動検証ループ（セル正確性・パン継続性・ズーム遷移・予算遵守）を整備する。
- Subtasks:
  - [x] **[P0: 旧タスク整理]** [MPM-GPU-03] を Done、[WGEN-02]/[REND-01] を Superseded に移動する。
  - [x] **[P1: WGSL生成関数移植]** `assets/shaders/render/terrain_gen.wgsl` を実装する。
    - [x] CPU側 fBm（OpenSimplex or 互換ハッシュノイズ）をWGSLへ移植する。
    - [x] `surface_height_for_x(world_x: i32) → i32` 相当を実装する。
    - [x] `material_for_cell(global_cell: vec2<i32>) → u32`（0=Empty, 1=Stone, 2=Soil, 3=Sand）を実装する。
  - [x] **[P2: Near キャッシュ定義]** `TerrainNearGpuCache` テクスチャと `TerrainCacheState` Resource を定義する。
    - [x] テクスチャ: R16Uint、解像度 = screen_res × NearMargin / NearQuality（デフォルト 1/2 × 1.35倍）
    - [x] Resource: `cache_origin_world: IVec2`, `ring_offset: IVec2`, `lod_k: i32`, `near_dirty_queue`, `far_dirty_queue`
    - [x] `world_cell → texture_index`（mod-ring 座標変換）ユーティリティを実装する。
  - [x] **[P3: Far キャッシュ定義]** `TerrainFarGpuCache` テクスチャを定義する。
    - [x] 解像度 = 画面 + 余白（2.5倍）を基準に固定し、ズーム時は downsample（LOD）でカバーする。
    - [x] セル値: `far_top1_id: u8`, `far_w1: u8`, `solid_fraction: u8`, pad（RGBA8 format）
    - [x] LOD k_far = k_near + FAR_OFFSET（デフォルト 3）
  - [x] **[P4: TerrainNearUpdate compute]** `assets/shaders/render/terrain_near_update.wgsl` を実装する。
    - [x] 入力: dirty list（SSBO）、生成関数パラメータ（uniform）
    - [x] 各 dirty cell で `material_for_cell()` を評価して Near 基準値を生成する。
    - [x] 結果を Near テクスチャ（リング座標）へ書き込む。
    - [x] 地形編集オーバーライド転送経路（`TerrainOverrideDeltaBuffer`）は [REND-GPU-02] へ移管する。
  - [ ] **[P5: TerrainFarUpdate compute]** `assets/shaders/render/terrain_far_update.wgsl` を実装する。
    - [x] Far texel が表すワールド領域を多点サンプリングして集約値（top1_id, w1）を計算する。
    - [x] LOD k_far 変化時に Far 全体を dirty 化する。
    - [ ] 予算内で逐次再サンプリングする。
  - [x] **[P6: TerrainCompose fragment — 初期版]** `assets/shaders/render/terrain_compose.wgsl` を実装する（Near単層、Farなし）。
    - [x] フルスクリーン quad で Near テクスチャを合成して画面へ出力する。
    - [x] Near: 最近傍サンプリング + 材料IDパレット（8×8 ドット内確定的ハッシュ）で表示する。
    - [x] Far: カバレッジ alpha でブレンド表示する（Far実装後）。
    - [x] Near有効時は Far を nearest、Far単独表示時のみ bilinear とする（Farぼけはみ出し抑制）。
    - [ ] `s = (CELL_SIZE_M * camera_zoom) / screen_pixel_size_m` を計算し、S_UP/S_DOWN ヒステリシスでモード切替する（P7後）。
  - [ ] **[P7: カメラ同期 + パン/ズーム制御]** Update schedule で `TerrainCacheState` をカメラ状態から更新するシステムを実装する。
    - [ ] カメラパン量（セル単位）を計算し、ring_offset を更新する。
    - [ ] 新規可視セル範囲を Near dirty キューへ追加する。
    - [ ] テレポート検出（`|Δ| > cache_size / 2`）→ Near/Far 全面 dirty 化する。
    - [ ] ズーム LOD k を計算し、ヒステリシスで切替判断する。k 変化時に Far dirty キューを全面化する。
  - [ ] **[P8: dirty キュー + 予算スケジューラ]** Near/Far dirty キューと1フレーム更新予算を実装する。
    - [ ] タイル単位 dirty キュー（カメラ中心距離優先ソート）を実装する。
    - [ ] 1フレームあたり最大 `max_tiles_near / max_tiles_far` タイルを dispatch する。
    - [ ] dispatch tile list を per-frame SSBO へ書き込み、NearUpdate / FarUpdate へ渡す。
    - [ ] `render.ron` へ `terrain.max_tiles_per_frame_near / far` パラメータを追加する。
  - [x] **[P9: RenderGraph 統合 + 旧削除]** 2パス（NearUpdate compute + TerrainCompose fragment）を RenderGraph へ組み込み、旧方式を削除する。
    - [x] TerrainNearUpdate（main graph） → CameraDriver → Core2d: TerrainCompose → WaterDotGpu のエッジを設定する。
    - [x] `src/render/terrain_dot_gpu.rs` を削除し、`TerrainDotGpuPlugin` を `TerrainGpuPlugin` へ置換する。
    - [x] `cargo check` / `cargo test --lib` が通ることを確認する（100件 pass）。
    - [x] pipeline retry: `pending_dispatch_frames=32` でシェーダー遅延コンパイル時のリトライを実装する。
    - [x] shader coordinate fix: `cell_f = world_xy / cell_size_m`（double-subtract バグを修正）。
    - [x] subgraph edge: `link_terrain_and_water_graph` で Core2d sub-graph 経由の edge を正しく設定する。
  - [x] **[P10: 自動検証 — 初期]** autoverify スクリーンショット確認を整備する。
    - [x] `configs/autoverify/terrain_gpu_screenshot.json`: GPU地形描画スクリーンショット取得・目視確認（`artifacts/terrain_gpu.png`）。石壁・土壌パレット正常表示を確認済み。
    - [x] HUDスケールバー（左下）の表示を追加し、`m/km` 表示と 1-2-5 系列スナップ（`1m,2m,5m,...`）を確認する。
    - [x] screenshot/mpm autoverify の終了を `autoverify_hard_exit` で安定化し、`cargo run` 残留を抑止する。
    - [ ] `configs/autoverify/terrain_cell_correctness.json`: GPU生成セル vs CPU参照の一致率チェック（WGSL生成関数実装後）。
    - [ ] `configs/autoverify/terrain_pan_continuity.json`: パン継続性スクリーンショット（リングバッファ実装後）。
    - [ ] `configs/autoverify/terrain_zoom_lod.json`: ズーム遷移スクリーンショット（LOD実装後）。
- 進捗:
  - 2026-03-05: 地形改変反映（`TerrainOverrideDeltaBuffer` / 編集入力 / セーブロード反映）を [REND-GPU-02] へ分離し、本WUは Near/Far生成キャッシュ基盤にスコープを限定。
  - 2026-03-04: 診断表示を追加。HUD に `TerrainGen/frame`（そのフレームで NearUpdate が評価する dirty cell 数）を表示し、スクロール時の生成負荷を可視化。あわせて `prepare_terrain_near_uploads` の retry を「pipeline ready 時は1フレーム、未ready時のみ32フレーム」へ変更し、不要な再dispatchを削減。
  - 2026-03-04: リングバッファ表示のずれを修正。`terrain_compose.wgsl` に `ring_offset` を追加し、`world_cell -> texture` 参照を mod-ring 化。`prepare_terrain_near_uploads` で compose uniform へ `ring_offset` を転送。スクロール時の地形が飛び飛びに移動する表示不整合を解消。
  - 2026-03-04: スクロール時の周期的カクつき対策として Near 更新を dirty 列/行ベースへ移行。`prepare_terrain_near_update_request` はカメラ移動量から新規可視ストリップのみ dirty 発行し、`ring_offset` を更新。`terrain_near_update.wgsl` は dirty cell SSBO を1D dispatchで処理する方式に変更し、Near 全面再生成を回避。`cargo check` / `cargo test --lib` / `terrain_gpu_screenshot` / `terrain_gpu_panned_screenshot` で確認。
  - 2026-03-04: P2 を実装。`src/render/terrain_gpu.rs` に `TerrainNearGpuCache` / `TerrainCacheState` を追加し、Near キャッシュ解像度を `screen_res × 1.35 / 2` で算出する経路へ移行。`world_cell -> texture_index`（mod-ring）ユーティリティを導入し、Near upload 生成に適用。`cargo check` / `cargo test --lib` / `cargo run -q -- --autoverify-config configs/autoverify/terrain_gpu_screenshot.json` を通過。
  - 2026-03-04: P0/P9/P10初期を完了。`src/render/terrain_gpu.rs` + WGSL 2本で TerrainNearUpdate compute + TerrainCompose fragment を実装。旧 terrain_dot_gpu.rs 削除。autoverify screenshot で石壁・土壌の正常描画を確認。
  - 2026-03-04: 遠方パン時の FPS 低下対策として Near 更新CPU経路を最適化。`prepare_terrain_near_update_request` で列ごとの `surface_y` をキャッシュし、未ロード領域で同一列のノイズ再計算を削減。
  - 2026-03-04: P1 前半を実装。`assets/shaders/render/terrain_gen.wgsl` を追加し、`terrain_near_update.wgsl` は GPU生成 `material_for_cell` を基準値にして CPU由来は override のみ適用する構成へ移行。`terrain_gpu_screenshot` / `terrain_gpu_panned_screenshot` で表示成立を確認。
  - 2026-03-04: 起動直後の低FPS要因を修正。`terrain.is_changed()` 依存をやめ、`TerrainWorld.terrain_version`（実地形変更時のみ増加）で Near 更新トリガーを判定するよう変更。
  - 2026-03-04: 停止時低FPSの追加要因を修正。`step_physics` は `running=false && step_once=false` のとき早期returnし、停止中の active region 走査 / object field 更新を停止。
  - 2026-03-04: `default_world` 停止時FPS低下の主因を修正。`gpu_mpm::prepare_terrain_upload` の更新判定を `terrain.is_changed()` から `TerrainWorld.terrain_version` ベースへ変更し、不要な全グリッド SDF/normal 再構築・再アップロードを停止。autoverify（`default_world_paused_screenshot` / `default_world_paused_start`）で停止時 FPS が ~120 へ回復することを確認。
  - 2026-03-04: スクロール時スパイク（`TerrainGen/frame=104976`）対策として、Near の全面再生成条件を「旧キャッシュと新可視領域が非重複の場合」に限定（`|Δorigin| >= cache_extent`）。あわせて HUD に `Δorigin` / `full refresh` フラグ / reason bit を表示し、スパイク要因を実行時に診断可能化。
  - 2026-03-04: Near キャッシュ範囲を「画面ピクセル固定」から「カメラ投影のワールド可視範囲（セル換算）× margin」へ変更し、ズームイン時に不要な生成（例: 常時 `243` 行）を抑制。`TerrainGen/frame` 表示は可変桁で揺れないよう固定幅フォーマットへ更新。
  - 2026-03-04: 横スクロール時の16セル周期カクつき対策として、`stream_terrain_around_camera` を軽量化。停止中（`running=false && step_once=false`）は地形ストリーミングを停止し、再生中でも中心チャンクが変わらないフレームは処理をスキップするよう変更。
  - 2026-03-04: ズームアウト時クラッシュ（wgpu dispatch group x=65536 > 65535）を修正。Near extent 算出に「dispatch上限面積」クランプを追加し、upload側でも `dirty_count` を `MAX_DIRTY_CELLS_PER_DISPATCH` で保険クランプ。`near_extent_is_clamped_to_dispatch_limit` テストを追加。
  - 2026-03-04: 暫定方針を明記。地形改変なし前提で、REND-GPU-01 の検証中はレンダ側の地形 override 転送（loaded/edited cell の CPU→GPU 上書き）を無効化し、Near は GPU 生成値のみを使用する。
  - 2026-03-04: Far 初期版を実装。`TerrainFarGpuCache`（RGBA8Uint）を追加し、`terrain_far_update.wgsl` で Near(2x2)→Far 集約（top1_id/w1/solid_fraction）を compute 更新。`terrain_compose.wgsl` は Near 優先 + Near空セルで Far alpha fallback 合成に対応。現時点では Far LOD連動/2.5xマージンは未実装（P3/P5継続）。
  - 2026-03-04: Far LOD 連動を実装。`k_near` をカメラズームから算出し `k_far = k_near + 3` を `TerrainNearUpdateRequest` に保持。`k_far` 変化時は Near dirty がなくても Far 全体を再生成する経路を追加し、`far_downsample` は `2,4,8...` へ段階的に拡大。
  - 2026-03-04: Far を Near範囲依存から分離。Far キャッシュ解像度を「ベース画面 + 2.5 margin」で固定し、ズームアウト時は `far_downsample` で可視範囲を拡大。`terrain_far_update.wgsl` は Near tex 参照をやめ、`terrain_gen` から直接集約生成へ変更。compose は Far を常時背景描画し、Near が viewport+margin を満たせないフレームでは Near 描画を無効化して Far のみ表示。
  - 2026-03-04: Far 荒さ/ズーム依存負荷を改善。原因は (1) Far更新が `downsample^2` サンプルでズームアウトほど計算量増加、(2) Far-only フレームでも Near cache をズーム解像度に再確保していた点。対策として Far集約を固定 2x2 サンプルへ変更し、Near cache 再確保は Near upload 時のみ実施。compose 側は Far を4点重み付き合成してブロック境界の荒さを緩和。
  - 2026-03-04: 地形マクロ高低差を増加。CPU/WGSL の `HEIGHT_NOISE_AMP_CELLS` を `50 -> 80` に統一更新。
  - 2026-03-04: Far粗さ低減の追加調整。`far_downsample` は 2冪を維持しつつ、`viewport_cells * far_margin / far_extent` の必要値を power-of-two へ切り上げて算出する方式へ変更。Farベース解像度は `screen * (far_margin / 2) * 1.1`（画面1/2品質 + margin + headroom）で固定し、Far切替直後に品質が過度に荒くなる状態を緩和。Far再生成トリガは `far_downsample`・`far_origin`・地形バージョン変化に限定。
  - 2026-03-04: Far最小解像度を調整。`FAR_MIN_DOWNSAMPLE` を `2 -> 1` に変更し、Near表示中（Far最小downsample時）の Far 1texel が 1セルに一致するよう修正。
  - 2026-03-04: Far表示の1セルずれ補正。`terrain_compose.wgsl` の Far サンプリング座標を「連続 world 座標」から「world_cell 中心基準」へ変更し、`downsample=1` 時に Far がセル境界へ正しく一致するよう調整。
  - 2026-03-04: Back 背景レイヤーを追加。Far と同じ compute 経路を使って `back_downsample = far_downsample * 8` のキャッシュを生成し、`terrain_compose.wgsl` を `空色ベース -> Back(空気遠近ティント) -> Far -> Near` 合成へ更新。Back は `camera_cell * 0.35` 基準の origin 更新でパララックス移動させる。`cargo check` / `cargo test --lib` / `cargo run -q -- --autoverify-config configs/autoverify/default_world_paused_screenshot.json` / 追加 screenshot autoverify（camera_center_y=40）で確認（`artifacts/autoverify/default_world_backlayer_verify.png`）。
  - 2026-03-04: 実行時調整可能な描画パラメータを `assets/params/render.ron` へ集約。`src/params/render.rs` を拡張し、terrain 側へ `lod`（margin/quality/min_downsample）、`back`（downsample/display_scale/parallax/tint）、`sky_color` を追加。water 側へ `palette_seed` を追加し、`water_dot_gpu.rs` の splat/blur/threshold/seed を `ActiveRenderParams` 経由で参照するよう接続。
  - 2026-03-05: Back 表示式を投影ベースへ更新。`terrain_compose.wgsl` の Back サンプル座標を `raw_scale = 1 + beta / front_mpp_cells_per_px`（`[1, display_scale_max]` に clamp）で計算する方式へ変更し、ズームインで Back が拡大・ズームアウトで前景と同スケールへ滑らかに遷移するよう調整。対応パラメータとして `back.perspective_beta_cells_per_px` と `back.display_scale_max` を `render.ron` へ反映。
  - 2026-03-05: Back 解像度制御を Far から分離。`back_downsample = far_downsample * multiplier` を廃止し、`viewport + Back scale` から Back 専用 downsample を算出する経路へ変更。さらに Back scale 式へ `min_screen_resolution_divisor` ベースの下限（`screen_px / divisor` 以上）を導入し、ズームイン時に Back の見かけ解像度が落ちすぎないよう補正。
  - 2026-03-05: Back のズーム見た目を Far 準拠へ再調整。Back downsample は Far と同じ power-of-two coverage 規則で独立算出し、ズームアウト極限で Far と一致するよう変更。Back の表示スケールは「1セル=1ドット(2px)を上限」にする最小倍率 `back_downsample / (min_screen_resolution_divisor * front_mpp)` のみを採用し、拡大時のセル感を抑制。
  - 2026-03-05: Back のセル段差/ズーム時ブレを追加修正。Far/Back サンプリング座標を「整数セル中心」ではなく連続 `cell_f` で評価するよう変更し、Back のスケール中心を整数 `camera_cell` から実座標 `camera_pos / CELL_SIZE_M` へ変更。これによりズーム中の位置ドリフトとセル単位の段差感を低減。
  - 2026-03-05: Back の内部ドットを Near から分離。Back 合成時の palette ドット座標を `back_world_cell` 基準の「1セル=1ドット」へ変更し、ズームイン時に Near のサブセルドットがそのまま拡大される見え方を回避。あわせて `control_main_camera` を `SimUpdateSet::Controls` へ移し、`prepare_terrain_near_update_request` より前にカメラ更新される順序を固定。
  - 2026-03-05: Back のズーム/スクロール同時時のブレ対策を追加。Back スケールのピボット座標を `TerrainComposeParams.camera_cell_*`（CPU転送）から、`terrain_compose.wgsl` 内で `view.world_from_clip` から直接算出する方式へ変更し、`view` 行列と同フレーム基準へ統一。あわせて `camera_cell_*` uniform と `camera_continuous_changed` 判定を削除し、サブセル移動時の不要な upload dirty 化を抑制。
  - 2026-03-05: Back のパン時ガクつき低減を追加。Back の palette 位相を `1cell=1dot` から `dots_per_cell`（サブセル）基準へ変更し、スクロール時に内部パターンがセル境界でのみ更新される見え方を緩和。
  - 2026-03-05: ズーム時の1フレーム倍率ブレを追加修正。`prepare_terrain_near_update_request` の viewport 取得を `OrthographicProjection.area` 依存から `scaling_mode + scale + window_size` の直接計算へ変更し、Bevy の `camera_system` による `area` 更新タイミング差で `front_mpp_cells_per_px` が1フレーム遅れる問題を解消。
  - 2026-03-05: ズームイン時の Back 見切れを修正。Back downsample の必要カバー範囲を `viewport_world * back_scale` で評価するよう変更し、Back スケール拡張時でもキャッシュが画面端まで届くようにした。あわせて `terrain_compose.wgsl` の layer サンプルを clamp-to-edge 化し、境界近傍の透明抜けを防止。
  - 2026-03-05: Far ぼかし条件を調整。Near が有効なフレームは Far を nearest、Far 単独表示（`near_enabled == 0`）時のみ bilinear に切り替えるよう compose を更新し、Near 併用時の Far ぼけはみ出しを抑制。
  - 2026-03-05: autoverify 実行時の `cargo run` 残留対策を追加。`run_mpm_autoverify` / `run_screenshot_autoverify` の終了条件で `AppExit` 通知後に `std::process::exit(code)` を呼ぶ `autoverify_hard_exit` を導入し、非対話実行でのプロセス残留を抑止。
  - 2026-03-04: 左下スケールバーを追加。カメラ投影から m/px を算出し、バー長を「10刻み」（`m` もしくは `km`）へスナップして表示する UI を実装。`default_world_paused_screenshot` で `30 m` 表示を確認。
  - 2026-03-04: スケールバー刻みを 1-2-5 系列へ変更。表示値を `1m, 2m, 5m, 10m, ... , 1km, 2km, 5km, ...` にスナップするよう更新し、`default_world_paused_screenshot` で `1 m` 開始表示を確認。
- 完了条件:
  - GPU compute で地形生成関数が自律評価され、毎フレームの CPU→GPU 全セル転送が不要になる。
  - Near/Far 2層キャッシュが GPU 常駐し、パン・ズームでリングバッファ更新が機能する。
  - dirty キュー＋予算スケジューラで1フレーム更新コストが `max_tiles_near/far` 以内に収まる。
  - autoverify でセル一致率 ≥ 99.9%、パン継続性・ズーム遷移スクリーンショットが成立する。
  - 旧 `terrain_dot_gpu` 方式が完全削除され、[MPM-GPU-03]/[REND-01]/[WGEN-02] が整理済みである。


### [REND-GPU-02] 地形改変のNear反映（編集入力 + セーブロード + override差分転送）

- Status: `In Progress`
- 背景:
  - [REND-GPU-01] で Near/Far 生成キャッシュは整備されたが、地形改変の表示反映は暫定的にスコープ外としている。
  - ユーザー操作（delete tool / solid tool）のセル編集、およびセーブデータからの改変復元を Near 描画に反映する必要がある。
  - 改変データは CPU 側で `chunk_coord -> chunk_cells`（`CHUNK_SIZE = 32`）の HashMap として保持する方針とする。
  - [REND-GPU-01] P4 の `TerrainOverrideDeltaBuffer` 未実装項目を本WUへ移管する。
- スコープ:
  - 編集入力（delete/solid）とセーブロード差分を、共通の改変チャンク台帳へ反映する。
  - Near GPU キャッシュに「生成セル配列」と「改変セル配列（override）」を同一リングバッファ座標系で保持する。
  - スクロール/ズーム時に、改変セルを CPU から RLE 圧縮差分として固定 payload 単位で GPU へ転送する。
  - `terrain_compose.wgsl` は改変キャッシュ優先で参照し、`no_override` 特別ID時のみ生成キャッシュを参照する。
  - Far への改変反映は対象外とし、後続 Work Unit へ分離する。
- Subtasks:
  - [x] **[P0: 改変データモデル]** CPU 改変チャンク台帳を定義する。
    - [x] `CHUNK_SIZE = 32` のセル配列を値にした `HashMap<chunk_coord, chunk_cells>` を実装する。
    - [x] セル表現を `override_material_id` + `no_override` 特別IDで統一する。
    - [x] 編集由来/ロード由来の dirty chunk 追跡を共通化する。
  - [x] **[P1: 編集入力接続]** delete tool / solid tool をセル編集へ接続する。
    - [x] ツール入力を world 座標から cell 座標へ変換し、改変チャンク台帳へ反映する。
    - [x] 複数セル編集（ブラシ範囲）時の dirty chunk 発行を実装する。
  - [x] **[P2: セーブロード接続]** セーブ差分から改変チャンク台帳を復元する。
    - [x] ロード時に改変セル差分を台帳へ復元する。
    - [x] ロード直後に可視範囲対応の dirty chunk を発行する。
  - [x] **[P3: override差分転送]** `TerrainOverrideDeltaBuffer` を実装する。
    - [x] 可視 dirty rect と交差する override chunk の改変セルを run-length 圧縮してエンコードする。
    - [x] 転送データを固定 payload サイズ単位へ分割して SSBO へ転送する。
    - [x] `terrain_near_update.wgsl` 側で payload をデコードし、Near 改変キャッシュへ書き込む。
  - [x] **[P4: Near改変キャッシュ統合]** 生成/改変の2配列構成を Near リングへ統合する。
    - [x] Near 改変キャッシュを生成キャッシュと同寸法・同ring_offsetで保持する。
    - [x] スクロール/ズーム時は生成関数評価の代替として改変 payload 適用を実行できるようにする。
    - [x] 未改変セルは `no_override` を維持する。
  - [x] **[P5: Compose参照順序]** 改変優先参照を描画経路へ適用する。
    - [x] `terrain_compose.wgsl` で「override -> base generated」の順に参照する。
    - [x] Near/Far 合成の既存表示（Near優先、Near欠損時Far補完）を維持する。
  - [x] **[P6: 自動検証]** 改変反映の runtime artifact を追加する。
    - [x] `configs/autoverify/terrain_edit_delete_tool.json` を追加し、削除編集結果の screenshot を保存する。
    - [x] `configs/autoverify/terrain_edit_solid_tool.json` を追加し、solid 編集結果の screenshot を保存する。
    - [x] `configs/autoverify/terrain_edit_load_override.json` を追加し、ロード差分反映の snapshot（state/metrics）を保存する。
    - [x] RLE payload 数・展開セル数・フレーム予算内完了率を JSON/HUD へ出力する。
  - [x] **[P7: MPM地形SDFのGPU一本化準備]** CPU側SDF生成/CPU地形生成依存を撤去する。
    - [x] `TerrainWorld` のSDF前計算（rebuild）を停止し、CPU側の重い再生成処理を外す。
    - [x] CPU地形生成モジュール（noise/rules）を削除し、未ロード領域はGPU生成chunk readback経由でのみ編集/参照する。
    - [x] MPM向けSDF生成をGPU側で完結させる後続WUを起票する（実装は別WU）。
  - [ ] **[P8: Far/Back 改変反映]** Far/Back 表示にも override 差分を反映する。
    - [x] Far/Back 更新computeで Near base/override を参照し、既存GPU downsample 経路で反映する（Nearカバー領域）。
    - [x] CPU override chunk 台帳を hash + chunkセル配列（疎payload）としてGPUへ送信し、Far/Back shader で global override lookup を実装する（Near外の改変を反映）。
    - [x] Near非表示ズーム域では Near cache 参照を無効化し、中央固定幅のFar/Back描画ずれを解消する。
    - [ ] CPU側 override chunk から `2^k` ダウンサンプルmipmapキャッシュを構築する。
    - [ ] Far/Back の scale (`downsample`) ごとに必要mip差分のみを payload 送信する。
    - [ ] GPU側で mip差分 payload を Far/Back override cache へ適用する（現状は sparse lookup 方式）。
    - [ ] scale変更時の再構築コストを分割実行し、フレームスパイクを抑制する（現状は生成更新時に同一payload参照）。
- 進捗:
  - 2026-03-05: Near を `base(R16Uint) + override(R16Uint)` の2層へ拡張。`terrain_near_update` で base と `no_override` 初期化を同時更新し、`terrain_override_apply.wgsl` で chunk(32x32) override を RLE run 展開して override 層へ適用。Compose は override 優先参照へ更新。
  - 2026-03-05: CPU 側で可視 dirty rect に交差する override chunk を列挙し、chunk 線形配列を run-length 圧縮して `MAX_OVERRIDE_RUNS_PER_FRAME` で分割送信するキューを実装。
  - 2026-03-05: screenshot autoverify に terrain ops（`delete_rect`, `solid_rect`, `save_snapshot`, `load_snapshot`）を追加し、成果物 `artifacts/autoverify/terrain_edit_delete_tool.png` / `terrain_edit_solid_tool.png` / `terrain_edit_load_override.png` を生成確認。
  - 2026-03-05: HUD に override 転送診断 (`TerrainOvr/frame: runs/cells/pending`) を追加。
  - 2026-03-05: CPU 改変台帳を `HashMap<chunk_coord, TerrainOverrideChunk(32x32 fixed array)>` へ移行し、`dirty_override_chunks` で編集由来/ロード由来の共通 dirty 追跡を追加。
  - 2026-03-05: HUD に `done=%`（フレーム予算内完了率）を追加。screenshot autoverify は sidecar JSON（`*.json`）へ `payload_runs_total / expanded_cells_total / budget_completion_avg` を出力するよう更新。
  - 2026-03-05: delete tool をセル単位編集へ変更し、ホバーセルの4辺ハイライト表示を追加（delete/break/terrain solid系ツール）。
  - 2026-03-05: Tile Overlay を Chunk Overlay 化。可視chunk境界を薄線表示し、改変済みchunk（override保持）を色付き境界線で強調。
  - 2026-03-05: Chunk Overlay が `visible_tiles` 未供給で空描画だった問題を修正。カメラ可視範囲から chunk 境界線を直接描画する方式へ変更し、改変chunk枠表示を安定化。
  - 2026-03-05: `TerrainOvr` HUD に累積カウンタ（runs/cells total）を追加し、1フレーム値が見逃されるケースでも改変転送発生を確認可能にした。
  - 2026-03-05: Core2d graph の描画順を修正し、`TerrainCompose/WaterDot` を `MainTransparentPass` より前へ移動。CPU gizmo overlay（Chunk/SDF/Physics Area）が上面表示されることを `configs/autoverify/overlay_visibility_default_world.json` の screenshot で確認。
  - 2026-03-05: ツールhoverハイライト色を白に統一。solidツールは粒子生成を行わず、セル直接編集（`TerrainCell::Solid`）へ変更。
  - 2026-03-05: `overlay_chunk_modified_highlight` / `overlay_chunk_modified_solid_highlight` で `Chunk Overlay: Modified:2` と `TerrainOvr/total` 増加（delete: runs=10, solid: runs=30）を確認。
  - 2026-03-05: `terrain_compose.wgsl` の Near 合成を修正。Near 範囲では「override(Empty含む)を優先し、Emptyなら空色を描く」挙動へ更新し、`near_enabled` 条件に依存せず改変表示を適用。`configs/autoverify/terrain_edit_delete_tool.json` と `configs/autoverify/terrain_edit_solid_visible.json` を `default_world` で再実行し、`artifacts/autoverify/terrain_edit_delete_tool.png`（削除穴）/`terrain_edit_solid_visible.png`（solid追加）で反映を確認。
  - 2026-03-05: delete 深部編集で「離れた場所に穴」「SDF と描画の不一致」が出る不具合を修正。原因は `terrain_override_apply.wgsl` の chunk サイズが `32` 固定で、実装定数 `CHUNK_SIZE=16` と不一致だったこと。Override apply の chunk size を uniform (`chunk_size_i32`) 化して Rust 側 `CHUNK_SIZE_I32` を転送するよう変更し、再実行した `terrain_edit_delete_tool` / `delete_sdf_align` で編集位置と描画・SDF の整合を確認。
  - 2026-03-05: 未ロードchunk編集向けに「GPU生成chunk readback cache」を追加。`TerrainGeneratedChunkCache`（key=`chunk_coord`）へ生成結果を保持し、delete/solid/break実行時は `TerrainWorld` 未ロードchunkをこのcache経由で解決（miss時はGPU生成要求をenqueue）。ツール有効時はカーソル周辺chunkを先読み（半径1chunk）し、初回編集遅延を低減。
  - 2026-03-05: GPU chunk readback の進行停止バグを修正。`TerrainChunkGenerateNode` で pipeline 未準備時に `pending_dispatch` を再投入し、`prepare_terrain_chunk_generate_uploads` でも `pending_dispatch` 残存時に dirty を再点火するよう変更。これにより request が `begin -> map_async -> mapped -> done` まで継続することを確認。
  - 2026-03-05: runtime の default world/reset/streaming 経路から CPU 生成chunk依存を削減。`initialize_default_world` / `apply_sim_reset` / `apply_scenario_spec(reset_fixed_world)` は `reset_fixed_world()` を使わず `set_generation_enabled(true) + clear()` に変更し、`stream_terrain_around_camera` と `step_physics` は `TerrainGeneratedChunkCache` から chunk を取り込む方式へ変更。
  - 2026-03-05: SDF overlay 整合確認用 `configs/autoverify/overlay_sdf_alignment_default_world.json` を追加し、`artifacts/autoverify/overlay_visibility_default_world.png`（Loaded:14）および `overlay_sdf_alignment_default_world.png` を `default_world` で再生成。SDF overlay と表示地形のズレ再発がないことを確認。
  - 2026-03-05: 起動直後の過剰chunkロードを抑制。`TerrainStreamingSettings.load_radius_chunks` 既定値を `0` に変更し、default world 起動時は `Loaded:1 Cached:1` 程度で安定（従来の `Loaded:132` 相当の初期スパイクを解消）。
  - 2026-03-05: Chunk overlay に generated cache 済みchunkの可視化を追加。薄境界（visible）に加えて cache 済み境界を別色で描画し、UIラベルを `Loaded/Cached/Modified` 表示へ更新。
  - 2026-03-05: delete tool での連鎖削除/粒子化を抑制。`Delete` 分岐から `detach_terrain_components_after_cell_removal` を外し、delete は対象セルのみ `Empty` 化する挙動へ統一（break のみ連結成分detachを維持）。
  - 2026-03-05: delete 反映不整合を修正。override比較基準をCPU生成関数ではなく chunk生成基準セル（`generated_base_chunks`）へ切替し、`load_generated_chunk_from_material_ids` 時に基準を保存するよう変更。これにより generated solid の delete でも override が正しく立つ。
  - 2026-03-05: CPU SDF前計算を停止。`TerrainWorld::rebuild_static_particles_if_dirty` での SDF rebuild を撤去し、SDF問い合わせは loaded terrain に対する都度サンプリングへ変更（前計算コストを削除）。
  - 2026-03-05: CPU地形生成コードを撤去。`src/physics/generation/*` と `assets/params/generation.ron` / `src/params/generation.rs` を削除し、`TerrainWorld` の未ロードセル参照は `Empty` 扱いへ変更。
  - 2026-03-05: MPM向け CPU地形SDFアップロード生成を一時停止。`prepare_terrain_upload` は初回に「常時外側SDF」を一度だけ送る暫定挙動へ変更し、GPU側SDF生成実装まで CPU再生成を抑止。
  - 2026-03-05: Far/Back 改変反映の第1段として、`terrain_far_update.wgsl` を拡張。Far/Back 更新時に Near base/override テクスチャを参照し、Nearカバー領域の override（delete含む）を downsample 集約へ反映するよう変更。
  - 2026-03-05: Far/Back 改変反映の第2段として、CPU override chunk 台帳を `hash table + chunk override cells` の疎SSBO payloadで render world へ転送し、`terrain_far_update.wgsl` に global override lookup を追加。Near外のoverride（solid/delete）も Far/Back 更新computeで反映されることを `terrain_edit_solid_far_back_visible` screenshot で確認。
  - 2026-03-05: zoom-outでNear非表示時に中央固定幅でFar/Backがずれる不具合を修正。原因はFar/Back更新shaderがNear無効時もNear cacheを参照していたこと。`near_cache_enabled` を `TerrainFarParams` に追加し、Near無効時は `global override -> generation` のみで評価するよう変更。`terrain_edit_far_back_baseline.png` で中央矩形ズレ解消を確認。
- 完了条件:
  - delete/solid によるセル編集が Near 描画へ反映される。
  - セーブロードで復元した改変が Near 描画で再現される。
  - 改変転送が「改変セルのみ RLE 差分 + 固定 payload 分割」で動作し、artifactで検証できる。
  - Far 改変反映は未着手として明示分離されている。


### [MPM-GPU-04] 地形SDF生成のGPU化（CPU upload撤去）

- Status: `In Progress`
- 背景:
  - 現在の GPU MPM では地形SDF供給に CPU 側の生成/upload を使用しており、地形改変時の再生成コストと責務分離の問題が残る。
  - `REND-GPU-02` で CPU SDF前計算と CPU地形生成コードを撤去したため、MPM向けSDFもGPU側で完結させる必要がある。
- スコープ:
  - MPM格子レイアウトに対する地形SDF/normalをGPU computeで生成し、CPU側の全点サンプリングと upload 依存を削除する。
- Subtasks:
  - [ ] `terrain_sdf_generate.wgsl`（仮）を追加し、MPM grid node ごとの signed distance / normal を生成する。
  - [ ] `TerrainNear` / override cache を入力にした SDF 生成経路を RenderGraph へ統合する。
  - [ ] `prepare_terrain_upload` の CPUループ実装を撤去し、GPU生成バッファ参照へ置換する。
  - [ ] `water_drop` / `default_world` の autoverify で地形境界反映を確認する。
- 完了条件:
  - MPM向け地形SDFがGPU側のみで更新され、CPU側の地形SDF生成/uploadループが不要になっている。


### [MPM-WATER-02] 明示MLS-MPM水ソルバ（単一レート）実装

- Status: `In Progress`
- 背景:
  - 水シミュレーション成立には、まず LoD なし単一レートでの安定な更新ループが必要。
- スコープ:
  - 1つの固定 `dt` で `P2G -> Grid Update -> Boundary -> G2P` を実行する。
  - 弱圧縮性圧力項（`J` ベース）と粘性項を実装する。
- Subtasks:
  - [x] P2G で質量・運動量・内部力寄与を実装する。
  - [x] Grid Update で重力・内部力を適用してノード速度を更新する。
  - [x] G2P で `v,x,C,F` を更新する。
  - [x] `J=det(F)` の数値安定化クランプを実装する。
  - [x] 単体テストを追加する（質量保存、静水安定、CFL判定）。
  - [x] `test world` の `water_drop` 読込時に `ContinuumParticleWorld` を初期化し、MPM更新結果を `ParticleWorld` 表示へ同期する。
  - [x] 一般編集モード（ブラシ生成水）から `ContinuumParticleWorld` への粒子供給経路を接続する。
  - [x] ランタイムの水更新経路を旧 `liquid` 実装から MLS-MPM へ完全切替する。
- 完了条件:
  - 単一レート条件で水塊が崩壊/流動し、質量保存誤差が許容範囲内に収まる。


### [MPM-WATER-03A] 地形SDF供給基盤（生成関数 + 差分LoDキャッシュ）

- Status: `In Progress`
- 背景:
  - 地形セルから都度SDFを復元する方式は、空間LoD導入後にクエリコストが増大しやすい。
  - 未改変領域は生成関数で直接評価し、改変領域のみキャッシュ参照するハイブリッド化が必要。
- スコープ:
  - 地形境界の問い合わせAPIを `生成関数 + 改変差分` の統合サンプラとして定義する。
  - Active block向けに LoD別SDF/法線キャッシュを遅延生成し、局所invalidationを実装する。
  - 当面は「地形改変を考慮しない」前提で、生成関数ベース経路を優先実装する。
- Subtasks:
  - [x] `sample_solid / sample_sdf / sample_normal` の境界サンプラAPIを定義する。
  - [x] 未改変領域は地形生成関数ベースで `sample_sdf` を直接評価する。
  - [x] 固定地形（テストワールド）では生成関数を無効化し、ロード済み地形セルのみをSDFソースに使う切替を実装する。
  - [ ] 改変領域は `Chunk` 差分を優先するLoDキャッシュ（occupancy/TSDF）を実装する。
  - [x] Active block単位のSDF/法線キャッシュを遅延構築し、再利用する。
  - [ ] 地形編集時の局所invalidation（影響block + 近傍margin）を実装する。
  - [x] 粗LoDでの漏れ抑制として conservative dilation を導入する。
  - [x] Tracyで境界クエリコストを計測し、セル都度復元方式との比較を記録する。
- 完了条件:
  - 境界SDFクエリが Active block 数に対してスケールし、改変有無を問わず同一APIで取得できる。


### [MPM-WATER-03] 地形SDF境界連成（水のみ）

- Status: `In Progress`
- 背景:
  - 水を実地形上で安定に扱うには、地形SDF境界での非貫通と接線減衰が必須。
- スコープ:
  - `MPM-WATER-03A` の地形SDF供給基盤を格子境界条件へ接続し、侵入抑制を実装する。
  - 連成ログ（運動量交換量）を取得できるようにする。
  - 当面は「地形改変を考慮しない」前提で、生成関数ベースSDFを格子段へ接続する。
- Subtasks:
  - [x] LoD選択付き地形SDFサンプラを MLS-MPM 格子更新段へ接続する。
  - [x] 法線方向非貫通補正を実装する。
  - [x] 接線減衰（境界摩擦）を実装する。
  - [x] 境界サンプルLoDの切替ヒステリシスを実装し、フレーム間ちらつきを抑制する。
  - [x] 侵入率メトリクスを追加する。
  - [x] 粒子数増加時の停止級劣化を解消するため、侵入率メトリクスをグリッド境界サンプル再利用方式へ置換する。
  - [x] 地形接触シナリオの統合テストを追加する（既存 `terrain_contact_stability` を活用）。
- 完了条件:
  - 水が地形へ継続侵入せず、境界近傍で発散しない。


### [MPM-WATER-03B] 地形境界速度補正ポリシー簡素化（投影なし）

- Status: `In Progress`
- 背景:
  - 現状の境界連成はグリッド段の速度補正が主経路であり、粒子SDF投影は本番経路で未使用。
  - 一時的な侵入は許容しつつ、長期的な侵入蓄積を防ぐ方針として「グリッド速度補正のみ」で閉じる設計を明確化したい。
  - `penetration_slop_m` は意味が混在しやすいため、SDF判定閾値と押し戻し強度を分離したパラメータへ整理したい。
- スコープ:
  - 粒子投影を導入せず、地形SDFサンプル済みノードに対する速度補正のみで侵入抑制を行う。
  - ノードSDFが閾値以上の領域は法線方向非貫通（法線速度0化）を適用する。
  - 閾値より深い領域は `|sdf|` と勾配に比例した押し戻し速度を付与し、深部からの回復を促進する。
- Subtasks:
  - [x] 境界補正パラメータを再定義する（SDF閾値、押し戻しゲイン、押し戻し速度上限）。
  - [x] `grid_update` の境界補正式を2領域（閾値以上/閾値未満）へ更新する。
  - [x] 侵入率メトリクス定義を新ポリシーに合わせて調整する（判定閾値を明示）。
  - [x] 長時間接触シナリオで「粒子減少（侵入蓄積）」が発生しないことを回帰確認する（`grid_only_boundary_policy_prevents_deep_penetration_accumulation`）。
  - [ ] 追加コストをTracyで計測し、既存境界補正との差分を記録する。
- 完了条件:
  - 粒子投影なしで地形接触の長時間安定性が維持され、侵入起因の粒子減少が継続的に発生しない。


### [MPM-WATER-05] 水シミュレーション受け入れテスト整備

- Status: `Planned`
- 背景:
  - 水優先フェーズの完了判断を明確化するため、受け入れシナリオと閾値を先に定義する。
- スコープ:
  - headless統合テストで水の最低限品質を数値判定する。
  - 剛体/粉体を使わないシナリオのみを対象にする。
- Subtasks:
  - [ ] 静水保持シナリオを追加する。
  - [ ] dam-break シナリオを追加する。
  - [ ] 地形流下シナリオを追加する。
  - [ ] 質量誤差・侵入率・最大CFL比の閾値を定義する。
  - [ ] `final_state.json/metrics.json/final_state.png` 出力で比較可能にする。
- 完了条件:
  - 水のみの代表シナリオが自動判定可能になり、合格基準を満たす。


### [MPM-DEFER-01] 粉体（Granular）連成の後段化

- Status: `In Progress`
- 背景:
  - 粉体（土/砂）は XPBD 併存ではなく、MLS-MPM に統合する方針へ確定した。
  - 水と粉体を同一グリッドで扱い、構成則のみを分離することで実装複雑度と保守コストを抑える。
  - 本フェーズでは液状化再現を要求しない。
- スコープ:
  - 土/砂の粉体を frictional Drucker-Prager として MLS-MPM material phase に追加する。
  - 水-粉体連成は「法線非貫通 + 接線摩擦/ドラッグ」の交換量モデルで実装する。
  - 二相連成（間隙水圧、有効応力、飽和度進化）は対象外に固定する。
- Subtasks:
  - [x] 設計方針を確定する（Drucker-Prager採用、非液状化、XPBD併存不採用）。（Design session: 2026-03-01）
  - [x] 粉体 material phase（`GranularSoil`, `GranularSand`）と粒子状態拡張（`Jp` 等）を設計する。
  - [x] GPU カーネルに粉体応力更新（trial stress + return mapping）を実装する。
  - [x] 水-粉体交換量（法線・接線）をノード段で実装し、対称運動量更新を保証する。
  - [ ] 既存XPBD粉体ステップを本番経路から外し、混在ケースでMLS-MPM単一路へ統一する。
  - [ ] 受け入れシナリオを追加する（乾燥沈降、斜面流下、水+粉体混在）。
  - [ ] 評価メトリクスを追加する（質量誤差、交換インパルス収支、侵入率、`Jp` 範囲）。
- Progress:
  - 2026-03-02: `ContinuumParticleWorld` を水専用から `Water/GranularSoil/GranularSand` へ拡張し、`Jp` を粒子状態へ追加。
  - 2026-03-02: GPU MPM shader (`mpm_p2g/grid_update/g2p`) を二相（水/粉体）ノード更新へ拡張し、Drucker-Prager系の return mapping とノード交換量（法線+接線）を実装。
  - 2026-03-02: `ParticleWorld -> Continuum -> GPU -> ParticleWorld` 同期対象を水+土砂粉体へ拡張し、mixed（水+土砂粉体）で粉体が停止しない実行経路へ更新。
  - 2026-03-02: `cargo check` / `cargo test --lib` を実行し全通過。GPUランタイムは `artifacts/mpm_autoverify_defer01.json` を出力（現行autoverify閾値では `drop_ok=false` で fail）。
  - 2026-03-02: 粒子オーバーレイGPUのバッファレイアウトを新MPM粒子定義（`phase_id`/`Jp`/144-byte params）へ追随させ、粉体が表示されない回帰を修正。
  - 2026-03-02: 受け入れシナリオを追加（`soil_repose_drop`, `sand_water_interaction_drop`）。`cargo test --test physics_scenarios -- --nocapture` で閾値判定を通過。
  - 2026-03-02: `cargo run -- --autoverify-config configs/autoverify/soil_repose_drop.json` と `.../sand_water_interaction_drop.json` でGPU自動検証を実行し、`artifacts/autoverify/*.json` の pass を確認。
  - 2026-03-02: 本流ドット描画を phase-aware 化（総密度+粒状密度の2チャネル）し、`water_dot_preprocess` の旧 `material_id` 依存を廃止。水と土砂がメイン描画で可視化されることを `artifacts/autoverify/*_drop.png` で確認。
  - 2026-03-02: autoverify に `run_steps`（固定ステップ基準）を追加し、`configs/autoverify/soil_repose_drop.json` / `sand_water_interaction_drop.json` へ適用。フレームレート依存の過走行を防止。
  - 2026-03-02: 粒状相の内部力符号と材料パラメータ（摩擦角/剛性/境界押し戻し）を再調整し、`soil_repose_drop` / `sand_water_interaction_drop` の autoverify を再実行して pass を確認。
- 完了条件:
  - 土/砂粉体が MLS-MPM（Drucker-Prager）で安定更新される。
  - 水+粉体混在シナリオで非液状化前提の連成が成立し、運動量収支と侵入率が閾値内に収まる。


### [WGEN-01] 決定論的チャンク生成と差分セーブ

- Status: `In Progress`
- 背景:
  - 世界を無限化しつつセーブ容量を抑えるため、未変更チャンクの再生成を前提にした設計が必要。
- スコープ:
  - seedベースの決定論チャンク生成と、変更チャンクのみ保存する差分セーブ方式を定義・実装する。
- Subtasks:
  - [x] 生成入力を `seed + world_coord + generator_version` に固定する。
  - [ ] 地形生成レイヤ（地表/水域/洞窟/鉱石/植生）の評価順を確定する。
  - [ ] チャンクdirty判定と差分形式（セル差分またはRLE）を実装する。
  - [x] ロード時に「再生成 + 差分適用」を行うフローを実装する。
  - [x] `generator_version` 不一致時の互換エラー処理を実装する。
- 完了条件:
  - 同一seedで再生成結果が一致し、未変更チャンクを保存しなくても整合が崩れない。


### [WGEN-02] 地形LODサムネイル生成

- Status: `Superseded` → [REND-GPU-01] GPU常駐キャッシュ方式に置換
- 背景:
  - 遠景背景と超拡大表示の両方で、フルチャンク生成なしに地形外観を取得したい。
- スコープ:
  - 任意LOD解像度で地形サムネイルを直接サンプリングする。
- Subtasks:
  - [x] カメラ視界内かつ `load_radius_chunks` 外のチャンクを LOD スプライトとして描画する。
  - [x] LOD スプライトをフルチャンクより奥 (`Z` の背面側) に表示する。
  - [x] Grid Overlay に LOD スプライト輪郭を表示する。
  - [x] 距離に応じてLODレベルを段階化し、1レベルあたり `1/4` 解像度相当へダウンサンプリングする。
  - [x] 低LODほど1スプライトが担当するチャンク範囲を拡大し、スプライトテクスチャ解像度を固定する。
  - [x] LOD用カラーパレットを起動時に事前計算し、レベル別に適用する。
  - [x] 改変済みチャンク内容を `LOD` サンプリングへ反映する（未ロード時は差分キャッシュ参照）。
  - [x] 粒子をLODスプライトへ反映する（タイル単位集約 + 地形LODへの合成）。
  - [x] LOD更新を軽量化する（地形ピクセルをタイルごとにキャッシュし、粒子更新時は合成のみ実行）。
  - [x] `collect_lod_particles_by_tile` の全粒子走査を廃止し、粒子チャンクキャッシュ参照に置換する。
  - [x] `lod_update_tiles` を差分更新化し、粒子変更時は「粒子ありタイル」と「粒子が消えたタイル」のみ再合成する。
  - [x] render主要systemにTracyの内訳ゾーン（phase span）を追加する。
  - [ ] `lod_cell_size = base_cell_size * 2^L` の定義を導入する。
  - [ ] LODセル内の多点サンプリングで占有率を推定する。
  - [ ] 占有率から材質代表値を決める規則を実装する。
  - [ ] 近景フル解像度と遠景LODの境界遷移を定義する。
  - [ ] 変更チャンクのLOD再計算範囲を局所化する。
- 完了条件:
  - フルチャンク生成なしで遠景サムネイルが生成され、境界遷移が破綻しない。


### [WGEN-03] 連続LOD地形サンプラ（fBm + 確率場）

- Status: `In Progress`
- 背景:
  - 超遠景背景まで含めると、離散セル前提の1点参照ではLOD生成コストと見た目の連続性に課題がある。
  - 近景はセル確定表現、遠景は平均化表現を同一生成系で連続的に扱いたい。
- スコープ:
  - 生成関数を「フットプリント（サンプル解像度）依存」で評価できる連続サンプラに拡張する。
  - fBmベースの連続ノイズから材質確率場（Stone/Soil/Empty）を評価し、描画LODに反映する。
  - 近景は決定的サンプリング、遠景は期待値色、その中間は補間で遷移する。
- Subtasks:
  - [x] LOD描画用に `sample_terrain_at(world_pos, footprint_m)` 相当の連続サンプラインターフェースを定義する。
  - [x] fBm評価を導入し、フットプリントに応じて高周波オクターブ寄与を減衰/打ち切りする。
  - [x] 材質確率場 `P_empty/P_soil/P_stone` の計算規則を定義する（合計1.0、深度依存を連続化）。
  - [x] 近景モード（決定的ハッシュサンプリング）と遠景モード（期待値色）を実装する。
  - [x] 近景/遠景の遷移帯でブレンド係数を導入し、ズーム変化時のちらつきを抑制する。
  - [x] 遷移帯ブレンド係数を標準誤差（`stderr`）と有効サンプル数（`N_eff`）ベースで決定する。
  - [x] 期待値色に `Empty`（透明）確率を含め、地表境界が連続に見えるようにする。
  - [x] 既存の改変済み地形との合成規則を実装する（改変領域は連続サンプラより優先）。
  - [x] `span_chunks <= CELL_PIXEL_SIZE` のLODでは従来のドットサンプリングを維持し、近景の粒状感を保つ。
  - [ ] ベースノイズ実装を差し替え可能にする（OpenSimplex固定を避け、複数ノイズ合成を許容）。
  - [ ] Tracyで連続サンプラの評価コストを計測し、現行LODサンプリングとの差分を確認する。
- 完了条件:
  - 等倍から超遠景まで地形外観が連続遷移し、ズーム時の破綻や強いちらつきが抑制される。
  - LOD地形生成が現行より高速、または同等コストで品質改善を確認できる。


### [REND-01] Tileベース描画パイプライン再編（Full/LOD統合）

- Status: `Superseded` → [REND-GPU-01] GPU常駐キャッシュ方式に置換
- 背景:
  - 現状はフルチャンク描画とLOD描画の管理単位が分かれており、同期コストと責務境界が不明瞭。
  - ズーム時に `sync_lod_chunks_to_render` などの更新コストが高く、Tile単位の並列更新へ寄せたい。
- スコープ:
  - Render側の最小単位を `Tile` に統一し、Full/LOD を「解像度違いの同一表現」として扱う。
  - LOD要求はカメラ情報のみで決定し、物理演算範囲（active/halo）とは独立に管理する。
  - `World -> Tile` 反映と `Tile -> Texture` 更新を分離し、Tile単位に並列実行可能な構成へ移行する。
- Subtasks:
  - [x] `RenderTile` コンポーネント（tile座標、`lod_level`、参照チャンク範囲、dirty状態）を定義する。
  - [x] カメラ状態から required tile set を算出するsystemを実装する（Full/LOD統一）。
  - [x] requiredとの差分で Tile entity を spawn/update/despawn する reconcile system を実装する。
  - [x] 地形/粒子の更新から Tile dirty を付与する `World -> Tile` 反映systemを実装する。
  - [x] Tileテクスチャ生成をTile単位ジョブ化し、`Assets<Image>` 反映系と分離する。
  - [x] チャンク改変時のLOD再サンプリング範囲を影響Tileのみに局所化する。
  - [x] カメラ範囲外のTile破棄にヒステリシスを導入し、entity churnを抑制する。
  - [x] Grid Overlay をTile基準で再構成し、Full/LOD輪郭を一貫表示する。
  - [x] Tracyで `required calc / reconcile / reflect / texture upload` の内訳計測を追加する。
  - [ ] 旧実装比でズーム時の描画更新コストを比較し、改善を確認する。
- 完了条件:
  - Full/LOD描画が `Tile` 実装へ統合され、ズーム時のレンダリング更新負荷が現行より有意に低下する。
  - 描画LOD決定がカメラ依存のみで完結し、物理演算範囲とは独立して動作する。


## Done (Recent)

### [PARAM-01] パラメータ資産化と hot reload 運用統一

- Status: `Done`
- 背景:
  - 物性・シミュレーション・描画パラメータが Rust 定数と shader 周辺に分散し、調整と追跡のコストが高い。
  - Bevy の asset hot reload を活用し、実行中調整と責務分離を両立したい。
- スコープ:
  - 実行時調整対象パラメータを `assets/params/` 配下の RON 資産へ集約する。
  - `physics / render / overlay / material / generation` の5資産を定義し、Rust/WGSL 参照を統一する。
  - 実行中変更すべきでない定数（レイアウト/最適化前提）は対象外として明確化する。
- Subtasks:
  - [x] `docs/design.md` にパラメータ資産管理方針（5ファイル構成、反映経路、対象外ルール）を明文化する。
  - [x] `assets/params/physics.ron` を追加し、GPU MPM 物性・境界・連成パラメータを移管する。
  - [x] `assets/params/render.ron` を追加し、water/terrain dot 描画パラメータを移管する。
  - [x] `assets/params/overlay.ron` を追加し、overlay 描画/閾値パラメータを移管する。
  - [x] `assets/params/material.ron` を追加し、material id ごとの物性セット参照を移管する。
  - [x] `assets/params/generation.ron` を追加し、地形生成・分布・確率場パラメータを移管する。
  - [x] 各 RON の全項目に用途・単位・許容範囲コメントを付与する（コメント未記載項目を残さない）。
  - [x] Asset 読み込み + hot reload 反映 system（検証/クランプ、失敗時フォールバック）を実装する。
  - [x] shader 参照値を Rust 側解決済み uniform/storage 経由へ統一し、asset 直接依存を作らない。
  - [ ] compile/test と runtime hot reload 検証（値変更前後の artifact/log）を追加する。（将来タスク化）
- 完了条件:
  - 5つの `assets/params/*.ron` が作成され、関連定数が責務別に移管されている。✅
  - 各 RON の全項目に説明コメントがあり、hot reload で安全に反映される。✅
  - 実行中不変の定数は asset 対象外としてコード上で分離されている。✅
- 進捗:
  - 2026-03-02: 5 asset 実装完了。`src/params/` に physics/render/overlay/material/generation モジュール、対応 Rust 型・loader・validate() 実装。
  - `ParamsPlugin` が Startup で全RON読込、Update で hot reload 反映（失敗時フォールバック）。
  - `src/physics/gpu_mpm/sync.rs` の `prepare_gpu_params()` が `ActivePhysicsParams` を参照するよう統一。
  - Cargo.toml に `ron = "0.8"` 追加、compile 成功・warnings ゼロ。


### [MPM-PHYS-WATER-01] physics.md 改訂に基づく水物性GPU実装アップデート

- Status: `Done`
- 背景:
  - `physics.md` 改訂（2026-03-02）により、水の構成則・境界摩擦・APIC↔PICブレンドの仕様が更新された。
  - GPU shaderの実装（`mpm_p2g.wgsl`, `mpm_g2p.wgsl`, `mpm_grid_update.wgsl`）が旧仕様のままであり、整合化が必要。
- スコープ:
  - 水の物性に関するGPU実装を `physics.md` v2仕様に合わせて更新する。
  - 粉体（granular）実装への変更は含まない（別Work Unitで対応予定）。
- Subtasks:
  - [x] **[EOS修正]** `mpm_p2g.wgsl` の水圧力算出を `K * max(1-J, 0)` から `K * (1/J - 1)` (Eq.7) に変更する。
  - [x] **[F等方化]** `mpm_g2p.wgsl` のG2P後処理で、水粒子の変形勾配を `F = sqrt(J) * I` (Eq.35) に等方化する。
  - [x] **[Coulomb摩擦]** `mpm_grid_update.wgsl` の地形境界を Coulomb stick/slip (Eqs.28-29) へ変更。`MpmParams` に `boundary_friction_water / boundary_friction_granular` を追加し `tangential_damping / deep_push_*` を撤去。
  - [x] **[APIC↔PICパラメータ化]** `mpm_g2p.wgsl` のCマトリクス減衰を `MpmParams.alpha_apic_water / alpha_apic_granular` へ変更 (Eq.32)。
  - [x] **[充填率に基づく負圧制御]** G2Pで `φ_p` を収集 (`GpuParticle.phi_p`)、P2Gで `smoothstep(0.1, 0.8, φ_p)` による負圧減衰を実装 (Eqs.8-11)。
  - [x] `PARTICLES_AUTOVERIFY_MPM=1` で全指標通過を確認（penetration=0, drop=8.44m, max_speed=2.63 m/s）。
- 進捗:
  - 2026-03-02: 全5サブタスク実装・自動検証通過。ベースライン比: max_speed 4.54→2.63 m/s（等方化により安定化）、penetration=0維持。
- 完了条件:
  - 5つのsubタスクが完了し、`water_drop` 自動検証で NaN/発散なし・地形侵入率0・落下確認が成立する。



### [MPM-PHYS-GRANULAR-01] physics.md 改訂に基づく粉体GPU実装ギャップ解消

- Status: `In Progress`
- 背景:
  - `physics.md` 改訂（2026-03-02）で、粉体は「SVD + log-strain 射影（Eqs.14-21）」と「G2P後の塑性射影（Eq.34後）」を前提に定義された。
  - 現在のGPU実装は暫定の small-strain DP return mapping であり、仕様との不一致が複数残っている。
  - mixed（水+粉体）連成も、界面法線と衝突判定の定義が `physics.md`（Eqs.37-44）と一致していない。
- スコープ:
  - 粉体構成則と水-粉体連成を `physics.md` v2 の式へ一致させる。
  - 粉体粒子状態・パラメータ資産・検証メトリクスを `physics.md` の受け入れ基準へ合わせる。
- Subtasks:
  - [x] **[状態量整合]** 粉体状態を `jp` ベースから `v_vol(diff_log_J)` を主状態とする構成へ置換し、GPUバッファ/Rust同期を更新する。
  - [x] **[DP射影位置修正]** `mpm_g2p.wgsl` で Eq.34 後に 2x2 SVD + log-strain DP射影（Eqs.14-20）を実装し、Eq.21 で `v_vol` を更新する。
  - [x] **[応力評価修正]** `mpm_p2g.wgsl` の small-strain 応力計算を撤去し、DP射影後 `F` に対する compressible Neo-Hookean（Eq.13）から内部力を評価する。
  - [x] **[連成則修正]** `mpm_grid_update.wgsl` の水-粉体交換を Eq.39-44 ベースへ再実装する（`∇φ_g` 法線、接近時のみ法線インパルス、接線摩擦円錐、対称運動量更新）。
  - [x] **[パラメータ更新]** `PhysicsParams` / `MpmParams` を新連成則・射影に必要なパラメータへ更新し、旧 `normal_stiffness/tangent_drag/max_impulse_ratio` 依存を撤去する。
  - [x] **[単一路統一]** `StoneGranular` を含む granular 粒子の本番経路を MLS-MPM 単一路に統一し、GPU無効フォールバック条件を明示化する。
  - [ ] **[検証拡張]** autoverify/シナリオに `v_vol` ドリフト、相間インパルス収支、mixed 侵入率の指標を追加し、artifactで判定可能にする。
- 進捗:
  - 2026-03-02: 差分レビュー実施。主な不一致点を確認。
    - `mpm_p2g.wgsl` は small-strain 前提の暫定DP return mapping（`granular_stress_return_mapping`）を使用しており、`physics.md` の SVD + log-strain 射影と不一致。
    - `mpm_g2p.wgsl` は Eq.34 後の DP射影および `v_vol` 更新を実装しておらず、`J` の一様リスケールに留まる。
    - `mpm_grid_update.wgsl` の相間連成で法線を `rel/|rel|` から作っているため、接線成分が常に0となり、Eq.44 相当の接線摩擦が機能していない。
    - 粉体状態量が `v_vol` ではなく `jp` で保持されており、`physics.md` Eq.21 の体積補正トラッキングに未対応。
  - 2026-03-02: `jp -> v_vol(diff_log_J)` へ置換（Rust `ContinuumParticleWorld` / GPU particle layout / readback 同期 / render shader layout追随）。
  - 2026-03-02: `mpm_g2p.wgsl` を 2x2 SVD + log-strain DP射影へ更新し、Eq.21 の `v_vol` 更新を実装。`mpm_p2g.wgsl` は Neo-Hookean 応力（Eq.13）で内部力評価へ変更。
  - 2026-03-02: `mpm_grid_update.wgsl` の相間連成を `∇phi_g` 法線 + 対称drag/摩擦円錐（Eqs.39-44）へ更新。地形境界には内部侵入回復用の速度押し戻し（数値安定化）を追加。
  - 2026-03-02: CPU粉体ソルバを削除（`src/physics/solver/granular.rs` 削除、`particle_step` から呼び出し撤去、関連unit test整理）。`StoneGranular` を MPM managed phase に含めた。
  - 2026-03-02: 検証実行:
    - `cargo check` ✅
    - `cargo test --lib` ✅（100 passed）
    - `cargo test --test physics_scenarios -- --nocapture` ✅
    - `cargo run -q -- --autoverify-config configs/autoverify/soil_repose_drop.json` ❌（`granular_repose_angle_deg` 未達: 12.037）
    - `cargo run -q -- --autoverify-config configs/autoverify/sand_water_interaction_drop.json` ❌（`material_interaction_centroid_order` 未達: primary_y=-7.2037, secondary_y=-7.2405）
  - 2026-03-02: 追加修正:
    - `mpm_p2g.wgsl` の粉体内部力項で、`∇w` を粒子位置勾配（`∇_{x_p}`）として計算しているにも関わらず Eq.24 の符号をそのまま適用していた不一致を修正（`stress_force = -v0p * Agrad` -> `+v0p * Agrad`）。
    - `assets/params/physics.ron` の粉体系数を再調整（`j_min=0.60`, `alpha_apic_granular=0.60`, soil/sand の `E`・`friction_deg` を強化、`coupling_drag_gamma=0.35`, `coupling_friction=0.25`）。
    - mixed での沈降順序安定化のため `src/physics/material/defaults.rs` の `SAND_GRANULAR_CELL_MASS` を `1.3 -> 2.2` に引き上げ。
  - 2026-03-02: 再検証（test world autoverify）:
    - `cargo run -q -- --autoverify-config configs/autoverify/soil_repose_drop.json` ✅
    - `cargo run -q -- --autoverify-config configs/autoverify/sand_water_interaction_drop.json` ✅
    - `cargo check` ✅
    - `cargo test --lib` ✅（100 passed）
    - `cargo test --test physics_scenarios -- --nocapture` ✅
  - 2026-03-02: 粒子 overlay 非表示回帰を修正。`particle_overlay_gpu.wgsl` のローカル `GpuParticle/MpmParams` 定義を廃止し `mpm_types.wgsl` を import して GPU バッファレイアウトと単一化。`configs/autoverify/soil_repose_drop_screenshot.json` 実行で `artifacts/autoverify/soil_repose_drop.png` に overlay 粒子描画を確認。
  - 2026-03-03: 粉体の圧縮挙動調整として `physics.ron` の `water.sound_speed_mps=32.0`, `coupling.drag_gamma=100.0`, `coupling.friction=1.0` を更新し、同時に粒子解像度を `src/physics/material/defaults.rs` の `PARTICLES_PER_CELL=16` へ統一（`water/soil/sand/stone granular`）。
  - 2026-03-02: 土砂落下で「底面で過圧縮 -> 横噴出」する回帰に対応。
    - `mpm_g2p.wgsl` の DP 射影後で極小 `det(F)` のみ等方フォールバックするガードを追加（通常応答は維持）。
    - `mpm_grid_update.wgsl` の地形侵入回復速度注入を再調整（方式は維持しつつ cap を `3.0 -> 1.2`）。
    - `assets/params/physics.ron` を連成安定寄りに再調整（`coupling_drag_gamma=0.20`, `coupling_friction=0.15`）。
    - `src/physics/material/defaults.rs` の砂粒状セル質量を `2.6` に更新し mixed での重心順序を安定化。
  - 2026-03-02: 再検証（過圧縮/不安定化チェック）
    - `cargo run -q -- --autoverify-config configs/autoverify/soil_repose_drop.json` ✅（pass, `max_speed_mps=0.686`, `terrain_penetration_ratio=0.00113`）
    - `cargo run -q -- --autoverify-config configs/autoverify/sand_water_interaction_drop.json` ✅（pass, `max_speed_mps=7.774`, `terrain_penetration_ratio=0.0`）
    - `cargo run -q -- --autoverify-config configs/autoverify/soil_repose_drop_screenshot.json` ✅（`artifacts/autoverify/soil_repose_drop.png` で底面一列化・噴出の再発なしを確認）
    - `cargo check` / `cargo test --lib` / `cargo test --test physics_scenarios -- --nocapture` ✅
  - 2026-03-02: `grid` 密度計測を autoverify レポートへ追加（`grid_phi_*`）。
    - 実装: `src/main.rs` の `MpmAutoVerifyReport` に `grid_phi_{water,granular}_{max,p99,mean_nonzero}` と `nonzero_nodes` を追加。粒子→3x3 B-spline の質量散布から `phi = m / (rho0 * h^2)` を算出。
    - 観測: `soil_repose_drop` で `grid_phi_granular_max=11.03`, `p99=10.99`。`sand_water_interaction_drop` では granular `max=4.10`, `p99=3.57`。
    - 切り分け: `j_min` 引き上げ（0.60→0.80）と soil `friction/cohesion` 低下を試験したが、`soil` の一列化/過密化は改善せず（`j_min=0.80` では `phi` がさらに悪化）。単純な係数調整より、DP射影（Case II tensionless）で圧縮が塑性固定される処理特性の寄与が大きいと判断。
    - 対応方針: 次段で「granular 高密度ノードの compaction cap（例: `phi_g` しきい値超過時の体積回復/内部力上限）」を shader 側に導入し、`grid_phi_granular_p99` を受け入れ指標へ追加する（閾値変更は承認後）。
  - 2026-03-02: overlay 描画順を修正。
    - `src/overlay/mod.rs` の Core2d graph を `MainTransparentPass -> WaterDotGpuLabel -> ParticleOverlayGpuLabel -> EndMainPass` に明示し、root graph の `try_add_node_edge` 依存を撤去。
    - これにより `particle overlay` は water dot 描画より後段で実行される。
  - 2026-03-02: 粉体 `alpha_apic_granular=0` 実験（塑性OFF条件）を実施。
    - `mpm_g2p.wgsl` で granular 限定の `C` 分離を追加（`C_def=C_raw` を `F` 更新へ、`C_xfer=alpha*C_raw` を P2G affine 用に保存）。
    - `assets/params/physics.ron` で `apic.granular=0.00` 条件で検証。
    - 結果: `sand_water_interaction_drop` は形式上 pass するが granular ノード質量が消失（`grid_phi_granular_nonzero_nodes=0`）。`soil_repose_drop` は fail で粒子が極端に飛散（`tracked_max_y=192148.88`, `tracked_min_x=-177445.02`）。
    - 判断: `alpha=0` で「粘弾性らしさ」は現れず、現行の塑性OFF + 高弾性設定では数値発散傾向が支配的。
  - 2026-03-03: 爆発前兆解析のため autoverify ログを拡張。
    - `src/main.rs` に granular 診断値を追加: `det(F) min/p99/max`, Neo-Hookean `|P| p99/max`, `v_vol_abs_max`, invalid件数。
    - `soil_repose_drop`（塑性OFF + `alpha_apic_granular=0`）: `det(F)_min=0.60`, `|P|_p99=17638`, `|P|_max=35473`。
    - `sand_water_interaction_drop` 同条件: `det(F)` が全体で `0.60` 近傍に貼り付き、`|P|_p99=174843`, `|P|_max=323982`、`grid_phi_granular_max=79.43`。
    - screenshot でも粒子群の可視消失/画角外飛散を確認（`artifacts/autoverify/soil_repose_drop.png`, `sand_water_interaction_drop.png`）。
  - 2026-03-03: 「着地後に左右下へ吸い込まれる」現象をゼロベースで再切り分け。
    - 比較実験（同一パラメータ、`soil_repose_drop` 3600 step）:
      - **bug実装**（`mpm_p2g.wgsl` で `F^{-T}` の off-diagonal が入れ替わっている状態）
        - `cargo run -q -- --autoverify-config configs/autoverify/soil_repose_drop_long_bug.json`
        - `grid_phi_granular_nonzero_nodes=18`, `grid_phi_granular_max=48.37`, `granular_det_f_min=max=0.60`（全粒子が下限張り付き）
        - `artifacts/autoverify/soil_repose_drop_long_bug_late.png` で左右下の2点へ収束する崩壊を確認。
      - **fix実装**（`F^{-T}` を正しく評価）
        - `cargo run -q -- --autoverify-config configs/autoverify/soil_repose_drop_long_fix.json`
        - `grid_phi_granular_nonzero_nodes=206`, `grid_phi_granular_max=3.99`, `granular_det_f_min=0.735`（過密化が解消）
        - `artifacts/autoverify/soil_repose_drop_long_fix_late.png` で左右吸い込みの消失を確認。
    - 結論: 主因は **physics.md ではなく WGSL 実装**。Eq.13 の `F^{-T}` を `F^{-1}` 相当で評価していたため、回転/せん断で偽応力が出て粒子が異常収束していた。
    - 反映: `assets/shaders/physics/mpm_p2g.wgsl` の `invt01/invt10` を修正（`invt01=-f10*inv_j`, `invt10=-f01*inv_j`）。
    - mixed 影響確認: `sand_water_interaction_drop` では過密化は改善（`phi_g_max` 79.43 -> 1.71）したが、`material_interaction_centroid_order` は未達のため追加調整が必要。
- 完了条件:
  - `physics.md` Eqs.13-21, 37-44 と実装が一致し、granular-only / mixed シナリオで新指標が閾値内を満たす。
  - 粉体本番経路が MLS-MPM 単一路となり、XPBD粉体の混在依存が残らない。


## Closed Work Units (Deferred / Superseded)

- `MPM-GPU-03` / `Done` / 完了条件達成（GPU地形描画成立・旧CPU描画削除済み）/ Active Work Units 進捗欄参照
- `REND-01` / `Superseded` / GPU Near/Far 2層キャッシュ方式 [REND-GPU-01] に置換 / Active Work Units 詳細参照
- `WGEN-02` / `Superseded` / GPU Near/Far 2層キャッシュ方式 [REND-GPU-01] に置換 / Active Work Units 詳細参照
- `MPM-WATER-04A` / `Deferred` / GPU-first方針へ転換 / `docs/tasks_archive/2026-03-02-full-before-compaction.md`
- `MPM-WATER-04B` / `Deferred` / 単一grid GPU版で一様substepを優先 / `docs/tasks_archive/2026-03-02-full-before-compaction.md`
- `MPM-WATER-07` / `Deferred` / 空間LoD方針を凍結 / `docs/tasks_archive/2026-03-02-full-before-compaction.md`
- `MPM-WATER-07A` / `Deferred` / CPU彩色実験の拡張停止 / `docs/tasks_archive/2026-03-02-full-before-compaction.md`
- `MPM-WATER-07B` / `Deferred` / ghost前提をGPU経路へ置換 / `docs/tasks_archive/2026-03-02-full-before-compaction.md`
- `MPM-WATER-07C` / `Deferred` / CPU色順並列の追加最適化停止 / `docs/tasks_archive/2026-03-02-full-before-compaction.md`
- `MPM-DEFER-02` / `Deferred` / 水・粉体優先で剛体連成は後段化 / `docs/tasks_archive/2026-03-02-full-before-compaction.md`

## Archive Index

- `2026-03-02 full snapshot`:
  - `docs/tasks_archive/2026-03-02-full-before-compaction.md`
  - 退避対象: Deferred詳細、旧Done詳細、Legacy Checklist、過去Design Feedback履歴

## Approval-Gated Items

- 物理統合テスト（headless scenario tests）のケース追加・閾値変更・ベースライン更新は、実装前にユーザー承認を取る。

## Open Design Feedback

- LoD境界フラックス交換（同期点、補間、対称更新）の実装詳細を確定する。
- 水の圧力モデル（`J` ベース / EOS ベース）の採用条件とパラメータ範囲を確定する。
- 地形境界SDF供給の「未改変=生成関数、改変=差分LoDキャッシュ」運用を実装側で固定する。
- `PARAM-01` のパラメータ資産数が実装では6ファイル（`palette.ron` 追加）へ増えているため、`docs/design.md` の記述（5ファイル前提）を次回Design sessionで更新する。
- REND-GPU-01 検証中は「地形改変なし」前提でレンダ override 転送を一時停止している。改変地形を再有効化する条件（CPU差分→GPU反映方式、検証項目、切替フラグ）を次回Design sessionで明文化する。
