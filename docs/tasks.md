# Tasks

## Task Format

- 本ドキュメントは「実装単位（Work Unit）」ごとに管理する。
- 各Work Unitは以下を持つ。
  - 背景
  - スコープ
  - サブタスク（チェックリスト）
  - 完了条件
- `docs/design.md` は最新目標のみを持ち、変更の背景・移行説明はここに記載する。

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
- 完了条件:
  - `water_drop` がGPU経路で安定再現され、品質指標を自動計測できる。

### [MPM-GPU-02] Water Drop表示のGPU完結化

- Status: `Planned`
- 背景:
  - 計算だけGPU化しても、毎フレームreadback描画では同期コストが残る。
- スコープ:
  - 先にデバッグ用円オーバーレイを整備して物理結果を検証し、その後GPUドット絵描画へ移行する。
- Subtasks:
  - [ ] デバッグオーバーレイで粒子円表示を実装し、`water_drop` の挙動確認に使う。
  - [ ] GPU粒子/密度バッファからドット絵テクスチャを生成する描画パスを実装する。
  - [ ] フラグメントシェーダーを第一候補として実装し、必要ならcompute前処理を追加する。
  - [ ] 表示品質（連続性、エイリアシング、tile境界、ちらつき）を確認する。
- 完了条件:
  - `water_drop` で「計算 + 描画」がGPU完結で実行できる。

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

### [MPM-WATER-04A] block分割 + CPU並列基盤（空間幅均一）

- Status: `Deferred`
- 備考:
  - GPU-firstへの方針転換により新規開発は停止。置換完了後は関連CPUコードを削除対象とする。
- 背景:
  - block単位並列を導入するには、まず全block同一 `h_b` の構成で ownership とデータ参照規約を確立する必要がある。
  - 粒子SoAは単一Resourceを維持し、block側はindex tableで参照する方式を採用する。
- スコープ:
  - 粒子の `owner_block_id` と blockごとの `owner_indices/ghost_indices` を導入する。
  - `P2G/Grid Update/G2P` を block単位ジョブ並列で実行できる基盤を実装する。
- Subtasks:
  - [x] `owner_block_id` 更新と再binning（block跨ぎ時のみ）を実装する。
  - [x] block index table（`owner_indices/ghost_indices`）をResourceとして実装する。
  - [x] `P2G` を block並列化し、書き込み先ノードを自blockに限定する。
  - [x] `G2P` を block並列化し、`owner_indices` のみ更新する。
  - [x] `unsafe` 高速経路を導入する場合の debug assertion（owner一致・重複なし）を実装する。
  - [x] 回帰テストを追加する（決定論性、owner重複検知、並列/逐次一致）。
- 完了条件:
  - 空間幅均一条件で block単位CPU並列が安定稼働し、粒子書き込み競合が発生しない。

### [MPM-WATER-04B] 水向けsubcycling（時間LoD, 空間幅均一）

- Status: `Deferred`
- 備考:
  - 可変更新頻度は単一grid GPU版の一様substep成立後に再評価する。
- 背景:
  - 時間LoDは ADR の必須要件であり、まず空間幅均一の block構成上で導入する。
- スコープ:
  - block ごとの `dt_b` を CFL 条件で計算し、フレーム内 subcycling を実行する。
  - 対象blockスケジューリングと更新順序の決定論性を確保する。
- Subtasks:
  - [x] `dt_b = min(Cu*h/u, Cc*h/(c+u), Ca*sqrt(h/a))` を実装する。
  - [x] subcycle スケジューラ（frame内の更新タイムライン）を実装する。
  - [x] 対象blockセットを再利用し、非対象blockを毎回全走査しない経路を実装する。
  - [x] block 更新順序を決定論的に固定する。
  - [x] CFL違反検知と自動縮退（安全側 `dt`）を実装する。
  - [x] 回帰テストを追加する（同一入力で再現性確認）。
- 完了条件:
  - 空間幅均一条件で時間LoDが有効化され、CFL違反なしで安定動作する。

### [MPM-WATER-07] 空間LoD block拡張（可変 `h_b`）

- Status: `Deferred`
- 備考:
  - 単一grid回帰を優先するため、空間LoD block拡張は凍結する。
- 背景:
  - 広域最適化には時間LoDだけでなく、blockごとに空間幅を変える空間LoDが必要。
  - 粗密境界で保存量を壊さないため、均一幅フェーズとは別Work Unitで段階導入する。
- スコープ:
  - blockごとに異なる `h_b` を許可し、粗密境界のフラックス交換と同期点処理を実装する。
  - 粒子再サンプル（merge/split）は導入せず、まず固定粒子で成立させる。
- Subtasks:
  - [ ] level別 block 管理（`h_b`, `dt_b`）を実装する。
  - [ ] coarse-fine 境界の質量/運動量フラックス交換を実装する。
  - [x] 粗密境界での ghost参照半径と補間規約を実装する。
  - [x] `water_drop` シナリオを規定block size（default span）+ level 0 構成へ戻す。
  - [ ] 境界補間での保存誤差メトリクスを追加する。
  - [ ] 空間LoD回帰テストを追加する（境界通過、保存量、安定性）。
- Progress:
  - 2026-02-28: block定義を「16x16セル（17x17ノード）」へ切り替え、境界ノード共有を許可。
  - 2026-02-28: 共有境界ノードのowner規約を実装（fine level優先、同levelは`origin.x`, `origin.y`が小さいblock優先）。
  - 2026-02-28: P2G圧力/運動量転送で非owner境界ノードへの書き込みを抑止。
  - 2026-02-28: ノードlookupキーをbase-cell基準へ統一し、level間で同一world位置を同一キーとして扱うよう修正（level 1のgrid/particleずれ対策）。
  - 2026-02-28: 空間LoD検証向けにblock rate算出の空間幅依存を抑制（`CELL_SIZE_M`基準）し、level跨ぎ時の見かけ減速を緩和。
  - 2026-02-28: overlayでcoarse blockに補助グリッド線を追加し、level 0/1間の見た目連続性を改善。
  - 2026-02-28: ghost再構築を全block対象に変更し、ghost更新後のactive集合でP2G/Grid Updateを実行するよう修正（block 1 -> block 0 境界停止の抑制）。
  - 2026-02-28: coarse->fine境界でreceiver blockのghost更新漏れを検出する回帰テスト `coarse_to_fine_boundary_requires_refreshing_receiver_block_ghosts` を追加。
  - 2026-02-28: water dot描画のグリッド密度サンプルを `first_block.h_b` 固定から world位置ベースのblock選択へ変更し、tile/blockスケール差による投影ずれを抑制。
  - 2026-02-28: owner再bin時に「stencil欠損ノード数が最小となる隣接block」を選ぶ補正を追加し、粗密境界手前での速度失速を緩和。
  - 2026-02-28: ghost運動量転送の異解像度パスでAPIC affine項を有効化し、`p2g_pressure` もowner block `h_b` 基準で評価するよう整合。
  - 2026-02-28: 粗密境界通過時の極端な速度低下を検出する回帰テスト `coarse_fine_boundary_crossing_does_not_stall_without_forces` を追加。
  - 2026-02-28: `water_drop` の MPM block 構成を `mpm_block_divisions=None` に戻し、規定spanの level 0 blockのみを使う設定へ更新。
- 完了条件:
  - 可変 `h_b` の空間LoDで、粗密境界を跨ぐ流れでも保存量誤差が許容範囲に収まる。

### [MPM-WATER-07A] block彩色実験（辺/頂点共有の排他彩色）

- Status: `Deferred`
- 備考:
  - block彩色はCPU block並列向け検証として完了扱いとし、追加実験は停止する。
- 背景:
  - block単位並列（特に色分け方式）を検討するため、与えられた空間LoD block配置に対し、
    「辺または頂点を共有するblock同士が同色にならない」彩色アルゴリズムの挙動確認が必要。
- スコープ:
  - block配置を入力として、未彩色状態からの貪欲彩色を実装する。
  - block配置変更時（split/merge）に彩色をリセットして再計算する。
  - 専用 test world で1秒ごとのランダム split/merge と再彩色を繰り返し、overlayで色を可視化する。
- Subtasks:
  - [x] `GridHierarchy` に「辺/頂点共有を競合とする」貪欲彩色を実装する。
  - [x] blockごとの `color class` を保持し、レイアウト再構築時に再彩色する。
  - [x] `block_coloring_experiment` scenario を追加する（level 3 blockを8x8敷き詰め、地形なし）。
  - [x] scenario実行中に1秒ごとランダム split/merge を行う実験ランタイムを追加する。
  - [x] Physics Area Overlay の MPM grid を level色分けから color class 色分けへ切り替える。
  - [x] Physics Area Overlay の MPM block 中央に、各blockの時間level（rate level）表示を追加する。
  - [x] `GridHierarchy` の block管理を quadtree index（linear quadtree）化し、位置クエリ/近傍構築を最適化する。
  - [ ] 彩色結果の色数推移とレイアウト変化コスト（再構築時間）を計測し、並列戦略検討材料として記録する。
- Progress:
  - 2026-02-28: 貪欲彩色（辺/頂点共有の排他制約）を `GridHierarchy` に実装し、layout変更時の再彩色を有効化。
  - 2026-02-28: 専用scenario `block_coloring_experiment` を追加し、初期 level 3 block 16x16 配置を導入。
  - 2026-02-28: 実験反復の高速化のため、初期配置を level 3 block 8x8 へ縮小。
  - 2026-02-28: 1秒ごとのランダム split/merge（数個操作）+ 再彩色の実験ランタイムを実装。
  - 2026-02-28: Physics Area Overlay を color class ベースに変更し、MPM color count表示を追加。
  - 2026-02-28: Physics Area Overlay で MPM block ごとの時間level表示を追加。
  - 2026-02-28: `GridHierarchy` に quadtree index を導入し、`block_index_for_position` と block近傍構築を quadtree ベースに切替。
- 完了条件:
  - 専用scenarioで split/merge 後も辺/頂点共有blockの同色衝突が発生せず、overlayで色分け状態を連続確認できる。

### [MPM-WATER-07B] ghost廃止（resident/support + outgoing queue）

- Status: `Deferred`
- 備考:
  - 単一grid化により前提が消えるため、今後はGPU経路へ直接置換する。
- 背景:
  - `owner_indices/ghost_indices` の再構築コストが粒子移動時に支配的になりやすく、特に異なる空間levelを跨ぐ流れでFPS低下が顕著。
  - block彩色の前提が整ったため、まずは彩色並列を有効化せず、データ経路のみを `ghost` 非依存へ置換する。
- スコープ:
  - 粒子集合を `resident`（単一所属, G2P対象）と `support`（複数所属可, P2G寄与）へ分離する。
  - P2Gは `support` を使って active grid へ直接加算し、ghost参照経路を廃止する。
  - G2Pは `resident` のみ更新し、block境界跨ぎは outgoing queue で移管する。
- Subtasks:
  - [x] 用語/データ構造を `owner/ghost` から `resident/support` へ置換する（互換期間はalias可）。
  - [x] `resident` の所属規約を「粒子中心が block AABB 内の block に単一所属」として実装する。
  - [ ] `resident` は位置更新直後に即時移管する（block境界を出た時点で移管、`block-halo` 遅延は使わない）。
  - [x] `support` 規約を「block AABB + halo 内の粒子を保持」とし、AABB+halo から外れた粒子を除去する。
  - [ ] `support` 流入トリガを「粒子が block の `AABB-halo` 外へ出たら隣接block候補へ追加」にする（境界帯の先行登録）。
  - [x] blockごとに `outgoing_particles` キュー1本を持ち、隣接受け入れ側で自block流入分を選別して `resident/support` を更新する。
  - [x] `P2G -> Grid Update -> G2P -> 移管反映` の順序で1 tickを構成し、時間LoDで非更新blockが混在しても寄与欠落しないことを確認する。
  - [x] 回帰テストを追加する（境界通過、粗密境界通過、resident重複なし、support重複除去、質量保存）。
- Progress:
  - 2026-02-28: `MpmBlockIndexTable` を `resident_indices/support_indices/outgoing_particles` 構成へ拡張し、旧 `owner/ghost` アクセスは互換alias化。
  - 2026-02-28: `refresh_block_index_table` を resident 再配置 + support 再構築へ置換し、ghost再構築経路を廃止。
  - 2026-02-28: `step_block_set_coupled` を supportベースP2G / residentベースG2Pへ変更し、G2P後の outgoing キュー移管を追加。
  - 2026-02-28: scheduler から `ghost_refresh` フェーズを削除し、`refresh_block_table` 単体で active block 判定を完結。
  - 2026-03-01: MPM簡易プロファイラのG2P計測を `mpm::g2p_collect` と `mpm::g2p_sort_apply` に分割し、並列区間と直列テールを個別に可視化。
  - 2026-02-28: mpm水ユニットテストを resident/support 前提へ更新し、`cargo test --lib` 全通過を確認。
- 完了条件:
  - ghost経路を使わずに既存の空間LoDシナリオが成立し、境界停止/極端減速の再発がない。
  - `refresh_block_table` / ghost更新系コストが削減され、性能比較が可能な計測結果を取得できる。

### [MPM-WATER-07C] block彩色順フェーズ並列化（同色 `into_par_iter`）

- Status: `Deferred`
- 備考:
  - CPU並列最適化の追加は停止し、GPU pass最適化へ注力する。
- 背景:
  - block彩色（辺/頂点共有の排他）が成立したため、CPU並列化をアトミック加算なしで段階導入できる。
  - `resident/support` 化により、block更新責務（P2G/Grid Update/G2P/移管）がblock単位で明確化された。
- スコープ:
  - 色ごとに block list を作成し、1色内は `into_par_iter` で並列実行する。
  - 色フェーズ間は逐次実行（barrier）とし、同一step内の順序を固定して決定論性を維持する。
  - まずは P2G / Grid Update / G2P の block処理のみを色順フェーズへ置換し、既存スケジューラ構造は維持する。
- Subtasks:
  - [ ] `GridHierarchy` の `color_class` から `color -> Vec<block_index>` を構築するキャッシュを実装する。
  - [ ] blockレイアウト変化（split/merge/rebuild）時に color list キャッシュを再構築する。
  - [x] block更新ループを「色外側ループ + 同色 `into_par_iter`」へ置換する。
  - [x] 色フェーズ順を固定化し、実行ごとの順序非決定を排除する。
  - [ ] `resident/support/outgoing` 更新と整合するよう、色フェーズ境界で同期点を明示する。
  - [x] 回帰テストを追加する（逐次経路との一致、決定論性、境界通過、保存量）。
  - [ ] 計測を追加する（phase別 wall/cpu、色数別スケーリング、逐次比較）。
- Progress:
  - 2026-02-28: `step_block_set_coupled` の並列経路を色フェーズ実行へ変更（色ごと逐次barrier、同色内 `par_iter`）。
  - 2026-02-28: P2G（mass/pressure）, Grid Update, G2P を同一色フェーズ規約で統一し、色順を固定化。
  - 2026-02-28: 既存 `block_parallel_path_matches_serial_path` を含む `cargo test --lib` 全通過で逐次経路との整合を確認。
- 完了条件:
  - 色順フェーズ並列化で逐次経路と同等挙動（許容誤差内）を維持する。
  - 同条件で逐次実行よりCPU時間短縮が確認できる（少なくとも代表シナリオで改善）。

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

- Status: `Deferred`
- 背景:
  - 現在の優先目標は水MLS-MPM成立であり、粉体は先行フェーズの対象外とする。
  - 現行ランタイムは「水粒子が1つでも存在するとMLS-MPM経路のみ実行」されるため、水+粉体の混在シナリオでは粉体更新が停止する。
  - 既存粉体XPBDは水粒子接触を明示除外しており、水-粉体交換量は未定義である。
- スコープ:
  - 粉体を MLS-MPM へ統合するか、別ソルバ併存にするかの比較検討と連成仕様策定を後段で実施する。
  - 先行導入は「XPBD粉体 + MLS-MPM水の併存連成」を第一候補とし、統合MPMはPoC評価後に採否判断する。
  - 水優先フェーズ完了まで既存水シナリオの受け入れ基準を悪化させない（性能/安定性の回帰禁止）。
- Subtasks:
  - [x] 粉体モデル候補（Drucker-Prager / frictional MPM / XPBD併存）を比較し、段階導入案を作成する。（Design session: 2026-02-27）
  - [ ] v1連成交換量を定義する（`drag impulse`, `buoyancy impulse`, `normal reaction` の保存則と符号規約）。
  - [ ] mixed step順序を定義する（`MPM step -> coupling exchange -> XPBD granular -> reaction apply`）と、`dt`/substep整合規約を決める。
  - [ ] 受け入れシナリオを定義する（静水沈降、斜面流下、地形境界接触）と評価メトリクス（質量誤差、運動量収支、侵入率）を確定する。
  - [ ] 統合MPM（frictional/plastic）PoCの採否ゲートを定義する（安定性、実装複雑度、CPUコスト）。
- 完了条件:
  - v1（併存連成）実装に着手可能なI/F仕様、交換量定義、試験仕様が文書化される。
  - 統合MPMへ進むか継続併存とするかの判断条件が明文化される。

### [MPM-DEFER-02] 剛体連成の後段化

- Status: `Deferred`
- 背景:
  - 水優先フェーズでは、剛体との双方向連成を有効化しない。
- スコープ:
  - 剛体ソルバとの接触・反力交換I/Fを後段タスクとして定義する。
- Subtasks:
  - [ ] 連成I/F（接触点、法線、インパルス交換）を設計する。
  - [ ] 作用反作用の対称更新規約を設計する。
  - [ ] 剛体連成シナリオを定義する。
- 完了条件:
  - 剛体連成の実装着手条件が明文化される。

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

- Status: `In Progress`
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

- Status: `In Progress`
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

## Approval-Gated Items

- 物理統合テスト（headless scenario tests）のケース追加・閾値変更・ベースライン更新は、実装前にユーザー承認を取る。

## Design Feedback (from Impl sessions)

- `Design session 2026-02-24` 反映:
  - ADR-0001 に基づき、連続体基盤を MLS-MPM（明示）へ確定した。
  - フェーズ1の優先目標を「水シミュレーション成立」に固定し、粉体/剛体は `Deferred` へ後段化した。
- `Design session 2026-02-27` 反映（MPM-DEFER-01 粉体導入検討）:
  - 粉体導入は「XPBD粉体 + MLS-MPM水の併存連成」をv1推奨とし、統合MPMはPoC評価で後段判断とする。
  - 現行実装では mixed（水+粉体）時に粉体更新が停止するため、実装着手時はstep選択ロジックとCouplingWorld交換順序の明文化が必要。
  - `docs/design.md` へ反映が必要な項目:
    - 連成ポリシーに v1（併存連成）と v2（統合MPM評価）の段階導入規約を追加。
    - mixed material 時のランタイム実行順序と作用反作用記録規約を追加。
- 仕様詳細化が必要（水優先フェーズ）:
  - LoD境界のフラックス交換スキーム（同期点、補間、対称更新）を実装レベルで確定する。
  - 水の圧力モデル（`J` ベースと EOS ベース）の採用条件とパラメータ範囲を確定する。
  - 地形SDF境界補正の順序（Grid段 vs G2P段）と侵入率評価規約を確定する。
  - 地形境界SDFの供給方式を「未改変=生成関数、改変=差分LoDキャッシュ」のハイブリッドで確定する。
  - block並列時の粒子所有権規約（owner/ghost、unsafe経路のassertion条件）を確定する。
- 既存XPBD系フィードバックの扱い:
  - XPBD外力統合・境界インパルス負債の改善事項は、旧ソルバ保守タスクとして保持し、MLS-MPM水優先フェーズ完了後に再評価する。

## Done

- [x] [MPM-WATER-06] MPMグリッド質量場ベース水描画
  - 背景:
    - MPM化で物理更新は軽量化したが、水粒子数増加時に粒子投影描画が先にボトルネック化する。
  - スコープ:
    - 水描画を「粒子スプラット主体」から「MPMグリッド質量場主体」へ切り替える。
    - 輪郭抽出（marching squares）は導入せず、ぼかし+閾値で滑らかな水面を得る。
    - 孤立飛沫は P2G で得た描画専用集計を使って点描画し、全粒子走査を避ける。
  - Subtasks:
    - [x] 水描画入力を粒子密度場からグリッド質量場サンプリングへ置換する。
    - [x] 質量場アルファ閾値を独立定数として追加する。
    - [x] グリッド質量場へ軽いぼかしを適用し、閾値ベースでアルファ生成する。
    - [x] P2Gで描画専用の非カーネル集計（ノード `mass_sum` / `mass_pos_sum`）を蓄積する。
    - [x] `mass_sum` の粒子等価数で単粒子/小クラスタ飛沫を分類する。
    - [x] `mass_pos_sum` 由来の重心位置へ飛沫を点描画し、全粒子逆引きを回避する。
    - [x] 飛沫点描画を固定色から水パレットサンプルへ変更する。
    - [x] 水描画対象チャンクを水近傍に限定し、全ロード地形チャンクの再ラスタライズを回避する。
  - 完了条件:
    - 水粒子数増加時の描画負荷が低減し、連続水塊と孤立飛沫の視認性が維持される。

- [x] [MPM-WATER-01] 水向けMLS-MPM最小データ経路の構築
  - 背景:
    - ADR-0001 で連続体基盤を MLS-MPM へ切り替える方針が採択された。
    - まず水を動かせる最小経路を確立し、粉体/剛体連成は後段へ分離する。
  - スコープ:
    - 水粒子状態（`x,v,m,V0,F,C`）と格子状態（`m,p,v`）を保持する最小データ構造を導入する。
    - P2G/G2P で共通使用する補間カーネル評価レイヤを実装する。
  - Subtasks:
    - [x] `ContinuumParticleWorld`（Water専用）を追加する。
    - [x] `GridHierarchy` / `GridBlock` / `GridNode` の最小構造を追加する。
    - [x] カーネル重みと勾配の共通APIを実装する（P2G/G2P共通）。
    - [x] 決定論性（固定順序・固定しきい値）の単体テストを追加する。
  - 完了条件:
    - 水粒子と格子の最小データが ECS 上で更新可能になり、P2G/G2P 実装の土台が整う。

- [x] [PHYS-03] Sub-blockマルチレート更新（固定dt + 境界インパルス負債）
  - 背景:
    - 流体は微小振動が残りやすく、粒子Sleepだけでは `sleep -> 歪み蓄積 -> wake` の連鎖が起きやすい。
    - 可変dtの直接導入は境界同期が難しいため、固定dtのまま更新頻度で負荷を下げたい。
  - スコープ:
    - `Chunk` 正本を維持しつつ、物理更新を `Sub-block` 単位でマルチレート化する。
    - レート境界は「高頻度側解法 + 低頻度側への境界インパルス負債蓄積」で接続する。
    - 過剰負債や侵入時は強制昇格し、境界破綻を防ぐ。
  - Subtasks:
    - [x] `Sub-block` データ構造を `world` に追加する（更新レート、昇降格カウンタ、負債バッファ）。
    - [x] `solver` にレートクラス別スケジューラを実装する（`rate=1/2/4`、最大比 `1:4`）。
    - [x] レート境界拘束を実装する（高頻度側解法、低頻度側への反作用負債蓄積）。
    - [x] 低頻度ブロック更新時の負債適用順を実装する（負債適用 -> 拘束反復）。
    - [x] 強制昇格条件を実装する（負債閾値、侵入量、質量流束、接触継続時間）。
    - [x] 侵入時の近傍伝播昇格を実装する（最小Active保持フレーム付き）。
    - [x] 近距離領域では far-field 境界バッファを使わない分岐を実装する。
    - [x] `render` のdirty反映を `Sub-block` 起点で局所化する。
    - [x] `overlay` に更新レート可視化と負債ヒートマップを追加する。
    - [x] `Tracy` 計測を追加し、`rate scheduler / debt accumulate / debt apply` の内訳を確認する。
    - [x] 単体テストを追加する（昇格判定、負債適用順、境界条件）。
  - 完了条件:
    - 固定dtのまま流体の低活動領域で更新コストが低下し、境界破綻（大振動・連鎖wake）が抑制される。

- [x] [ARCH-01] `material` / `generation` モジュール再編
  - 背景:
    - `material_*` と `generation` の責務を分離し、パラメータ/乱数/ルールの見通しを改善したい。
  - スコープ:
    - `material_*` を `material/` サブモジュールへ移動し、`generation/` を `params/random/rules` に分割する。
  - Subtasks:
    - [x] `material` を `material/mod.rs`, `material/types.rs`, `material/defaults.rs` 構成へ移行する。
    - [x] `generation` を `generation/mod.rs`, `generation/params.rs`, `generation/random.rs`, `generation/rules.rs` 構成へ移行する。
    - [x] 生成パラメータに説明コメントを付与する。
    - [x] 公開API（`physics::material::*`, `physics::generation::*`）互換を `pub use` で維持する。
  - 完了条件:
    - 既存呼び出し側を壊さず、モジュール責務が分離される。

- [x] [WGEN-04] 地形起伏の2層化（大域+小域）
  - 背景:
    - 大域起伏を増やすと小域ディテールが相対的に不足するため、独立調整できる構成が必要。
  - スコープ:
    - 地表高さを macro/detail 2層ノイズ合成に変更し、土層厚みのランダム性も維持する。
  - Subtasks:
    - [x] `HEIGHT_NOISE_DETAIL_*` パラメータを追加し、詳細凹凸を独立調整可能にする。
    - [x] surface 高さ評価を macro/detail の合成へ変更する。
    - [x] 既存の確率場サンプラ（`sample_material_probabilities`）でも同一surface評価を用いる。
  - 完了条件:
    - 大域起伏と小域起伏を別パラメータで調整できる。

- [x] [REND-02] LOD Tile粒子の可視化強化
  - 背景:
    - 退避粒子がLODで消える、粒子が極小/薄色で視認しにくい問題があった。
  - スコープ:
    - Tile合成へ deferred 粒子を取り込み、LOD粒子の見た目（サイズ/色）を改善する。
  - Subtasks:
    - [x] `ParticleRenderChunkCache` に deferred 粒子キャッシュを追加する。
    - [x] LOD tile 合成で live + deferred 粒子を描画する。
    - [x] 標準LODが粗い状態でも、等倍タイル上で deferred 粒子可視化を維持する。
    - [x] LOD粒子描画を半径投影ディスク描画へ変更する（`1px` 固定を廃止）。
    - [x] LOD粒子色の濃度を引き上げる。
  - 完了条件:
    - 退避粒子がLODで消えず、粒子が視認可能なサイズ/濃さで表示される。

- [x] [PHYS-02] Parallelトグルの実効化
  - 背景:
    - UIの `Parallel` ボタンON/OFFで time profile が変わらず、設定が反映されていない疑いがあった。
  - スコープ:
    - active region 有効時の強制逐次化を除去し、並列経路でも領域判定を維持する。
  - Subtasks:
    - [x] `liquid` ソルバの強制 `parallel_enabled=false` を削除する。
    - [x] 並列経路に `active/halo` 判定を導入し、逐次経路と挙動整合を取る。
    - [x] 並列閾値を定数ではなく `solver_params.parallel_particle_threshold` 参照へ統一する。
  - 完了条件:
    - `Parallel` ボタンの切替が実際の並列経路選択に反映される。

- [x] Design Feedback反映: 物理演算領域（active/halo/inactive）、水境界、近傍グリッド安全策、Tile/LOD描画責務、連続LOD確率場、統計ブレンド規則を `design.md` に統合
- [x] [TST-01] 物理統合テスト artifact 可視化出力
  - 背景:
    - headless統合テストの数値判定は整備済みだが、目視比較のための画像artifact出力が未完了。
  - スコープ:
    - 既存シナリオランナーの終了状態から画像を生成し、JSON artifactと同じ出力規約で保存する。
  - Subtasks:
    - [x] `final_state.png` の生成処理を実装する。
    - [x] 画像出力の有効/無効をテスト実行引数または設定で切り替え可能にする。
    - [x] 出力画像の解像度・パレット・背景色の規約を定義する。
    - [x] READMEに画像artifact確認手順を追記する。
  - 完了条件:
    - 代表シナリオで `final_state.json` / `metrics.json` / `final_state.png` が同一run_id配下に出力される。

- [x] [WGEN-00] OpenSimplex地表生成（石 + 表土）
  - 背景:
    - 無限地形生成の初期段階として、まず地表高低差と地層の最小ルールを確定して実装する。
  - スコープ:
    - OpenSimplex ノイズで `surface_y(x)` を決定し、`Rock` と `Soil` を2層で生成する。
  - Subtasks:
    - [x] 地形生成パラメータを定義する（`WORLD_SEED`, `HEIGHT_NOISE_FREQ`, `HEIGHT_NOISE_AMP_CELLS`, `BASE_SURFACE_Y`, `SOIL_DEPTH_CELLS`）。
    - [x] ワールド座標 `x` から `surface_y(x)` を返す純関数を実装する（OpenSimplexベース）。
    - [x] 生成ルールを実装する（地表より上は `Empty`、地表付近は `Soil`、それより下は `Rock`）。
    - [x] チャンク境界で地形が不連続にならないことを確認する（隣接チャンク一致テスト）。
    - [x] 既存の差分セーブ/ロード経路と整合するよう、未変更チャンク再生成を確認する。
  - 完了条件:
    - 同一seedで同じ地形が再生成され、全チャンクで地表連続性が保たれる。

- Legacy Checklist
- (Keep completed tasks here; do not delete history)
- [x] テストケース再生用の簡易テストモードを追加（再生/停止/1step/ループ）
- [x] テストモードからartifact保存操作を追加（現在step/終了状態）
- [x] 開発手順をREADMEへ追記（簡易再生モード起動）
- [x] `tests/physics_scenarios.rs` を追加し、headless統合テストランナーを実装（固定dt手動step）
- [x] `ScenarioSpec` を導入し、初期状態・step数・判定閾値をデータ化
- [x] 基本シナリオを追加（単一物体落下、水のみ挙動、地形接触安定）
- [x] `physics_scenarios` の閾値判定を最終stepのみから常時判定（step 0..N）へ変更し、失敗ログに `step` と `condition` を出力
- [x] 条件付き assertion（例: `water_surface_height_p95`）の条件をUI表示に明示（`when: step >= N`）
- [x] シナリオ終了時artifact出力を実装（`final_state.json`, `metrics.json`）
- [x] 失敗時artifact保持と出力先規約を整備（`artifacts/tests/<scenario>/<run_id>/`）
- [x] 数値判定ユーティリティを実装（侵入率、最大速度、睡眠率など）
- [x] 開発手順をREADMEへ追記（headless実行方法、artifact確認方法）
- [x] v9物理テスト基盤仕様を `design.md` に追加（headless統合テスト + artifact + 簡易再生）
- [x] 並列化対応のため接触補正ワークバッファを導入（スレッドローカル集計 + reduce）
- [x] XPBD移行の回帰テストを追加（水挙動非退行、地形接触補正、砕石化直後のSleep lock）
- [x] 非水接触用 `GranularSolver` を追加し、水ソルバから実行パスを分離
- [x] 粉体-粉体のXPBD法線非貫通拘束を実装（`lambda_n` 蓄積 + `compliance_n`）
- [x] 粉体接触のXPBD摩擦拘束を実装（接線補正 + `mu_s` / `mu_k` クランプ）
- [x] 粉体接触の反発係数 `e` を速度更新段へ統合
- [x] 粉体-地形接触を共通XPBD接触APIへ統合（静的相手 `inv_mass=0`）
- [x] 粉体-オブジェクト接触を共通XPBD接触APIへ統合（反作用インパルス集計を接続）
- [x] `XPBD_CONTACT_COMPLIANCE_N/T` と `GRANULAR_SUBSTEPS/ITERS` を設定定数として追加
- [x] 粉体の接触Wake条件を強化し、破壊直後粒子の一時Sleep禁止を実装
- [x] Design Feedback反映: 非水接触のXPBD段階移行仕様を `design.md` §19 として追加
- [x] 地形セルの持続荷重評価量を実装（局所変位/歪み指標とサンプリング周期を定義）
- [x] 地形セルの持続荷重破壊判定を実装（`disp_or_strain > threshold` の継続時間で破壊）
- [x] 地形持続荷重判定をSleep/Wakeと統合（Activeセル中心に評価し、影響伝播でWake）
- [x] 地形破壊回帰テストを追加（片持ち形状の先端荷重で根元側が遅れて破断するケースを検証）
- [x] 粒子状態に `Active/Sleeping` フラグを追加し、全粒子共通で管理
- [x] Sleep判定を実装（`SLEEP_DISP_THRESHOLD` / `SLEEP_VEL_THRESHOLD` / `SLEEP_FRAMES`）
- [x] Wake判定を実装（接触/拘束補正、衝突インパルス、編集イベント）
- [x] Wake近傍伝播（`WAKE_RADIUS`）とヒステリシス閾値を実装
- [x] Sleep粒子の演算スキップを実装（積分・拘束・衝突の対象除外）
- [x] 振動防止の安定化を実装（最小Active保持フレーム等）
- [x] Sleep/Wake回帰テストを追加（流体含む全粒子での状態遷移）
- [x] `DETACH_FLOOD_FILL_MAX_CELLS` 定数を追加し、地形分離用の上限付きflood fillを実装
- [x] 地形分離判定を実装（`count <= limit` はオブジェクト化、`count > limit` は地形保持）
- [x] 地形→オブジェクト変換時の初期物理状態を実装（速度/角速度/姿勢の初期化）
- [x] 分割・分離の受け入れテストを追加（オブジェクト部分破壊、地形小塊分離、地形大塊保持）
- [x] 4近傍連結判定ユーティリティを実装（オブジェクト分割/地形分離で共通利用）
- [x] オブジェクト破壊時に対象オブジェクトのみ連結成分分解を実行
- [x] オブジェクト分割ロジックを実装（最大成分を既存 `ObjectId`、他成分を新規 `ObjectId`）
- [x] 分割後のオブジェクト再構築を実装（粒子集合・`rest_local`・質量・描画/SDFキャッシュ）
- [x] 破壊セルのみを除去し、無関係セルが同時破壊されないよう処理を修正
- [x] v7粒子Sleep/Wake仕様を `design.md` に追加（全粒子共通適用、流体は休眠しづらい前提、チャンク強制sleepは後段）
- [x] v6連結成分分割仕様を `design.md` に追加（4近傍分割 + 地形の上限付きflood fill近似）
- [x] セーブ/ロード後の整合性検証を実装（ID重複・参照不整合・境界外データ）
- [x] セーブデータのバージョニングを実装（`save_version` と互換不可時エラー）
- [x] マップ読込機能を実装（状態クリア後に `TerrainWorld` / `ParticleWorld` / `ObjectWorld` 再構築）
- [x] マップ保存機能を実装（地形・粒子・オブジェクト・最小設定をスナップショット化）
- [x] v4では粉体→固体逆変換を無効に固定（実装しないことをコード上で保証）
- [x] 粉体の接触＋摩擦モデルを実装（`mu_s`, `mu_k`, `e`, 必要なら転がり抵抗）
- [x] `Sand` / `Soil` を同モデルで追加し、低強度パラメータで崩れやすさを調整
- [x] `Rock(SolidCell) -> Gravel(GranularParticle)` 遷移を実装
- [x] 固体セル破壊時のセル→粉体粒子変換を実装（平方格子配置）
- [x] オブジェクト内部応力（拘束反力/ひずみ）閾値ベースの破壊判定を実装
- [x] 衝突インパルス閾値ベースの破壊判定を実装
- [x] マテリアルごとの `PARTICLES_PER_CELL` 定数を追加（同機構を水粒子サイズ調整にも適用）
- [x] マテリアル形態を導入（`SolidCell` / `GranularParticle`）し、`Rock` / `Gravel` / `Sand` / `Soil` を定義
- [x] Design Feedback反映: `ObjectPhysicsField` 仕様を固定配列グリッド（候補 `ObjectId` 固定長保持）へ更新（`design.md` §13.7）
- [x] Design Feedback反映: 衝突判定仕様を「候補IDはphysics grid、`distance/normal` はローカルSDF直接評価」へ更新（`design.md` §13.7）
- [x] v4粉体・破壊遷移仕様を `design.md` に追加（マテリアル形態、破壊トリガ、セル→粒子変換）
- [x] v5マップセーブ/ロード仕様を `design.md` に追加（保存対象、形式/互換、基本フロー）
- [x] ツールバーtooltipを最前面表示へ変更（`GlobalZIndex` を付与）
- [x] 粒子物性を `physics/material.rs` に集約（`ParticleMaterial` / 半径 / 質量 / 接触パラメータ）
- [x] 岩同士の接触応答を調整（低反発・高摩擦の速度応答を追加）
- [x] 地形境界の密度拘束にゴースト寄与を追加（`density_lambda` と `compute_delta` の両方で境界近傍圧力を反映）
- [x] 地形 `boundary_push` をフェイルセーフ寄りに調整（水粒子は侵入時の押し戻し係数を1.0）
- [x] `physics::compute_delta` を自前並列化（1粒子1書き込み + ワークバッファ方式）
- [x] `density_lambda` を自前並列化（スレッドローカルscratch利用）
- [x] `density_lambda` / `compute_delta` で近傍gather結果を再利用し、同一反復内の重複探索を削減
- [x] 近傍グリッドを連続メモリ表現へ最適化（`cell_start/end` 方式）
- [x] 共有加算パスをスレッドローカル集計 + reduceへ統一（反作用インパルス等）
- [x] 反復回数の適応化を導入（誤差閾値ベース、必要時のみ）
- [x] 並列化後の挙動回帰を確認（既存テスト + 長時間実行で安定性確認）
- [x] v3並列化方針を `design.md` に確定（Tracy優先順位 + CPU並列化順序 + GPU移行互換制約）
- [x] Tracyを `bevy/trace + LogPlugin(custom_layer)` 構成へ変更し、`bevy/trace_tracy` 依存を回避
- [x] Tracy実行時のみ `dynamic_linking` を無効化できるfeature構成へ変更（通常実行は高速リンクを維持）
- [x] Tracy実行時の起動不安定対策として、実行feature構成を見直し（通常実行とTracy実行の起動パスを分離）
- [x] Bevy の Tracy プロファイリング設定を追加（`tracy` feature）し、物理ステップ各処理を `physics::*` スパンで計測可能化
- [x] README に Tracy 実行手順を追加（通常実行 / `cargo watch` 監視実行）
- [x] `SimulationPlugin` を廃止し、`main.rs` で `PhysicsPlugin` / `InterfacePlugin` / `OverlayPlugin` / `RenderPlugin` / `CameraControllerPlugin` を個別有効化
- [x] `src/simulation/` を `src/physics/` へ改名し、物理本体モジュールの名前空間を `physics::*` に統一
- [x] ディレクトリ構造を再編（`src/simulation` / `src/overlay` / `src/interface` / `src/render`）
- [x] UI責務を `interface` モジュールへ改名・移設し、`InterfacePlugin` として分離
- [x] Overlay/Render をそれぞれ独立Pluginとして別モジュール化し、`SimulationPlugin` は構成オーケストレーションに集約
- [x] シミュレーション責務をPlugin分割（`PhysicsCorePlugin` / `UiPlugin` / `OverlayPlugin` / `RenderPlugin`）し、`SimulationPlugin` は構成管理に限定
- [x] モジュール構造を再編（`state.rs` に共有Resource/SystemSet、各Pluginを個別モジュールへ分離）
- [x] カメラ操作Plugin（`CameraControllerPlugin`）を独立維持し、`main.rs` から Simulation群と並列に登録する構成を維持
- [x] `ObjectPhysicsField` を `HashMap<IVec2, ...>` から固定世界サイズの配列ベース実装へ置き換え（セル->候補ObjectIdの固定長格納）
- [x] 物体ごとのローカルSDF（距離場）を生成・保持するデータ構造を追加（`ObjectLocalSdf`）
- [x] 動いた物体のみ、物理用SDFグリッド（疎セル）へ再投影するdirty更新を実装（削除時は再初期化）
- [x] Broadphaseを実装: 物体AABBを空間グリッドに登録し、再投影範囲を局所化
- [x] 流体粒子の干渉判定を変更: 物理用SDFグリッド参照で `distance/normal/object_id` を取得
- [x] 流体-物体干渉の反作用集計を実装（最小距離寄与の `object_id` へインパルス蓄積）
- [x] 受け入れ確認を実施（`cargo check` / `cargo test` 実行、流体-物体反作用テスト追加）
- [x] 人工物描画方針を実装: `1オブジェクト=1Entity`（`Transform` + 描画コンポーネント + `ObjectId/Handle`）
- [x] 人工物のドット描画をオブジェクトローカル基準へ変更（移動・回転時にパターンが追従）
- [x] 形状不変時は `Transform` 更新のみ、形状変更時のみテクスチャ再ラスタライズする更新フローを実装
- [x] 自然物（流体）は従来どおり世界側テクスチャ投影で描画する経路を維持（人工物描画と分離）
- [x] `ObjectWorld` を追加し、`ObjectId` / `particle_indices` / `rest_local` / `mass_sum` / `shape_stiffness_alpha` / `shape_iters` を管理
- [x] Stoneドラッグの1ストローク追跡を実装（開始・更新・終了イベントの収集）
- [x] Stoneドラッグ中の生成セルを分類（`Frozen` 地形に重なる/接続するセルは地形、その他は物体候補）
- [x] マウスリリース時に物体候補セルから粒子集合を生成し、1ストローク=1オブジェクトとして登録
- [x] 物体作成時に `rest_local` をCOM基準で初期化（`sum(m_i * q_i)=0`）
- [x] `FixedUpdate` のsubstepへ形状マッチング投影を統合（2D極分解で `R` を計算）
- [x] 形状マッチング反復回数 `shape_iters` と剛性 `shape_stiffness_alpha` の調整パラメータを導入
- [x] 速度更新を形状マッチング後に整合させ、過大速度をクランプ
- [x] 物体候補セルが空の場合はオブジェクトを作らず、地形生成のみ行う分岐を実装
- [x] 受け入れ確認を実施（ほぼ剛体挙動、Stoneドラッグ地形判定、1ストローク1オブジェクト）
- [x] Grid Overlay表示時に、オブジェクト（人工物）のセルグリッドを重ね描きする可視化を追加
- [x] Design Feedback反映: 水-岩干渉仕様を「岩SPH寄与なし + SDF `boundary_push`」へ更新（`design.md` §12.1/§12.5）
- [x] Design Feedback反映: 新規定数 `TERRAIN_SDF_SAMPLES_PER_CELL` / `TERRAIN_SDF_PUSH_RADIUS_M` / `TERRAIN_REPULSION_STIFFNESS` を仕様へ追加（`design.md` §12.2）
- [x] Design Feedback反映: 水描画仕様に「岩セル上の水ドット抑制・岩優先表示」を追加（`design.md` §8.2）
- [x] 水-地形干渉仕様を更新（セルAABB押し戻しを廃止し、SPH一本化を `design.md` に反映）
- [x] 水-地形干渉方針を更新（地形全粒子化 + 水-地形SPH + AABB投影をフェイルセーフ化）
- [x] 水粒子v1アルゴリズム仕様を `design.md` に確定（PBF/XPBD + Frozen地形干渉 + Space/R操作）
- [x] 地形v0実装仕様の確定（`cell_size=0.25m`, `chunk=32x32`, `TerrainWorld/ParticleWorld` 分離、Plugin/描画フロー）
- [x] Bevy開発環境の構築（Rust + Bevy 0.18系）
- [x] 開発高速化設定の導入（`bevy/dynamic_linking`, `bevy/file_watcher`）
- [x] ホットリロード方針の確定と検証（アセット再読込 + コードホットパッチ可否）
- [x] ファイル監視ベースの自動再コンパイル・自動再ロード手順の整備
- [x] 初期運用方針を「安定運用のみ」に固定（`file_watcher` + `dynamic_linking`、`hotpatching` は採用しない）
- [x] 地形データ管理の初期実装（1ブロック=1セル、全地形 `Frozen` 固定）
- [x] 地形の初期生成・保持・参照APIを実装（チャンク/グリッド単位の読み書き）
- [x] シミュレータ基本構造を実装（`ParticleWorld`/`TerrainWorld` Resource と更新ループの骨組み）
- [x] Bevy Plugin化（`SimulationPlugin`）して `App` から有効化できるようにする
- [x] `FixedUpdate` にシミュレータ更新システムを接続（現段階では地形は非動的更新）
- [x] 地形描画の初期実装（グリッド->タイル/スプライト描画）
- [x] 地形更新と描画更新の接続（変更セルのみ dirty 更新）
- [x] 最小動作確認シーンを作成（地形の生成・表示・カメラ追従）
- [x] カメラ操作を実装（スクロール拡大縮小、`WASD`/矢印パン、中ボタンまたは`Option(Alt)+ドラッグ`パン）
- [x] カメラ処理をモジュール分離し、調整定数を先頭へ集約（`camera_controller.rs`）
- [x] 拡大縮小をカーソル位置基準に変更し、スクロール感度を調整
- [x] 地形描画を単色から「materialごと4色パレット + 決定論的ランダム埋め」に変更
- [x] 世界生成を固定`4x4`チャンクに変更し、カメラ移動時の追加チャンク生成を停止
- [x] 世界の左右端に上端までの壁を追加（上境界はオープン）
- [x] 水粒子シミュレーションの定数セットを実装（重力、substep、反復回数、半径、粘性）
- [x] `ParticleWorld` を水粒子SoA構造に拡張（`pos/prev_pos/vel/mass/material` + ワークバッファ）
- [x] 動的粒子用の近傍探索グリッドを実装（substepごと再構築）
- [x] 水の密度拘束ソルバ（PBF/XPBD系）を実装（density/lambda/delta_pos）
- [x] 水粒子と `Frozen` 地形セルの衝突投影を実装（円 vs セルAABB）
- [x] `FixedUpdate` に水粒子stepを接続（`SUBSTEPS` ループ）
- [x] 水粒子の可視化を実装（まずはデバッグ描画で可）
- [x] `Space` キーでシミュレーション再生/停止トグルを実装
- [x] `R` キーで初期状態リセットを実装（地形再生成 + 水粒子再生成 + 停止状態）
- [x] 受け入れ条件テストを実施（落下、非貫通、停止中不変、リセット復元）
- [x] 地形 `Solid` セルの全粒子化を実装（セル中心に静的地形粒子を生成）
- [x] `TerrainWorld` に静的地形粒子バッファと再構築処理を追加（地形変更時のみ更新）
- [x] 水近傍探索に地形静的粒子グリッドを統合（`3x3` 近傍参照）
- [x] 水の密度拘束（rho/lambda/delta）へ地形静的粒子の寄与を追加
- [x] 水-地形衝突のセルAABB押し戻し処理を削除し、干渉をSPHベースへ一本化
- [x] 地形境界の過密緩和を確認する検証シーン/テストを追加
- [x] 水描画をドット絵化（SPHに基づきセル内ドットへ存在度を分配し、閾値超えドットを水パレットで塗る）
- [x] 左上HUDにFPSと潜在最大FPSを表示（潜在最大FPS = 1 / フレーム内の実物理処理時間）
- [x] 粒子デバッグオーバーレイを実装（地形粒子・水粒子を半径円で表示）
- [x] 右下UIボタンでデバッグオーバーレイ表示を切り替え
- [x] 近傍探索グリッドとチャンク境界を可視化するグリッドオーバーレイを実装
- [x] 右下UIにグリッドオーバーレイトグルボタンを追加
- [x] 粒子デバッグオーバーレイを円から点表示へ変更（描画負荷軽減）
- [x] 粒子デバッグオーバーレイを低解像度円表示へ調整（見た目サイズを維持しつつ軽量化）
- [x] REST_DENSITY基準の水描画半径を導入し、水オーバーレイ半径と水ドット描画に適用
- [x] 水ドット描画にseparable blur + smoothstepを追加し、液体境界を滑らか化
- [x] WATER_DOT_THRESHOLDをREST_DENSITY比へ変更（密度場を mass*kernel で評価）
- [x] blur半径を定数化し、`WATER_BLUR_RADIUS_DOTS=3`で調整可能に変更
- [x] 通常状態のマウスドラッグで水粒子へ速度を付与する操作を追加
- [x] 画面下ツールバーを追加し、`Water`/`Stone`/`Delete` ツール選択を実装
- [x] `Water` ツールのドラッグで水粒子を生成
- [x] `Stone` ツールのドラッグで地形（Frozen）を生成
- [x] `Delete` ツールで一定半径内の水粒子と地形を削除
- [x] `Esc` キーでツール選択を解除
- [x] Stoneツールを1セル単位描画に変更（カーソルサイズ1）
- [x] Water生成をドラッグ軌跡ベースへ変更（標準粒子間隔に応じた個数を配置）
- [x] `h_water/dx` と `h_rock/dx` を分離し、水-岩カーネル半径を平均で混合する方式へ変更
- [x] 境界フェイルセーフの `boundary_push` を無効化し、水-岩干渉をSPH拘束のみへ統一
- [x] 水-岩干渉を `boundary_push` のみに切り替え（岩のSPH寄与は無効化）
- [x] 岩境界干渉をSDFベースの押し戻しへ置換（地形粒子近傍ループを廃止し、法線方向へ連続押し戻し）
- [x] 描画時に岩セル上の水ドット生成/描画を抑制し、重なり時は岩ドットを優先表示
