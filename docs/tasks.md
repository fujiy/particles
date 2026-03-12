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
  - [x] GPU MPMの最小受け入れとして `water_drop` を autoverify で通す。
  - [ ] 残存する不要CPU経路（連続体/旧互換）を削除し、参照ドキュメントを整理する。
- 完了条件:
  - `water_drop` がGPU経路で安定再現され、品質指標を自動計測できる。
  - GPU本番経路に対して不要なCPU依存が整理されている。
- 進捗（要約）:
  - 2026-03-06: `water_drop` 自動検証で `passed=true` を確認。
  - 2026-03-06: GPU-first化に伴う dead-path/未使用経路の大規模削除を継続。
  - 詳細履歴は `docs/tasks_archive/2026-03-07-before-chunk-structuring.md` を参照。

### [REND-GPU-01] GPU常駐地形描画パイプライン（Near/Far 2層キャッシュ + リングバッファ）

- Status: `In Progress`
- 背景:
  - 現行の `terrain_dot_gpu` 由来経路には、dirty差分更新・予算制御・ズーム遷移の未完項目が残る。
- スコープ:
  - Near/Far の2層キャッシュをGPU常駐で運用し、CPU転送を「dirty差分のみ」に限定する。
  - dirtyキュー + 予算スケジューラ + ズーム遷移を `render.md` 準拠で完成させる。
- Subtasks:
  - [x] WGSL生成関数移植、Near/Farキャッシュ定義、Near更新・Compose基盤を実装する。
  - [x] RenderGraph統合と旧 `terrain_dot_gpu` 削除を完了する。
  - [ ] Far更新の予算内逐次再サンプリングを実装する。
  - [ ] カメラ同期（パン/ズーム）と dirty キュー制御を仕上げる。
  - [ ] `terrain_cell_correctness` / `terrain_pan_continuity` / `terrain_zoom_lod` の autoverify を完了する。
- 完了条件:
  - Near/Far 2層キャッシュがパン・ズームで安定動作する。
  - 予算制御下で更新コストが規定内に収まり、autoverify 判定が成立する。
- 進捗（要約）:
  - 2026-03-05: Near/Far/Back の表示品質と更新安定性を段階改善。
  - 2026-03-05: runtime 調整値を `assets/params/render.ron` 中心へ統合。
  - 詳細履歴は `docs/tasks_archive/2026-03-07-before-chunk-structuring.md` を参照。

### [REND-GPU-02] 地形改変のNear反映（編集入力 + セーブロード + override差分転送）

- Status: `In Progress`
- 背景:
  - 改変地形の表示反映はNearで成立したが、Far/Back 反映の最適化（mip差分転送）が未完了。
- スコープ:
  - 改変台帳を正本として Near/Far/Back へ一貫反映し、スパイクを抑えた差分転送へ完成させる。
- Subtasks:
  - [x] 改変チャンク台帳、編集入力、セーブロード復元、Near反映を実装する。
  - [x] RLE差分転送と Compose の override 優先参照を実装する。
  - [x] Far/Back への反映を sparse lookup 方式で接続する。
  - [ ] override chunk の `2^k` mipキャッシュ構築を実装する。
  - [ ] scaleごとの必要mip差分のみ送信する payload 方式へ移行する。
  - [ ] scale変更時の再構築を分割実行してフレームスパイクを抑制する。
- 完了条件:
  - 改変反映が Near/Far/Back で成立し、ズーム時も予算内で更新できる。
- 進捗（要約）:
  - 2026-03-05: delete/solid/load の自動検証成果物を整備。
  - 2026-03-05: Near外改変の Far/Back 反映を shader lookup で成立。
  - 詳細履歴は `docs/tasks_archive/2026-03-07-before-chunk-structuring.md` を参照。

### [MPM-PHYS-GRANULAR-01] physics.md 改訂に基づく粉体GPU実装ギャップ解消

- Status: `In Progress`
- 背景:
  - DP射影・水粉体連成の主要実装は更新済みだが、検証メトリクス拡張が未完了。
- スコープ:
  - 粉体構成則・連成則を `physics.md` と一致させ、受け入れ判定を artifact ベースで完結させる。
- Subtasks:
  - [x] 状態量 `v_vol` 化、SVD+log-strain 射影、Neo-Hookean 応力評価を実装する。
  - [x] 水-粉体連成（法線+接線+対称更新）を更新する。
  - [ ] autoverify/シナリオに `v_vol` ドリフト、相間インパルス収支、mixed侵入率の指標を追加する。
- 完了条件:
  - `physics.md` の式と実装が整合し、granular-only/mixed の新指標が閾値内を満たす。
- 進捗（要約）:
  - 2026-03-03: 主要な実装不一致（`F^{-T}` 等）を修正し、回帰症状を解消。
  - 詳細履歴は `docs/tasks_archive/2026-03-07-before-chunk-structuring.md` を参照。

### [MPM-CHUNK-01] GPU chunk構造導入 + chunk SDF初期反映（静的residency, water_drop完遂）

- Status: `In Progress`
- 背景:
  - `docs/chunks.md` 仕様に沿って、MPM計算領域を world-dense 前提から resident chunk slot 前提へ移行する。
  - 実装初段では検証容易性を優先し、CPU↔GPU の movers/residency差分更新を導入しない。
- スコープ:
  - GPU側に chunk slot メタ構造（slot座標、neighbor slot、active mask）を導入する。
  - MPM格子バッファを `slot_id * 1024 + local_node_index` の addressing で参照する。
  - chunkごとの地形SDF/normalバッファを導入し、**初期地形のみ**を one-shot で反映する。
  - 起動時に resident chunk set を一度だけ構築し、実行中は固定（静的residency）とする。
  - CPUとのやりとりによる chunk buffer 更新（movers readback / diff upload）は本WUの対象外とする。
- Subtasks:
  - [x] `ChunkMetaBuffer` / `ChunkSdfBuffer` / `neighbor_slot_id[8]` のGPUレイアウトを確定する。
  - [x] 起動時 one-shot の resident chunk 初期化（occupied + halo）を実装する。
  - [x] P2G/Grid/G2P の node 参照を chunk slot addressing 経路へ切替する。
  - [x] SDF/normal 参照を chunk SDF バッファ経路へ切替し、境界補正を成立させる。
  - [x] `water_drop` autoverify で `passed=true` を達成する。
  - [x] 検証artifactに `resident_chunk_count`, `invalid_slot_access_count`, `chunk_sdf_samples` を出力する。
  - [x] Chunk Overlay / Physics Area Overlay を統合し、`chunk_meta_buf` を使うGPU描画へ移行する。
  - [x] Overlay UI を1ボタン化し、overlay詳細を左上HUDへ集約。GPU overlay線をズーム非依存1px化して chunk境界+grid を描画する。
  - [x] SDF Overlay のCPU描画経路（gizmo/sprite）を廃止し、`terrain_sdf_buf` を参照するGPU描画経路へ切替する。
- 完了条件:
  - CPUとの chunk 差分同期なしで `water_drop` が最後まで完走し、既存判定を満たす。
  - 実行中に invalid slot 参照が発生しない。
  - chunk SDF 経由の境界反映で侵入率が閾値内に収まる。
- 進捗:
  - 2026-03-07: GPU側に `chunk_meta_buf` を追加し、静的residencyの one-shot 構築（起動時 / 粒子再upload時）を導入。resident chunk bbox から `grid_origin/grid_dims` を再計算して MPM 実行領域を world-dense 固定から切替。
  - 2026-03-07: 初期地形から chunk SDF/normal を one-shot 生成してGPU uploadする経路を実装。`grid_update` と統計passは同バッファを参照。
  - 2026-03-07: autoverify report へ `resident_chunk_count`, `invalid_slot_access_count`, `chunk_sdf_samples` を追加。`configs/autoverify/water_drop_motion.json` 実行で `passed=true`, `invalid_slot_access_count=0` を確認。
  - 2026-03-07: Physics Area Overlay を Chunk Overlay に統合し、GPU側 `chunk_meta_buf` + `params_buf` を参照する描画パスへ移行。`water_drop` スクリーンショット検証で `resident:64` のオーバーレイ表示を確認（`artifacts/autoverify/overlay_chunk_physics_water_drop.png`）。
  - 2026-03-07: Overlay UI を `Chunk/Grid Overlay` ボタン1つに統合し、詳細表示は左上HUDへ移設。WGSLを `fwidth` ベースへ更新してズームに依存しない1px線へ変更し、chunk境界に加えて内部grid線を描画することを `overlay_chunk_zoom_out.png` / `overlay_chunk_zoom_in.png` で確認。
  - 2026-03-07: SDF Overlay をGPU pass化（`sdf_overlay_gpu.wgsl`）し、`terrain_sdf_buf` を直接サンプリングする描画へ変更。`artifacts/autoverify/sdf_overlay_gpu_water_drop.png` で描画成立を確認。
  - 2026-03-07: `mpm_types.wgsl` の `node_index/node_in_bounds` を `chunk_origin/chunk_dims/chunk_node_dim` ベースへ変更し、P2G/G2P/GridUpdate を chunk slot addressing 参照へ切替。CPU側 `terrain_sdf/normal` upload も slot-major 並びに変更。`water_drop_motion` 再検証で `passed=true`, `invalid_slot_access_count=0` を確認。
  - 2026-03-10: solver 格子を `NODE_SPACING_M=0.125`, `chunk_node_dim=32` へ移行し、1 chunk = 16x16 cells = 32x32 nodes, 1 tile = 8x8 nodes, 1 chunk = 4x4 tiles の構成へ更新。`world_grid_layout` と active tile 統計を node 基準へ切替し、terrain occupancy upload は cell->2x2 half-cell 複製へ変更。`cargo test --lib` と `configs/autoverify/water_drop_motion.json` 再検証で `passed=true`, `active_tile_count=54`, `chunk_sdf_samples=36864`, `runtime_rebuild_count=0` を確認。
  - 2026-03-10: 追加の見た目調整として、水粒子ドットの密度グリッド origin を camera-centered から固定 world origin へ戻し、パン時もドット模様がワールド座標基準で安定するよう修正。particle overlay は塗りつぶし円から `fwidth` ベースの輪郭リングへ変更し、`cargo check` / `cargo test --lib` / `configs/autoverify/overlay_particle_water_drop.json` で回帰確認とスクリーンショット検証を行う。

### [MPM-CHUNK-02] occupancy/residency event同期 + exceptional mover fallback

- Status: `In Progress`
- 背景:
  - `MPM-CHUNK-01` は静的residencyであり、chunk跨ぎ移動が増えるケースでは成立しない。
- スコープ:
  - 通常の隣接chunk移動は GPU 内で `home_chunk_slot_id` を更新し、CPU/GPU 同期は occupancy/residency change event に縮退する。
  - 粒子ごとの mover readback は exceptional mover（8近傍外移動）向け fallback として維持する。
- Subtasks:
  - [x] GPU chunk meta に `particle_count_curr/next`（または occupied bit 同等情報）を追加し、ステップ内 occupancy 遷移を抽出可能にする。
  - [x] GPU 側で `newly occupied` / `newly empty` / `frontier expansion request` の chunk event 抽出パスを実装する。
  - [x] `exceptional mover` は既存 mover 抽出パスを fallback として維持し、通常 chunk event 同期と分離する。
  - [x] Chunk/Grid Overlay の occupied/halo/free 可視化を GPU `chunk_meta` 更新結果のみで成立させ、overlay 表示状態に依存した CPU 同期経路を排除する。
  - [x] CPU 側で event 駆動の `occupied_particle_count` / `resident_ref_count` 更新、slot allocate、`neighbor_slot_id` 差分更新を実装する。
  - [ ] `not occupied && not halo` 条件での slot free/reuse を安定化して常時有効化する（現状は free 経路を保留）。
  - [x] slot table / `neighbor_slot_id` の GPU 反映を非同期キュー化し、通常ステップを同期停止させない。
  - [x] CPU->GPU 反映を `chunk meta diff` 中心に切替し、通常フレームで mover result upload を不要化する。
  - [x] exceptional mover 発生時のみ既存 mover 解決ルートを起動する分岐を実装する。
  - [ ] `water_drop_motion` / `sand_water_interaction_drop` / 境界跨ぎ高頻度ケースで、`runtime_rebuild_count=0` かつ性能改善を確認する。
- 完了条件:
  - 通常ケースでは粒子 mover 全件 readback なしで `home_chunk_slot_id` と chunk table が整合し続ける。
  - exceptional mover のみ fallback で正しく収束する。
- 進捗:
  - 2026-03-07: 新方針へ再定義。同期単位を mover 全件から occupancy/residency change event に変更し、mover全件同期は exceptional fallback 専用に縮退する方針を確定。
  - 2026-03-07: mover 抽出/読出/結果反映パス（`mpm_extract_movers` / `mpm_apply_mover_results`）を追加し、GPU→CPU readback と CPU→GPU 反映バッファを導入。
  - 2026-03-07: CPU 側に `occupied_particle_count` / `halo_ref_count` を持つ slot 台帳を追加。mover 差分から occupied 0↔1 遷移を更新し、resident slot を incremental に維持。
  - 2026-03-07: `neighbor_slot_id` を resident 変化 slot + Moore近傍で再計算し、`chunk_meta` 差分 upload（多件時は全量 upload fallback）を実装。
  - 2026-03-07: `water_drop_motion` と `sand_water_interaction_drop` の autoverify を実行し、いずれも `passed=true` / `invalid_slot_access_count=0` を確認。
  - 2026-03-07: mover 同期を strict lockstep 化。`readback適用 -> upload準備` の順序セットを明示し、`pending_mover_readback/pending_mover_apply` が解消するまで次 substep を停止。`mpm_apply_mover_results` の ACK を main world へ返し、ACK 受領で mover_result バッファを明示クリアする経路へ変更。
  - 2026-03-07: mover readback を「GPU copy が発生したフレームのみ map」するよう修正し、同一 mover バッファの再読を防止。`update_halo_ref_count` の境界処理も見直し、chunk layout 境界での不要 rebuild を除去。
  - 2026-03-07: autoverify report へ `runtime_rebuild_*` / `pending_mover_*` 指標を追加。再検証（`water_drop_motion`, `sand_water_interaction_drop`）で `runtime_rebuild_count=0` を確認。
  - 2026-03-07: `mpm_extract_movers` を更新し、隣接 chunk 移動は GPU 内で `home_chunk_slot_id` を即時更新。`mover` readback は frontier/8近傍外など unresolved 粒子のみを返す exceptional fallback 経路へ縮退。
  - 2026-03-07: `ChunkMetaBuffer` に `particle_count_curr/next` と `occupied_bit_curr/next` を追加。`apply_gpu_readback` で readback snapshot から occupancy 変化を再構築し、`chunk_meta diff` を非同期反映する経路へ変更（GPU step の同期停止を撤去）。
  - 2026-03-07: `prepare_gpu_run_state` の strict lockstep を撤去して substep 実行を再許可。`water_drop_motion` / `sand_water_interaction_drop` の autoverify はいずれも `passed=true`。現状は両シナリオで `runtime_rebuild_count=1`（`new_chunk_oob=1`）が残るため、閾値0達成は未完。
  - 2026-03-07: runtime rebuild 発火時の `requested/start/pending` ログを追加し、`water_drop` シナリオを上下左右の壁で閉領域化。`water_drop_motion` で `passed=true` / `runtime_rebuild_count=0` を再確認。
  - 2026-03-07: overlay 色設定を `assets/params/overlay.ron` に移管（occupied/halo/free の edge/grid 色）。`chunk_physics_overlay_gpu.wgsl` は uniform 参照へ変更し、色のハードコードを廃止。
  - 2026-03-07: `apply_gpu_readback` で生成した `chunk_meta_diffs` が prepare 先頭クリアで失われる不整合を修正（`pending_snapshot_refresh` で prepare 側適用へ移動）。
  - 2026-03-07: `mpm_chunk_meta_update.wgsl`（clear/accumulate/finalize）を追加し、GPU 側で per-step に `particle_count`/`occupied`/`resident(halo)` を更新。Chunk/Grid Overlay の occupied/halo/free 更新を CPU readback 非依存化。
  - 2026-03-07: `mpm_extract_chunk_events.wgsl` と `chunk_event_*` readback バッファを追加し、`newly occupied/newly empty/frontier` を GPU で抽出して CPU へ同期する経路を実装。CPU 側は event 駆動で `occupied_particle_count/halo_ref_count/resident` を更新し、`chunk_meta diff` upload へ接続。
  - 2026-03-07: `pending_snapshot_refresh`（全粒子 readback 差分再構築）経路を撤去し、chunk table 更新の主経路を chunk event readback へ切替。
  - 2026-03-07: `configs/autoverify/water_drop_motion.json` で再検証し `passed=true` を確認。現状は `runtime_rebuild_count=18`（frontier request 起因）が残っており、rebuildゼロ化は未完。
  - 2026-03-07: CPU residency を `coord->slot` map（`chunk_to_slot`/`slot_to_chunk`）で保持する経路を導入。frontier event は rebuild せず halo refcount を再構築して resident を更新する方針へ変更。
  - 2026-03-07: slot window を `MAX_RESIDENT_CHUNK_SLOTS` 固定矩形（16x16）として初期化し、runtime の frontier-rebuild を抑止。`water_drop_motion`/`sand_water_interaction_drop` で `runtime_rebuild_count=0` を確認。
  - 2026-03-07: 現状の制約として、fixed slot window 外へ halo が必要になった場合は `runtime_rebuild` せず警告ログのみ（`frontier halo reached slot-window edge`）。`sand_water_interaction_drop` で `invalid_slot_access_count=1` / `runtime_rebuild_reason_halo_update_fail=42` を確認。
  - 2026-03-07: GPU 側の `chunk_origin/chunk_dims` 依存を縮小するため、`mpm_chunk_meta_update` の occupancy 集計を world→slot 逆算から `particle.home_chunk_slot_id` 参照へ変更し、resident 判定も `neighbor_slot_id` 経路に切替。
  - 2026-03-07: `mpm_p2g` / `mpm_g2p` / `mpm_grid_update` を `home_slot + neighbor_slot_id + chunk_meta.chunk_coord(slot->coord)` ベースの node 参照へ更新。`water_drop_motion` / `sand_water_interaction_drop` 再検証で `passed=true` / `runtime_rebuild_count=0` / `invalid_slot_access_count=0` を確認。
  - 2026-03-07: CPU slot table を free-list 付き pool へ拡張し、frontier event 時に halo chunk を動的 allocate する経路を追加。`prepare_terrain_upload` の初期構築も resident chunk 集合ベースへ変更し、terrain SDF は slot->coord から再生成。
  - 2026-03-07: exceptional mover で未登録 chunk へ飛んだ際も CPU 側で slot allocate して fallback 解決できるよう更新。再検証で `water_drop_motion` / `sand_water_interaction_drop` は `passed=true` / `runtime_rebuild_count=0`。
  - 2026-03-07: free/reuse を有効化した実験で `sand_water_interaction_drop` の `invalid_slot_access_count` と assertion 不安定化が発生したため、現時点では free 経路を保留（allocate-only 運用）に戻して安定性を優先。

### [MPM-CHUNK-03] active tile再構築と sparse実行最適化

- Status: `In Progress`
- 背景:
  - chunk構造導入後の性能は active tile 最適化の有無に強く依存する。
- スコープ:
  - resident chunk 内で active tile mask を毎step再構築し、clear/grid update を sparse 化する。
- Subtasks:
  - [x] chunkごとの `active_tile_mask` 初期化・mark（neighbor含む）を実装する。
  - [x] clear/grid update を active tile 限定 dispatch へ切替する。
  - [x] `active_tile_count` / `inactive_skip_rate` を計測する。
  - [x] `water_drop` と mixedシナリオで性能回帰を計測する。
- 完了条件:
  - active tile 非対象領域の計算を確実にスキップできる。
  - 品質を維持したままGPU pass時間が改善する。
- Progress:
  - 2026-03-08: `mpm_active_tiles.wgsl` を追加し、GPU で `active_tile_mask` の clear/mark/compact を毎substep再構築する経路を実装。`clear` / `grid_update` は active tile list + indirect dispatch へ切替し、dense node dispatch を撤去。
  - 2026-03-08: chunk overlay shader を `active_tile_mask` 参照へ更新し、active tile fill/edge 色を `assets/params/overlay.ron` へ追加。`overlay_chunk_active_tiles_water_drop.png` で chunk overlay 上に active tile が重畳表示されることを確認。
  - 2026-03-08: CPU readback snapshot から `active_tile_count` / `inactive_skip_rate` を再計算して HUD と autoverify report に追加。`water_drop_motion.json` では `active_tile_count=17`, `inactive_skip_rate=0.8786`, `runtime_rebuild_count=0`、`sand_water_interaction_drop.json` では `active_tile_count=18`, `inactive_skip_rate=0.9521`, `runtime_rebuild_count=0` を確認。
  - 2026-03-08: 追加依頼として chunk/grid overlay の可読性を調整。下地ピクセル輝度に応じて overlay 色の輝度を反転する post-process 経路へ変更し、active tile は塗りつぶしをやめて tile 内の MPM node grid 表示のみに変更。描画対象がないフレームでも source を先に copy することで黒画面化しないよう修正した。
  - 2026-03-08: 直接の GPU timestamp 計測基盤は未導入のため、現状の性能確認は sparse 化率と scenario 実行完走を主指標としている。pass時間の直接比較が揃うまで WU は `In Progress` のままとする。

### [MPM-CHUNK-04] 地形改変・ロードに伴う chunk SDF 更新

- Status: `In Progress`
- 背景:
  - 初段の静的SDFでは runtime地形改変と整合しない。
- スコープ:
  - 地形改変/ロード差分から chunk SDF dirty を生成し、必要slotのみ更新する。
- Subtasks:
  - [x] 地形改変イベントから chunk SDF dirty 範囲を計算する。
  - [x] chunk SDF 差分更新computeを実装する。
  - [x] セーブロード直後の chunk SDF 再構築フローを実装する。
  - [ ] 改変地形シナリオで表示・SDF・接触の一致を検証する。
- 完了条件:
  - 地形改変後も chunk SDF と描画・接触が整合する。
  - 全面再生成なしで差分更新が成立する。
- Progress:
  - 2026-03-07: CPU の slotごとSDF差分生成を廃止し、`terrain_cell_solid` 差分のみを upload して GPU compute (`mpm_terrain_sdf_update.wgsl`) で `terrain_sdf/terrain_normal` を再計算する経路へ移行。
  - 2026-03-07: `TerrainWorld` の dirty chunk から `collect_slots_for_dirty_chunks` で dirty slot(+近傍) を抽出し、必要 slot のみ更新する差分フローを実装。
  - 2026-03-07: `water_drop_motion` / `sand_water_interaction_drop` / `terrain_edit_solid_tool` の autoverify 実行で回帰なしを確認（いずれも実行終了コード 0、MPM系は `passed=true`）。
  - 2026-03-08: 粒子ツールの `AddParticles` が stale な `upload.particles` を基準に full upload し、GPU readback 後の配置を初期状態へ巻き戻す不具合を確認。編集適用基準を最新 `readback_snapshot` 優先へ修正する。
  - 2026-03-08: `configs/autoverify/water_drop_spawn_append.json` を追加し、`cargo run -q -- --autoverify-config ...` で `passed=true` を確認。artifact では `spawn_ops_applied=1`, `spawned_particles_requested=32`, `total_particles_before_first_spawn=5712`, `gpu_count_water_liquid=5744`, `runtime_rebuild_count=0` を確認し、追加 spawn 後も巻き戻しなく継続落下することを検証。
  - 2026-03-11: 粒子ツールドラッグ中に古い GPU readback が新しい編集結果を上書きし、一時停止や巻き戻りが見える不具合を修正。`MpmGpuUploadRequest` / `MpmReadbackSnapshot` / render readback に `particle_revision` を導入し、world edit は新しい世代の粒子集合を基準に適用、`apply_gpu_readback` は古い世代を破棄するよう更新。`cargo check` / `cargo test --lib` 全通過（40件）。

### [MPM-CHUNK-06] 粒子ツールの GPU 増分編集キュー化

- Status: `Done`
- 背景:
  - 現行の粒子ツール入力は `GpuWorldEditRequest` 自体は増分命令だが、反映経路は CPU 側で `upload.particles` 全体を再構成し、GPU `particle_buf` へ full upload している。
  - この構成では drag 中の高頻度入力で readback/upload 世代の競合が起きやすく、巻き戻り・見かけ上の停止・入力コストの O(N) 化を招く。
- スコープ:
  - 粒子ツールの add/remove を GPU 常駐の編集キューへ直接送る方式へ変更する。
  - world edit 時に CPU 側で全粒子列を再構成しない構成へ移行し、chunk occupancy / residency / mover fallback と整合させる。
- Subtasks:
  - [x] `GpuWorldEditRequest` を render world へ抽出し、GPU が消費する fixed-capacity 編集キュー/カウントバッファを定義する。
  - [x] add/remove を処理する compute pass を追加し、`particle_buf` / particle count / free slot 管理を GPU 側で増分更新する。
  - [x] chunk occupancy / chunk event / `home_chunk_slot_id` 整合を、編集 pass 実行直後の GPU データ基準で更新する。
  - [x] CPU 側 `apply_world_edit_requests` の full upload 経路を撤去し、save/load など明示的な全量置換ケースだけが full upload を使うよう整理する。
  - [x] 常時必要な `particle_count / particle_revision` を full `MpmReadbackSnapshot` から分離し、通常フレームの生存判定と revision 管理を軽量 resource 化する。
  - [x] save/load / runtime rebuild / autoverify のみが full 粒子 readback を要求する on-demand 経路を追加する。
  - [x] `MpmReadbackSnapshot` を撤去し、CPU fallback が必要な箇所は on-demand full readback 完了待ちへ置き換える。
  - [x] 粒子ツールドラッグの自動検証ケースを追加し、連続入力時に巻き戻り・停止・粒子欠落が起きないことを artifact ベースで確認する。
- 完了条件:
  - 粒子ツールの通常 add/remove 操作で CPU 全量 particle upload が発生しない。
  - drag 中に見かけ上の巻き戻りや停止が発生せず、GPU particle count と描画結果が一貫する。
  - save/load / scenario load のような全量置換ケースとの経路分離が明確になっている。
- Progress:
  - 2026-03-11: 粒子ツールの巻き戻り症状を受け、現行の「増分 request + CPU full upload」方式から「GPU 増分編集キュー」方式へ移行する follow-up WU を新設。
  - 2026-03-11: resident slot が既に存在する `AddParticles` について、main world で `GpuWorldEditAddOp` キューを構築し、render world へ抽出して `mpm_world_edit_add.wgsl` で `particle_buf` へ直接追記する GPU 増分追加経路を実装。CPU 側は pending add batch と `particle_revision` を保持し、readback 反映前の一時的な粒子数・後続 remove fallback と整合するよう更新。
  - 2026-03-11: GPU 増分追加後の次フレームで `prepare_gpu_params` が stale な `upload.particles.len()` を particle count として再送し、追加粒子が一瞬表示された後に消える不具合を修正。particle count は full upload 実行フレーム以外では `readback_snapshot + pending_gpu_adds` 基準へ統一した。
  - 2026-03-11: drag 中の一部追加粒子が高速点滅し、粒子数も同数だけ上下する不具合に対し、pending add batch ごとに `expected_total_particle_count` を保持するよう更新。`apply_gpu_readback` は、その revision を名乗っていても期待 count を満たさない incomplete readback を破棄するようにし、pending batch の早すぎる ack/drop を防止した。
  - 2026-03-11: `MpmReadbackSnapshot` の常駐 mirror を撤去する前段として、`MpmParticleReadbackStatus { particle_count, particle_revision }` を追加。通常フレームの `gpu_mpm_active` 判定、particle count 構築、full upload revision 採番をこの軽量 resource へ移し、full 粒子列が必要な経路だけが `MpmReadbackSnapshot` を参照する形へ寄せ始めた。
  - 2026-03-11: `MpmFullParticleReadbackRequest` / `MpmFullParticleReadbackCache` を追加し、particle readback を常時 mirror ではなく on-demand 化。save/load、replay artifact 保存、autoverify、runtime rebuild、CPU fallback world edit は必要 revision の full readback 完了待ちで進む形へ切り替え、`MpmReadbackSnapshot` resource を撤去した。
  - 2026-03-11: CPU fallback world edit の粒子集合選択を「stale な full readback cache 固定」から「`upload` と on-demand readback cache のうち新しい revision を採用」へ修正。pending full upload が古い cache に巻き戻されないようにした。
  - 2026-03-11: `RemoveParticles` と「未resident chunk への追加」はまだ CPU full upload fallback を保持する実装に留めた。WU 完了には GPU remove / chunk occupancy 更新 / fallback 経路整理が引き続き必要。
  - 2026-03-11: `cargo test --lib` 44件通過。`world_edit_add_particles_keeps_newer_upload_over_stale_full_readback_cache`、`apply_gpu_readback_caches_requested_payload_and_clears_request`、`consume_world_edit_add_ack_drops_acked_batches` などを追加し、snapshot 撤去後の revision 選択・on-demand readback・GPU add ack を固定。
  - 2026-03-11: `cargo run -q -- --autoverify-config configs/autoverify/water_drop_spawn_append.json` を再実行し、`artifacts/autoverify/water_drop_spawn_append.json` で `passed=true`, `spawn_ops_applied=1`, `spawned_particles_requested=32`, `gpu_count_water_liquid=5744`, `runtime_rebuild_count=0`, `runtime_rebuild_waiting_readback_count=0`, `invalid_slot_access_count=0` を確認。
  - 2026-03-12: `run_mpm_autoverify` に spawn 後の総粒子数整合チェックを追加し、`spawned_particles_requested` と `total_particles_before_first_spawn` から導く expected total と最終 `gpu_stats.total()` が一致しない場合は fail するよう更新。`configs/autoverify/water_drop_spawn_drag.json` を追加し、6 連続 spawn の drag 相当ケースを検証。`artifacts/autoverify/water_drop_spawn_drag.json` で `passed=true`, `spawn_ops_applied=6`, `spawned_particles_requested=72`, `gpu_count_water_liquid=5784`, `runtime_rebuild_count=0`, `invalid_slot_access_count=0`, `failed_assertions=[]` を確認。
  - 2026-03-12: `GpuWorldEditRemoveQueueRequest` と `mpm_world_edit_remove.wgsl` を追加し、remove は on-demand full readback で確定した particle id 群を GPU compaction pass へ送る構成へ更新。render world は scratch particle buffer へ survivors を詰め直し、`particle_count` / `particle_revision` を増分更新する。
  - 2026-03-12: nonresident chunk への `AddParticles` でも CPU full particle upload へ戻さず、CPU 側で slot pool を初期化/拡張して chunk meta・terrain slot diffs を先に upload し、そのまま GPU add queue へ流す経路を追加。world edit pass 後は paused 中でも chunk meta finalize / chunk-event extract を回して occupancy と residency を更新する。
  - 2026-03-12: `run_mpm_autoverify` に `remove_ops` と `removed_particles_requested` を追加し、`spawned - removed` 後の最終粒子総数を自動検証できるよう拡張。`configs/autoverify/water_drop_spawn_remove_drag.json` は readback 待ちを挟んでも安定に remove が成立するよう、落下後の安定水塊を対象にした 2 回 remove ケースへ調整。`artifacts/autoverify/water_drop_spawn_remove_drag.json` で `passed=true`, `spawn_ops_applied=6`, `spawned_particles_requested=72`, `remove_ops_applied=2`, `removed_particles_requested=24`, `gpu_count_water_liquid=5760`, `runtime_rebuild_count=0`, `invalid_slot_access_count=0`, `failed_assertions=[]` を確認。
  - 2026-03-12: `apply_world_edit_requests` の最後の例外 full-upload fallback を整理。明示的な full replacement が pending なときだけ `upload.particles` を直接拡張し、それ以外の `residency 未初期化 + GPU 粒子あり` ケースは on-demand full readback から residency / mover upload / chunk meta upload だけを再構築して、そのまま GPU add queue へ流すよう更新。`prepare_terrain_upload` は world edit が先に積んだ chunk/terrain slot upload を消さずに引き継ぐよう修正し、unit test で固定した。
  - 2026-03-12: drag spawn 中に新規粒子が入力方向と逆側へ数セルずれて見える不具合に対し、`readback_status.particle_count` と `pending_gpu_adds` を同時加算していた particle count 二重化を修正。GPU add queue 反映済みの粒子数は `readback_status` 単体を effective total とみなし、`prepare_gpu_params` / `prepare_particle_upload` / pending batch expected count を一致させた。`cargo check` / `cargo test --lib` / `configs/autoverify/water_drop_spawn_drag.json` 再確認で `passed=true`, `spawn_ops_applied=6`, `spawned_particles_requested=72`, `gpu_count_water_liquid=5784`, `runtime_rebuild_count=0`, `invalid_slot_access_count=0` を確認。
  - 2026-03-12: chunk halo / frontier 拡張が初期 resident set から進みにくくなっていた退行を修正。GPU の chunk occupancy 集計が `home_chunk_slot_id` 固定で outer halo 進入を 1 フレーム以上遅れて観測していたため、`mpm_chunk_meta_update.wgsl` の occupancy accumulate を「現在位置から隣接 slot へ再解決」する方式へ更新し、frontier request と active chunk 更新が halo 進入フレームで出るようにした。`cargo check` / `cargo test --lib` / `configs/autoverify/water_drop_motion.json` / `configs/autoverify/water_drop_spawn_drag.json` で再確認し、`artifacts/autoverify/water_drop_motion.json` は `passed=true`, `terrain_penetration_ratio=0.04744`, `runtime_rebuild_count=0` を確認。
  - 2026-03-12: 遠距離 spawn 後に overlay / SDF pass が最初の resident chunk 数のまま止まる退行を修正。render world の `MpmGpuBuffers.active_chunk_count` が full chunk upload 時しか更新されず、`chunk_meta_diffs` だけで resident slot が増えたフレームでは古い chunk 数を保持していたため、`prepare_gpu_uploads` で diff upload 時も `params.resident_chunk_count` を反映するよう更新した。あわせて `MpmRenderDiagnostics` を追加し、autoverify artifact へ render 側 active chunk 数を出力するようにした。`configs/autoverify/default_world_far_spawn_chunk_growth.json` を追加し、default world で遠距離 2 点 spawn の後に `resident_chunk_count=29`, `render_active_chunk_count=29`, `gpu_count_water_liquid=24`, `runtime_rebuild_count=0` を確認。`configs/autoverify/water_drop_motion.json` も再実行し、`render_active_chunk_count=30`, `runtime_rebuild_count=0`, `invalid_slot_access_count=0` を確認。
  - 2026-03-12: default world で 6x6 相当の初期 active chunk から先へ広がらない退行に対し、新規 `AddParticles` 用に allocate した target chunk/halo を GPU add apply 前から resident として `chunk_meta` へ反映する補助を追加。さらに `upload_chunks/upload_chunk_diffs/upload_terrain_cell_slot_diffs` を one-shot `*_frame` flag に分離し、render world が stale な full chunk upload flag を握り続けて後続 diff 更新を潰す状態を解消した。`world_edit_add_particles_marks_target_halo_slots_resident_before_gpu_apply` を追加し、`cargo test --lib` 47件 / `cargo check` を通過。`configs/autoverify/default_world_far_spawn_chunk_growth.json` は 5 点遠距離 spawn に拡張して `resident_chunk_count=50`, `render_active_chunk_count=50`, `spawned_particles_requested=60`, `runtime_rebuild_count=0` を確認し、6x6 超えの resident/render active chunk 拡大を artifact で固定した。
  - 2026-03-12: 上記 refactor 後も default world の遠距離 spawn 実粒子が原点側 1 chunk へ潰れる退行が残っていたため、GPU world-edit add の pending batch を入力のないフレームでも oldest 順に再送するよう更新し、render world 側では retry で `particle_count` / `particle_revision` を二重更新しないよう整理した。あわせて render world の chunk/terrain upload 判定が main-world 専用 `*_frame` flag を参照していた不整合を修正し、さらに `apply_chunk_event_readback` / `apply_mover_readback` が world-edit 側で先に積んだ `chunk_meta` upload を上書きしないよう merge 型 staging に変更した。`world_edit_add_requeues_oldest_pending_batch_without_new_input` を追加し、`cargo test --lib` 48件 / `cargo check` を通過。`configs/autoverify/default_world_far_spawn_chunk_growth.json` を再確認し、`artifacts/autoverify/default_world_far_spawn_chunk_growth.json` で `resident_chunk_count=45`, `render_active_chunk_count=45`, `spawned_particles_requested=60`, `readback_unique_chunk_count=5`, `readback_min_x=-39.96875`, `readback_max_x=40.21875`, `runtime_rebuild_count=0` を確認して、default world の「1x1 chunk に潰れる」症状が artifact 上で解消した。
  - 2026-03-12: さらに render world で terrain slot update list だけが main-world 専用 `upload_terrain_cell_slot_diffs_frame` を参照しており、`terrain_sdf_update` が走らず `terrain_node_solid` が更新されない退行を修正した。これにより `play` しても時間が進まない症状は解消し、`configs/autoverify/water_drop_motion.json` は `mean_drop=8.437712`, `gpu_max_speed_mps=8.655025`, `resident_chunk_count=36`, `render_active_chunk_count=36`, `runtime_rebuild_count=0` まで回復した。一方で最終 `terrain_penetration_ratio=0.05217087` が閾値 `0.05` をわずかに超えており、`passed=false` は現在この軽微な penetration 超過のみが原因になっている。
  - 2026-03-12: default world で中心付近の数 chunk しか粒子が描画されない退行に対し、`water_dot_gpu` の密度グリッドを `WORLD_MIN/MAX_CHUNK_*` 固定範囲から world-locked な大域 window へ切替した。window origin は 1 chunk 単位で snap し、カメラ追従ではなくワールド基準で保持したまま、遠距離 spawn 粒子も preprocess 範囲へ入るよう修正。あわせて極端な zoom-out でも `dot_count <= 65535 * 64` となるよう dot resolution を段階的に粗くし、`wgpu` の dispatch 上限超過 panic を防止。`render::water_dot_gpu` unit test 3件、`cargo check`、`cargo test --lib`、`cargo run -q -- --autoverify-config configs/autoverify/default_world_far_spawn_chunk_growth.json` を通し、`artifacts/autoverify/default_world_far_spawn_chunk_growth.json` で `passed=true`, `readback_min_x=-39.96875`, `readback_max_x=40.21875`, `readback_unique_chunk_count=5`, `render_active_chunk_count=45` を確認。
  - 2026-03-12: user approval に基づき `water_drop` scenario spec の `max_penetration_rate` を `0.05 -> 0.10` へ緩和した。再検証で `artifacts/autoverify/water_drop_motion.json` は `passed=true`, `terrain_penetration_ratio=0.05217087`, `mean_drop=8.437712`, `gpu_max_speed_mps=8.655025`, `runtime_rebuild_count=0` を確認し、停止退行解消後に残っていた threshold-only fail を解消した。

### [MPM-CHUNK-05] 安定化・運用仕上げ（容量、監視、フェイルセーフ）

- Status: `Planned`
- 背景:
  - 本番運用では slot容量上限、frontier変化スパイク、exceptional mover急増、診断不足が主要リスクになる。
- スコープ:
  - chunk構造の運用安全性を担保し、異常時の挙動を定義する。
- Subtasks:
  - [ ] `max_resident_slots` 超過時の方針（drop/expand/fallback）を実装する。
  - [ ] 診断HUD/artifactに slot使用率、割当失敗回数、fallback回数を追加する。
  - [ ] 長時間ランと大移動ケースの soak test を整備する。
- 完了条件:
  - chunk構造が長時間実行で安定し、異常時の挙動が定義どおりになる。

### [PARAM-HEX-01] RONカラー記法をhex形式へ拡張

- Status: `Done`
- 背景:
  - `overlay.ron` や `palette.ron` の色指定が `r/g/b/a` 記法のみのため、VSCode のカラーピッカー入力が使いにくい。
- スコープ:
  - `#RRGGBB` / `#RRGGBBAA` を受け付けるデシリアライズ互換を `params` 側で実装する。
  - 既存 `r/g/b/a` フォーマットの後方互換を維持する。
  - `interface/overlay/palette` の対象 RON を hex 記法へ移行する。
- Subtasks:
  - [x] `src/params` に共通 hex パース/デシリアライズヘルパーを追加する。
  - [x] `UiColor`, `UiColor8`, `OverlayColor`, `PaletteColor` を hex 対応へ更新する。
  - [x] `assets/params/interface.ron`, `assets/params/overlay.ron`, `assets/params/palette.ron` を hex 形式へ更新する。
  - [x] パース回帰を抑える単体テストを追加する。
- 完了条件:
  - 3種類（`r/g/b`, `r/g/b/a`, hex）を読み分けず読み込める。
  - 既存シナリオ検証が壊れず、`cargo check`/`cargo test` が通る。
- 進捗:
  - 2026-03-07: `hex` 文字列の色指定を受けるパーサーを追加し、`palette/interface/overlay` へ反映。
  - 2026-03-07: `assets/params/*.ron` の該当色値を hex 形式へ移行。
  - 2026-03-07: 追加テストを含む状態で `cargo check` / `cargo test` を全件通過。

## Done (Recent)

### [UI-PROFILE-01] 左上HUDのCPU/GPUプロファイル棒グラフ可視化

- Status: `Done`
- 背景:
  - 左上HUDには FPS と統計テキストはあるが、CPU/GPU のどの処理がフレーム時間を支配しているかを実行中に判断できない。
  - `active tile` や render/compute pass の最適化を進めるには、CPU span と GPU scope を同一 HUD 上で比較できる常時可視化が必要。
- スコープ:
  - FPS 表示の直下、`Sim: Running/Paused` 表示の上に CPU / GPU それぞれの水平積み上げ棒グラフを追加する。
  - CPU は Bevy trace span を subscribe して update 間隔ごとに集計し、GPU は `wgpu-profiler` の scope 単位結果を集計する。
  - 棒グラフはカテゴリ色 + 詳細差分色で描画し、hover で名前と「実時間1秒あたりの処理時間」を表示する。
  - 更新頻度、サイズ、色、最小表示閾値などの runtime tuning 値は `assets/params/interface.ron` へ追加する。
- Subtasks:
  - [x] Work Unit を追加し、CPU/GPU profiler 可視化の実装単位を定義する。
  - [x] CPU trace subscriber を追加し、span enter/exit から update 間隔内の合計時間を集計する。
  - [x] GPU profiler を render graph / draw pass / compute pass へ統合し、scope 結果を main world へ反映する。
  - [x] 左上 HUD に CPU / GPU 棒グラフ UI、hover tooltip、動的スケール計算を実装する。
  - [x] `interface.ron` に profiler 可視化パラメータを追加し、hot reload 可能にする。
  - [x] `cargo check` / `cargo test --lib` と、必要な runtime artifact で可視化挙動を確認する。
  - [x] hover tooltip の runtime 確認を完了する。
- 完了条件:
  - CPU / GPU の各グラフが 1 秒あたり 10 回（または設定値）で更新され、直近区間の積算時間を表示する。
  - 主要 scope/span が色分けされ、hover で span/scope 名と時間を確認できる。
  - グラフ幅スケールが実フレームレートに追従し、余裕があるときは最大幅まで伸びる。
- 進捗:
  - 2026-03-12: Work Unit を追加し、CPU は Bevy trace subscriber、GPU は `wgpu-profiler`、表示は左上 HUD 埋め込みの棒グラフ UI で進める方針を確定。
  - 2026-03-12: `bevy/trace` を常時有効化し、`physics::profiler` に custom tracing layer + shared accumulator + render-world `wgpu-profiler` 統合を追加。CPU は exclusive span time、GPU は query tree の exclusive self time を集計する方式で snapshot 化。
  - 2026-03-12: `assets/params/interface.ron` / `params::interface` に profiler 設定群（更新頻度、棒グラフ寸法、閾値、カテゴリ色）を追加。左上 HUD に CPU/GPU 2 行の積み上げ棒グラフを挿入し、segment hover tooltip を実装。
  - 2026-03-12: `wgpu-profiler` は `wgpu 27` と揃う `0.25.0` へ pin。`cargo check` と `cargo test --lib` 51件通過。
  - 2026-03-12: profiler tooltip の hit test を UI 物理座標ベースから Bevy UI の論理座標ベースへ修正し、pause 中は profiler 集計窓を更新せず reset するよう変更。`cargo check` / `cargo test --lib` / `configs/autoverify/profile_hud_water_drop.json` を再実行。
  - 2026-03-12: `configs/autoverify/profile_hud_water_drop.json` を追加し、`cargo run -q -- --autoverify-config configs/autoverify/profile_hud_water_drop.json` を実行。`artifacts/autoverify/profile_hud_water_drop.png` で HUD 上の CPU/GPU 棒グラフ描画を確認。
  - 2026-03-12: profiler hover の当たり判定を Bevy UI の実装に合わせて `physical_cursor_position()` ベースへ戻し、hover 中 segment を白でハイライトする再着色経路を追加。`cargo check` / `cargo test --lib` / `configs/autoverify/profile_hud_water_drop.json` を再実行。
  - 2026-03-12: profiler 色カテゴリを `physics` / `render` / `others` の3系統へ整理。色設定は `assets/params/interface.ron` の `profiler.colors` に集約し、`terrain` / `water` / `overlay` / `ui` は表示上 `render` 色へ正規化するよう更新。
  - 2026-03-12: profiler tooltip に出すカテゴリ名も snapshot 生成時に `physics` / `render` / `others` へ正規化し、`water/...` などの旧カテゴリ名が HUD に出ないよう更新。`cargo check` / `cargo test --lib` / `configs/autoverify/profile_hud_water_drop.json` を再実行。
  - 2026-03-12: profiler bar 内の segment 順序を時間順から固定順へ変更。`physics -> render -> others` のカテゴリ順、その中は detail 名の辞書順で snapshot を整列するよう更新。`cargo check` / `cargo test --lib` を再実行。
  - 2026-03-12: `assets/params/interface.ron` の `profiler.max_segments_per_lane` を増やして表示可能 segment 数を拡張し、`others/remainder` への過剰な集約を抑える運用へ調整。
  - 2026-03-12: hover tooltip の runtime 確認が完了し、Work Unit を `Done` へ移動。

### [MPM-OVERLAY-01] MPMセル質量オーバーレイ追加

- Status: `Done`
- 背景:
  - 既存の Chunk/Grid Overlay と SDF Overlay では、resident chunk 内で MPM グリッドへ集約された質量分布を直接確認できない。
  - 境界近傍や world edit 後の質量偏りを診断するには、GPU 常駐 `grid_buf` からセル単位の質量を即座に見られる可視化が必要。
- スコープ:
  - resident chunk に対して、MPM グリッド値からセル質量を算出して GPU overlay として描画する。
  - overlay 有効時にカラーバーを UI 表示し、現在のカラーマップ上限が分かるようにする。
  - screenshot autoverify から表示状態を切り替えられるようにする。
- Subtasks:
  - [x] resident chunk の `grid_buf` を参照する GPU mass overlay pass を追加する。
  - [x] Mass Overlay トグルボタンとカラーバー UI を追加する。
  - [x] HUD / screenshot autoverify override を mass overlay 対応へ拡張する。
  - [x] `water_drop` screenshot artifact で表示成立を確認する。
- 完了条件:
  - resident chunk のみを対象にセル質量が描画される。
  - overlay 有効時にカラーバーが表示され、色レンジ上限が UI から確認できる。
  - screenshot artifact で描画成立を再現できる。
- 進捗:
  - 2026-03-12: `mass_overlay_gpu.wgsl` と `src/overlay/mass.rs` を追加し、resident chunk の `grid_buf.water_mass + granular_mass` から 2x2 node 平均でセル質量を算出する GPU pass を実装。`overlay.ron` に色とレンジ係数を追加し、Mass Overlay ボタン / カラーバー / HUD 表示 / screenshot override を接続した。
  - 2026-03-12: `configs/autoverify/overlay_mass_water_drop.json` を追加し、`cargo run -q -- --autoverify-config configs/autoverify/overlay_mass_water_drop.json` で `artifacts/autoverify/overlay_mass_water_drop.png` / `.json` を生成。artifact 上で `Mass Overlay: ON` とカラーバー表示、resident chunk 内のセル質量可視化成立を確認した。

### [MPM-PHYS-WATER-BOUNDARY-01] 静的地形の壁面境界修正（solid node除外 + 粒子フェイルセーフ）

- Status: `Done`
- 背景:
  - 現行の静的地形境界は、壁内 node を含む stencil と SDF 近傍の速度回復に依存しており、壁際水粒子が早期反発・不自然な剪断・横加速を示す。
  - `water_drop` でも静止近傍で壁際盛り上がりや自由表面段差の残留が観測されており、通常系の壁接触を node 外向き回復で扱う設計を見直す必要があった。
- スコープ:
  - 静的セル地形について、MPM node を `solid/fluid` の 2 値で分類し、P2G/G2P/APIC/境界近傍内部力を fluid-side stencil 再正規化へ切り替える。
  - 境界 node の接触は法線速度射影 + クーロン摩擦で扱い、壁内 node への一律 recovery 速度注入を通常系から外す。
  - 異常系フェイルセーフとして、移流後に SDF で侵入粒子を射影・速度補正する粒子レベル補正を追加する。
  - 将来の剛体連成向け設計指針として、連続境界量（cut fraction / face fraction / SDF）への一般化と、接触インパルスの対称更新による運動量保存要件を整理する。
- Subtasks:
  - [x] `physics.md` 方針に沿って、static terrain 用の `solid node mask + stencil reweight` の transfer 設計を WGSL/Rust レイアウトへ落とす。
  - [x] 壁近傍で `solid node` を transfer から除外し、P2G/G2P/APIC が一貫した重み再正規化を使うよう実装する。
  - [x] `grid_update` の静的地形境界から常時 recovery 速度注入を外し、boundary node の法線射影 + 摩擦のみへ更新する。
  - [x] 移流後の粒子 SDF フェイルセーフ（位置射影 + 法線内向き速度除去）を実装する。
  - [x] 壁際単セル / 1セル離隔 / `water_drop` 静止面の検証artifactを追加し、壁沿い落下・壁際盛り上がり・自由表面段差の改善を確認する。
  - [x] Design note として、剛体 2-way coupling 時に連続境界量と反作用インパルス対称更新が必要になる点を `tasks.md` 上でフォローアップ管理する。
- 完了条件:
  - [x] 壁際単セルテストで、水粒子が不自然な横反発や剪断なく壁沿いに落下する。
  - [x] 壁から 1 セル離した粒子配置で、通常系の wall-induced side push が発生しない。
  - [x] `water_drop` の静止近傍で壁際盛り上がりと表面段差が改善し、既存の侵入率・完走条件を維持する。
- 進捗（要約）:
  - 2026-03-09: GPU terrain SDF 更新パスに `terrain_node_solid` を追加し、P2G/G2P/APIC transfer を fluid-side stencil 再正規化へ更新。`mpm_grid_update` から static terrain 向け recovery 速度注入を削除し、G2P 最終段に粒子 SDF フェイルセーフ（位置射影 + 法線内向き速度除去）を追加。
  - 2026-03-09: Chunk/Grid overlay が `terrain_node_solid` を参照するよう更新し、solid node の grid 線を描かない表示へ変更。`artifacts/autoverify/water_boundary_wall_single_cell.png`, `artifacts/autoverify/water_boundary_wall_offset_one_cell.png`, `artifacts/autoverify/water_boundary_water_drop_surface.png` で壁際ケースと静止面を確認。
  - 2026-03-09: 回帰確認として `configs/autoverify/water_drop_motion.json` を再実行し、`artifacts/autoverify/water_drop_motion.json` で `passed=true`, `terrain_penetration_ratio=0.03326`, `runtime_rebuild_count=0` を確認。
  - 2026-03-09: 壁近傍 APIC の不整合に対して、P2G 内部力を `∇ŵ`（fluid-side 再正規化勾配）へ更新し、G2P は truncated support の一次・二次モーメントを使う centered APIC 復元へ変更。`vp=Σŵv_i` の一次モーメントずれは `mean_dx` 補正で打ち消し、欠けた support では covariance 最小固有値から局所的に APIC を減衰させるよう更新。再検証で `artifacts/autoverify/water_drop_motion.json` は `passed=true`, `terrain_penetration_ratio=0.03361`, `max_speed_mps=6.60567`, `runtime_rebuild_count=0` を確認。
  - 2026-03-10: `mpm_grid_update` の terrain 境界 Coulomb 投影に実装バグを修正。従来は `mu=0` でも slip 分岐で `out_v -= vt_vec` となり boundary node の接線速度を丸ごと 0 にしていたため、壁際粒子が stencil 経由で偽の減速を拾っていた。修正後は slip 時の減衰量を `stick_limit / |v_t|` のみ差し引く形へ更新し、`cargo check` / `cargo test --lib` / `configs/autoverify/water_drop_motion.json` 再実行で `passed=true`, `terrain_penetration_ratio=0.03361`, `runtime_rebuild_count=0` を確認。
  - 2026-03-10: 水の接地後リバウンド調整用に `runtime.boundary_velocity_normal_projection_scale` を追加。node-level 非貫通で内向き法線速度をどの程度除去するかを `physics.ron` から 0.0–1.0 で調整可能にし、既定値 1.0 で `cargo check` / `cargo test --lib` / `configs/autoverify/water_drop_motion.json` を再確認して `passed=true`, `terrain_penetration_ratio=0.02258`, `runtime_rebuild_count=0` を確認。
  - 2026-03-09: Design follow-up を本 Work Unit 上で明示。剛体 2-way coupling へ拡張する際は、2 値 `solid node mask` のままでは接触が粗くなるため、連続境界量（cut fraction / face fraction / SDF）と、node 射影/粒子フェイルセーフで除去した法線インパルスの反作用を剛体側へ対称更新する設計が必要。

- `DESIGN-CHUNK-PLAN-01` / `Done` / `chunks.md` 準拠で `design.md` 更新、`tasks.md` コンパクション、chunk実装WU分解を完了 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `PARAM-01` / `Done` / パラメータ資産化と hot reload 基盤を実装 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `MPM-PHYS-WATER-01` / `Done` / `physics.md` 改訂に合わせた水物性GPU実装を反映 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`

## Closed Work Units (Deferred / Superseded)

- `MPM-GPU-03` / `Done` / 完了条件達成（GPU地形描画成立・旧CPU描画削除済み）/ `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `MPM-GPU-04` / `Superseded` / chunk構造前提のSDF経路として `[MPM-CHUNK-01..04]` に再編 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `REND-01` / `Superseded` / GPU Near/Far 2層キャッシュ方式 `[REND-GPU-01]` に置換 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `WGEN-02` / `Superseded` / GPU Near/Far 2層キャッシュ方式 `[REND-GPU-01]` に置換 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `MPM-WATER-02` / `Deferred` / 単一grid GPU本流への再編で内容を上位WUへ統合 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `MPM-WATER-03A` / `Deferred` / chunk SDF設計へ再編 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `MPM-WATER-03` / `Deferred` / chunk SDF境界処理へ再編 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `MPM-WATER-03B` / `Deferred` / 境界補正仕様は chunk SDF側に統合して再定義 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `MPM-WATER-05` / `Deferred` / シナリオ検証は chunk導入後に閾値再設定 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `MPM-DEFER-01` / `Deferred` / 粉体系詳細は `MPM-PHYS-GRANULAR-01` に集約 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `WGEN-01` / `Deferred` / 地形生成系は render/chunk経路確定後に再設計 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
- `WGEN-03` / `Deferred` / 連続LODサンプラは地形キャッシュ完成後に再開 / `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`

## Archive Index

- `2026-03-07 chunk replan snapshot`:
  - `docs/tasks_archive/2026-03-07-before-chunk-structuring.md`
  - 退避対象: 旧 `tasks.md` 全文（進捗履歴・旧WU詳細を含む）
- `2026-03-02 full snapshot`:
  - `docs/tasks_archive/2026-03-02-full-before-compaction.md`
  - 退避対象: 旧コンパクション前タスク全文

## Approval-Gated Items

- 物理統合テスト（headless scenario tests）のケース追加・閾値変更・ベースライン更新は、実装前にユーザー承認を取る。

## Open Design Feedback

- `MPM-CHUNK-01` 着手前に、`max_resident_slots` の初期値と上限超過時ポリシー（停止/拡張/fallback）を明確化する必要がある。
- 静的residency段階で粒子が resident 範囲外へ出た場合の仕様（クランプ/消去/一時停止）を定義する必要がある。
- chunk SDF の離散化仕様（node中心かcell中心か、normal再構成法、境界しきい値）を `design.md` へ固定する必要がある。
- Render 側地形キャッシュ（Near/override）と MPM用 chunk SDF の更新順序を明文化し、同一フレーム整合を保証する必要がある。
- fallback 切替閾値を `mover_count` 単体ではなく、`frontier change` と `exceptional mover` 指標ベースで再定義する必要がある。
- `MPM-CHUNK-02` 現段は `chunk_origin/chunk_dims` に基づく固定 slot pool 上で resident on/off と diff upload を実装しており、free list を使った任意 slot 再配置（完全疎 slot addressing）は未導入。P2G/Grid/G2P を `home_chunk_slot_id + neighbor_slot_id` 参照へ完全移行する追加WUが必要。
