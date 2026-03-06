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
  - [ ] P2G/Grid/G2P の node 参照を chunk slot addressing 経路へ切替する。
  - [x] SDF/normal 参照を chunk SDF バッファ経路へ切替し、境界補正を成立させる。
  - [x] `water_drop` autoverify で `passed=true` を達成する。
  - [x] 検証artifactに `resident_chunk_count`, `invalid_slot_access_count`, `chunk_sdf_samples` を出力する。
  - [x] Chunk Overlay / Physics Area Overlay を統合し、`chunk_meta_buf` を使うGPU描画へ移行する。
- 完了条件:
  - CPUとの chunk 差分同期なしで `water_drop` が最後まで完走し、既存判定を満たす。
  - 実行中に invalid slot 参照が発生しない。
  - chunk SDF 経由の境界反映で侵入率が閾値内に収まる。
- 進捗:
  - 2026-03-07: GPU側に `chunk_meta_buf` を追加し、静的residencyの one-shot 構築（起動時 / 粒子再upload時）を導入。resident chunk bbox から `grid_origin/grid_dims` を再計算して MPM 実行領域を world-dense 固定から切替。
  - 2026-03-07: 初期地形から chunk SDF/normal を one-shot 生成してGPU uploadする経路を実装。`grid_update` と統計passは同バッファを参照。
  - 2026-03-07: autoverify report へ `resident_chunk_count`, `invalid_slot_access_count`, `chunk_sdf_samples` を追加。`configs/autoverify/water_drop_motion.json` 実行で `passed=true`, `invalid_slot_access_count=0` を確認。
  - 2026-03-07: Physics Area Overlay を Chunk Overlay に統合し、GPU側 `chunk_meta_buf` + `params_buf` を参照する描画パスへ移行。`water_drop` スクリーンショット検証で `resident:64` のオーバーレイ表示を確認（`artifacts/autoverify/overlay_chunk_physics_water_drop.png`）。

### [MPM-CHUNK-02] movers抽出 + CPU residency更新 + chunk meta差分反映

- Status: `Planned`
- 背景:
  - `MPM-CHUNK-01` は静的residencyであり、chunk跨ぎ移動が増えるケースでは成立しない。
- スコープ:
  - GPU movers抽出、CPU側 `chunk_coord -> slot_id` 更新、GPU差分反映を incremental に導入する。
- Subtasks:
  - [ ] movers append/readback/result反映バッファを実装する。
  - [ ] `occupied_particle_count` / `halo_ref_count` による slot割当・解放を実装する。
  - [ ] `neighbor_slot_id` の差分再計算と GPU diff upload を実装する。
  - [ ] mover高負荷時の full rebuild fallback を実装する。
  - [ ] chunk境界跨ぎシナリオで回帰検証する。
- 完了条件:
  - 粒子の chunk移動時に `home_chunk_slot_id` と chunk table が整合し続ける。
  - mover率が高いケースでも破綻せず、fallbackで収束する。

### [MPM-CHUNK-03] active tile再構築と sparse実行最適化

- Status: `Planned`
- 背景:
  - chunk構造導入後の性能は active tile 最適化の有無に強く依存する。
- スコープ:
  - resident chunk 内で active tile mask を毎step再構築し、clear/grid update を sparse 化する。
- Subtasks:
  - [ ] chunkごとの `active_tile_mask` 初期化・mark（neighbor含む）を実装する。
  - [ ] clear/grid update を active tile 限定 dispatch へ切替する。
  - [ ] `active_tile_count` / `inactive_skip_rate` を計測する。
  - [ ] `water_drop` と mixedシナリオで性能回帰を計測する。
- 完了条件:
  - active tile 非対象領域の計算を確実にスキップできる。
  - 品質を維持したままGPU pass時間が改善する。

### [MPM-CHUNK-04] 地形改変・ロードに伴う chunk SDF 更新

- Status: `Planned`
- 背景:
  - 初段の静的SDFでは runtime地形改変と整合しない。
- スコープ:
  - 地形改変/ロード差分から chunk SDF dirty を生成し、必要slotのみ更新する。
- Subtasks:
  - [ ] 地形改変イベントから chunk SDF dirty 範囲を計算する。
  - [ ] chunk SDF 差分更新computeを実装する。
  - [ ] セーブロード直後の chunk SDF 再構築フローを実装する。
  - [ ] 改変地形シナリオで表示・SDF・接触の一致を検証する。
- 完了条件:
  - 地形改変後も chunk SDF と描画・接触が整合する。
  - 全面再生成なしで差分更新が成立する。

### [MPM-CHUNK-05] 安定化・運用仕上げ（容量、監視、フェイルセーフ）

- Status: `Planned`
- 背景:
  - 本番運用では slot容量上限、急激なmover増加、診断不足が主要リスクになる。
- スコープ:
  - chunk構造の運用安全性を担保し、異常時の挙動を定義する。
- Subtasks:
  - [ ] `max_resident_slots` 超過時の方針（drop/expand/fallback）を実装する。
  - [ ] 診断HUD/artifactに slot使用率、割当失敗回数、fallback回数を追加する。
  - [ ] 長時間ランと大移動ケースの soak test を整備する。
- 完了条件:
  - chunk構造が長時間実行で安定し、異常時の挙動が定義どおりになる。

## Done (Recent)

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
- mover率が高いケースの fallback 切替閾値（`mover_count` ベース）を事前に決める必要がある。
- `chunks.md` 目標では `CHUNK_NODE_SIZE=32`（`NODE_SPACING_M=0.125`）を前提としているが、現実装の solver 格子は `h=0.25` のため `MPM-CHUNK-01` では `chunk_node_dim=16` を採用している。`MPM-CHUNK-02` 着手前に解像度移行方針（`h` 引き下げ時の安定条件と閾値再設定）を決める必要がある。
