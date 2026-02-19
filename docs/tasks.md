# Tasks

## Todo
- (No open tasks)

## Done
- (Keep completed tasks here; do not delete history)
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

## Design Feedback (from Impl sessions)
- (If implementation reveals required design changes, record requests here)
