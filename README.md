# particles

Bevy 0.18 ベースの 2D 物理シミュレーション実験プロジェクトです。

## 実行方法

### 通常実行

```bash
cargo run
```

## ファイル監視で自動コンパイル実行

`cargo-watch` を使って、ファイル変更時に自動で再コンパイルして再実行します。

```bash
cargo install cargo-watch
cargo watch -x run
```

- Rust コード変更: 自動で再コンパイルして再起動
- アセット変更: `bevy/file_watcher` により自動再読込

## Tracy でプロファイリング

Bevy の `trace` と `LogPlugin` の `custom_layer` で Tracy を有効化できます。
物理ステップ内は `physics::*` スパンで分割して計測されます（`predict_positions` / `solve_density_constraints` / `shape_matching` など）。
このプロジェクトの Tracy は `ondemand` 構成なので、Profiler 未接続時の負荷を抑えます。

### Tracy プロファイル（通常実行）

```bash
cargo run --release --features tracy
```

### Tracy プロファイル（ファイル監視）

```bash
cargo watch -x "run --release --features tracy"
```

- もし表示が不安定な場合は `cargo run --no-default-features --features tracy` で `dynamic_linking` を外して試してください

## Headless 物理シナリオテスト

固定dtの手動stepで、物理シナリオ統合テストをheadlessで実行できます。

```bash
cargo test --test physics_scenarios -- --nocapture
```

実行結果のartifactは以下に出力されます。

- `artifacts/tests/<scenario>/<run_id>/final_state.json`
- `artifacts/tests/<scenario>/<run_id>/metrics.json`

`metrics.json` の閾値判定に失敗した場合は、テストログにscenario名とartifactパスが表示されます。

## 簡易再生モード（Scenario Replay）

右上の `Load` ダイアログに、通常セーブ (`Save Slots`) とテストケース (`Test Cases`) が並びます。  
`Test Cases` から選んで `Load` すると、テストワールドとして読み込まれます。

再生/停止は通常のゲーム操作（`Space` など）をそのまま使用します。

テストワールド時は、画面右側にアサーション評価（`OK/NG`、期待値、実測値）が表示されます。

再生モード中も artifact は以下へ保存されます。

- `artifacts/tests/<scenario>/<run_id>/final_state.json`
- `artifacts/tests/<scenario>/<run_id>/metrics.json`
