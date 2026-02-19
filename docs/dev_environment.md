# Bevy 開発環境メモ（Impl）

## 1. 構成
- Rust: stable（`rustc 1.92.0` で確認）
- Bevy: `0.18.0`
- 有効化 feature:
  - `dynamic_linking`
  - `file_watcher`
- `assets/` ディレクトリを配置済み（ファイル監視の有効化条件）

`Cargo.toml` で Bevy feature を固定しているため、通常の `cargo run` で開発構成になる。

## 2. 起動
```bash
cargo run
```

## 3. 自動再コンパイル + 自動再ロード
`cargo-watch` を使う。

```bash
cargo install cargo-watch
cargo watch -x run
```

- Rust コード変更: 自動で再コンパイルし、アプリを再起動
- アセット変更: `bevy/file_watcher` により変更検知して再読込

## 4. ホットリロード方針（初期運用）
- 採用:
  - `file_watcher`（アセット再読込）
  - `dynamic_linking`（開発時ビルド体験改善）
- 不採用:
  - `hotpatching`（コード hotpatch は初期運用に入れない）

## 5. 検証コマンド
有効 feature と不採用 feature を確認する。

```bash
cargo tree -e features | rg "bevy/(dynamic_linking|file_watcher|hotpatching)"
```

期待値:
- `bevy/dynamic_linking` が出る
- `bevy/file_watcher` が出る
- `bevy/hotpatching` は出ない

起動確認（GPU があるローカル環境向け）:

```bash
cargo run
```
