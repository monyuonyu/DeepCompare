# DeepCompare

意味的な類似度で行を対応付けるコード比較ツール。

行を MiniLM で埋め込み、コサイン類似度を手がかりに Needleman-Wunsch で対応を決める。
変数名を一括で変えた行のように、文字列としては別物でも役割が同じ行を同じ行として並べられる。

---

## 現状

**書き直しの最中。** `master` に元の Python 版があり、`rust-rewrite` ブランチに
2 つの実装が並んでいる。

| | `crates/`（Rust） | `csharp/`（C# + Avalonia） |
|---|---|---|
| GUI | egui（独自描画） | Avalonia + XAML（Fluent） |
| 推論 | candle | 自前実装（`System.Numerics.Tensors`） |
| 画面 | テキスト比較のみ | 起動画面 / フォルダー比較 / テキスト比較 |
| 配布（Linux） | **単一 exe 35 MB** | 46 MB + Skia/HarfBuzz 14 MB（3 ファイル） |
| 配布（Windows） | **単一 exe 33 MB** | 単一 exe 72.8 MB（非 AOT） |

**両者は同じ重みファイル（`assets/minilm.dcm`）を使い、比較結果は完全に一致する**
（行データ部の SHA256 が一致）。違いは見た目と配布形態だけ。

Rust 版は動作する状態のまま残してある。C# へ舵を切った判断が変わったときに戻れるようにするため。

### 進捗

- [x] エンジン（符号化判定・アライメント・行内差分・推論）— Rust 56 相当 / C# 56 テスト
- [x] テキスト比較の画面
- [x] フォルダー比較の画面と起動画面（C# 版のみ）
- [ ] Windows での AOT 発行（VS Build Tools 導入待ち）

---

## 設計上の判断

### 重みを int8 で自前量子化し、推論も自前で持つ

同じ結論に二度到達した。Rust では `ort`（ONNX Runtime）が静的リンクにソースからの
ビルドを要求しクロスコンパイルが現実的でないため candle を選び、C# では
`libonnxruntime`（28.5 MB）がネイティブライブラリゆえ NativeAOT でも実行ファイルに
畳み込めず別ファイルとして残るため、推論そのものを書いた。

重みは行ごとの対称 int8 で保存し、読み込み時に f32 へ戻す。演算は f32 のままなので
量子化された行列積を書き起こす必要がない。

    86.7 MiB -> 21.8 MiB (25.2%)   相対二乗誤差 0.86%

**配布されている量子化 ONNX には使えない問題があった。** あちらは活性値まで実行時に
量子化するため、詰め物が入ると同じ行でも埋め込みが 1% ほど動く（実測 0.9907）。
つまり無関係な行を 1 行足しただけでスコアが変わる。重みだけを量子化すればこれは起きない。

### 前向き計算の正しさをどう担保するか

自前で書いた以上、正しさは示す必要がある。GELU の種類、LayerNorm を残差の前に置くか
後に置くか、注意マスクの向き、プーリングの分母——どれを取り違えても「それらしい数字」は
出てしまう。そこで **ONNX Runtime（完全に独立した実装）と突き合わせている**。

| 実装 | 参照実装との一致（コサイン、最小 / 平均） |
|---|---|
| Rust f32 | 0.999999 / 1.000000 |
| Rust int8 | 0.999095 / 0.999422 |
| C# f32 | **1.000000** / 1.000000 |
| C# int8 | **0.999095** / 0.999422 |

Rust 版と C# 版が小数 6 桁まで同値。この段があったおかげで、移植中の不具合を
「実装の誤り」と「量子化の誤差」に切り分けられた。

### 対応付けを二段に分ける

旧実装は両ファイルの全行の総当たりで類似度行列を作っていた。1000 行同士なら
100 万マスの DP に 100 万回のモデル推論が乗る。

新実装はまず文字列一致で確実に対応する区間を畳み、残った「変化した塊」に対してだけ
意味的アライメントを行う。完全に一致するファイル同士なら**モデルは一度も動かない**。

3000 行対 3000 行、2% の行が変化という条件での実測（どちらも同じ Rust 実装なので、
差は言語ではなく手順によるもの）:

| | 所要 | 埋め込んだ行 |
|---|---|---|
| 二段構え | **0.423 秒** | 108 / 6000（1.8%） |
| 素朴（全行を埋め込み、全面 DP） | 9.159 秒 | 6000 |

### フォルダー比較は「同じかどうか」までしか見ない

どこがどう違うかは行を開いたときに計算する。フォルダー全体に意味的な比較をかけると、
一覧が出るまでに何分もかかる。

内容比較はサイズが違えば読まずに確定し、同サイズのときだけ流し読みして差異が出た時点で
打ち切る。`.git`、`node_modules`、`target`、`bin`、`obj` などは既定で除外する
（生成物を並べると本当に見たい差分が埋もれる）。シンボリックリンクは辿らない
（循環すると走査が終わらない）。

---

## 旧実装から直したこと

書き直しにあたって、元の `DeepCompare.py` を読んで見つかった問題を仕様として扱った。

| 箇所 | 問題 | 対処 |
|---|---|---|
| `diff_characters` | 差分を HTML 文字列として組み立て、本文をエスケープしていなかった。`<`, `&` を含む行——C++ のテンプレート、HTML、XML、ジェネリクス——は表示が壊れるか内容が消えた | 文字列ではなく範囲の列を返す。書式付けは描画側の責任にした |
| `compare_files` | GUI スレッドで同期実行。起動引数つきだとモデル読み込み前に走り、スプラッシュ画面ごと固まった | 比較もモデル読み込みも作業スレッドが持つ |
| `align_lines` | 類似度行列を丸ごと実体化した上での全面 DP | 一致区間を畳んでから、変化した塊だけを DP にかける |
| `align_lines` の罰則 | 空き 1 つにつき -0.5、類似度は 0..1。**対にすれば必ず得なので、空きが選ばれることが原理的に無かった**。左が `[A, B]`、右が `[B', C]` で B だけが対応する場合も A↔B' と B↔C という誤った対を作る | 対角のスコアを `類似度 - 閾値` とし、空きを 0 にした。閾値は画面から調整できる |
| `update_table` | 差分行ごとに `QLabel` を生成。行数分のウィジェットが並ぶので数千行で操作不能 | 画面に映っている行だけを描く |
| `open(..., encoding="utf-8")` | 固定。Shift_JIS のファイルで即エラー | BOM・UTF-8・Shift_JIS・EUC-JP・UTF-16 を判定 |
| `create_local_server` | `os.path.exists(INSTANCE_KEY)` がパスではなく名前を見ていて無意味 | シングルインスタンス機能自体を落とした |

## 移植中に見つけた、道具側の問題

C# へ移す過程で 3 件出た。いずれも fp32 と比較して切り分けることで特定できた。

- **`Microsoft.ML.Tokenizers` の `BertTokenizer` が記号を黙って捨てる。** `+`(1009)、
  `<`(1026)、`>`(1028) は語彙にあるのに出力から消え、`x + 1` が `x 1` になっていた。
  コード比較では `x + 1` と `x - 1` の区別が失われるため許容できず、トークナイザを自前で実装した。
- **.NET のコードページ 20932 は不正な EUC-JP を受理する。** `0xC3 0x28` を「構」として
  読むためバイナリを EUC-JP と誤判定していた。正しい日本語はどちらのコードページでも同じ
  バイト列で往復するので、壊れた入力を与えたときにだけ表面化する。51932 が正しい。
- **量子化 ONNX の詰め物が結果に漏れる**（上記）。

## 既知の限界

**日本語の比較品質には構造的な上限がある。** このモデルの正規化は濁点を落とす
（`だがぱ` → `たかは`）。`paraphrase-MiniLM-L6-v2` が英語中心の uncased モデルである
ためで、実装の問題ではない（Python の参照実装も同じ挙動）。多言語モデルに替えれば
解消するが、モデルが数倍大きくなる。

---

## ビルド

### C# 版

    cd csharp
    dotnet test tests/DeepCompare.Engine.Tests/DeepCompare.Engine.Tests.csproj
    dotnet run --project src/DeepCompare.App/DeepCompare.App.csproj

単一 exe の発行（Linux、NativeAOT）:

    dotnet publish src/DeepCompare.App/DeepCompare.App.csproj -c Release -r linux-x64 -o out

Windows 向けは NativeAOT のリンクに MSVC が要るため、Windows 上で発行する。
MSVC を用意できない場合は非 AOT の単一ファイルで代替でき、これは Linux からでも作れる:

    dotnet publish src/DeepCompare.App/DeepCompare.App.csproj -c Release -r win-x64 \
        --self-contained true -p:PublishAot=false -p:PublishSingleFile=true \
        -p:IncludeNativeLibrariesForSelfExtract=true -p:EnableCompressionInSingleFile=true

### Rust 版

    cargo build --release
    sudo apt-get install gcc-mingw-w64-x86-64   # Windows 向けクロスビルド用
    rustup target add x86_64-pc-windows-gnu
    cargo build --release --target x86_64-pc-windows-gnu

### モデルアセットの再生成

`assets/minilm.dcm` は追跡しているので通常は不要。作り直す場合:

    mkdir -p assets/src && cd assets/src
    B=https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2/resolve/main
    curl -sSLO $B/model.safetensors && curl -sSLO $B/vocab.txt
    cd ../..
    cargo run --release -p deepcompare-model-prep -- assets/src/model.safetensors assets/minilm.dcm

参照実装との突き合わせをやり直す場合は、加えて ONNX 版と検証用の環境が要る:

    curl -sSL -o assets/src/model.onnx $B/onnx/model.onnx
    python3 -m venv .venv-ref && .venv-ref/bin/pip install onnxruntime numpy tokenizers
    .venv-ref/bin/python tools/reference_embeddings.py

## 画面を開かずに使う

どちらの実装にも同じ経路がある。遠隔での検証や CI で、別環境の出力と機械的に
突き合わせるために用意した。

    deepcompare --print 左 右 -o 出力    比較結果を 1 行 1 レコードのテキストで出す
    deepcompare --font-check             日本語を表示できる書体があるかを調べる

`-o` があるのは Windows の都合で、GUI サブシステムの exe には標準出力が繋がらないため
コンソールから実行しても何も見えない。

Rust 版にはさらに `--screenshot out.png` があり、自分の描画結果を PNG に保存する。
画面を撮るのではないので、Windows がロックされていても本物の描画が得られる。

## ライセンス

MIT。モデルは
[sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
（Apache-2.0）。
