# DeepCompare

意味的な類似度で行を対応付けるコード比較ツール。

行を MiniLM で埋め込み、コサイン類似度を手がかりに Needleman-Wunsch で対応を決める。
変数名を一括で変えた行のように、文字列としては別物でも役割が同じ行を同じ行として並べられる。

C# + Avalonia。外部ランタイムを必要としないネイティブ実行ファイルとして配布する。

---

## 画面

- **起動画面** — 比較するものを選ぶ。ファイルでもフォルダーでも、ドラッグ＆ドロップでも指定できる。
- **フォルダー比較** — 再帰的に走査し、差異のあるファイルを一覧にする。行を開くとテキスト比較へ移る。
- **テキスト比較** — 行を対応付けて左右に並べ、行内の変わった部分だけ色を変える。

---

## 設計上の判断

### 重みを int8 で自前量子化し、推論も自前で持つ

「軽量な単一 exe にしたい」という要求が、この選択のすべての根拠になっている。

推論に ONNX Runtime を使うと `libonnxruntime`（28.5 MB）が付いて回る。あれは
ネイティブライブラリなので NativeAOT でも実行ファイルに畳み込めず、別ファイルとして
残ってしまう。必要なのは 6 層の BERT encoder 一本だけなので、そこだけ書いた。

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

| 重み | 参照実装との一致（コサイン、最小 / 平均） |
|---|---|
| f32 | **1.000000** / 1.000000 |
| int8 | **0.999095** / 0.999422 |

`dotnet test csharp/tests/DeepCompare.Engine.Tests/DeepCompare.Engine.Tests.csproj` で再現できる。
この段があるおかげで、不具合を「実装の誤り」と「量子化の誤差」に切り分けられる。

### 対応付けを二段に分ける

素朴にやると、両ファイルの全行の総当たりで類似度行列を作ることになる。1000 行同士なら
100 万マスの DP に 100 万回のモデル推論が乗る。

そうはせず、まず文字列一致で確実に対応する区間を畳み、残った「変化した塊」に対してだけ
意味的アライメントを行う。完全に一致するファイル同士なら**モデルは一度も動かない**。

3000 行対 3000 行、2% の行が変化という条件での実測:

| | 所要 | 埋め込んだ行 |
|---|---|---|
| 二段構え | **0.42 秒** | 108 / 6000（1.8%） |
| 素朴（全行を埋め込み、全面 DP） | 9.16 秒 | 6000 |

### 対応とみなす類似度は閾値として表す

対角のスコアを `類似度 - 閾値`、空きを 0 とする。こうすると「類似度がこの値を上回るときだけ
対にする」がそのまま式の意味になる。

罰則として表すと壊れる。「空き 1 つにつき -0.5」を置いて類似度をそのまま加算すると、
類似度が 0..1 に収まる以上、対にすれば 0 以上、空き 2 つなら -1.0 なので、
**どれだけ無関係な行同士でも必ず対にされ、空きが選ばれることが原理的に無くなる**。
左が `[A, B]`、右が `[B', C]` で B だけが対応する場合でも A↔B' と B↔C という誤った対ができる。

閾値は画面から調整できる。短いコード行は似ていても値が伸びにくく、
`self.config = config` と `self.settings = settings` のように明らかに対応する行でも
既定の 0.50 付近に落ちることがあるため。

### 行内差分は範囲の列として返す

文字列を組み立てて返すと、書式のために本文を加工することになる。範囲の列を返せば
書式付けの責任は描画側に移り、`<` や `&` を含む行——C++ のテンプレート、HTML、XML、
ジェネリクス——が壊れる余地がなくなる。

### フォルダー比較は「同じかどうか」までしか見ない

どこがどう違うかは行を開いたときに計算する。フォルダー全体に意味的な比較をかけると、
一覧が出るまでに何分もかかる。

内容比較はサイズが違えば読まずに確定し、同サイズのときだけ流し読みして差異が出た時点で
打ち切る。`.git`、`node_modules`、`target`、`bin`、`obj` などは既定で除外する
（生成物を並べると本当に見たい差分が埋もれる）。シンボリックリンクは辿らない
（循環すると走査が終わらない）。

---

## 実装中に見つけた、道具側の問題

いずれも f32 と比較して切り分けることで特定できた。

- **`Microsoft.ML.Tokenizers` の `BertTokenizer` が記号を黙って捨てる。** `+`(1009)、
  `<`(1026)、`>`(1028) は語彙にあるのに出力から消え、`x + 1` が `x 1` になっていた。
  コード比較では `x + 1` と `x - 1` の区別が失われるため許容できず、
  [トークナイザを自前で実装した](csharp/src/DeepCompare.Engine/WordPieceTokenizer.cs)。
- **.NET のコードページ 20932 は不正な EUC-JP を受理する。** `0xC3 0x28` を「構」として
  読むためバイナリを EUC-JP と誤判定していた。正しい日本語はどちらのコードページでも同じ
  バイト列で往復するので、壊れた入力を与えたときにだけ表面化する。51932 が正しい。
- **`MathF.Round` の既定は偶数丸め。** Rust の `round()` は 0 から遠い側へ丸めるので、
  同じ重みから別のバイト列が出る。`MidpointRounding.AwayFromZero` を明示する。

## 既知の限界

**日本語の比較品質には構造的な上限がある。** このモデルの正規化は濁点を落とす
（`だがぱ` → `たかは`）。`paraphrase-MiniLM-L6-v2` が英語中心の uncased モデルである
ためで、実装の問題ではない（Python の参照実装も同じ挙動）。多言語モデルに替えれば
解消するが、モデルが数倍大きくなる。

---

## ビルド

    cd csharp
    dotnet test tests/DeepCompare.Engine.Tests/DeepCompare.Engine.Tests.csproj
    dotnet run --project src/DeepCompare.App/DeepCompare.App.csproj

### 発行

ネイティブ実行ファイル（NativeAOT）:

    dotnet publish csharp/src/DeepCompare.App/DeepCompare.App.csproj -c Release -r linux-x64 -o out
    dotnet publish csharp/src/DeepCompare.App/DeepCompare.App.csproj -c Release -r win-x64 -o out

Linux では `clang` と `zlib1g-dev`、Windows では MSVC（Visual Studio Build Tools の
C++ ワークロード）がリンクに要る。Skia と HarfBuzz はネイティブライブラリなので、
実行ファイルの隣に置かれる。

MSVC を用意できない場合は、非 AOT の単一ファイルで代替できる。こちらは Linux からでも作れる:

    dotnet publish csharp/src/DeepCompare.App/DeepCompare.App.csproj -c Release -r win-x64 \
        --self-contained true -p:PublishAot=false -p:PublishSingleFile=true \
        -p:IncludeNativeLibrariesForSelfExtract=true -p:EnableCompressionInSingleFile=true

### モデルアセットの再生成

`assets/minilm.dcm` は追跡しているので通常は不要。作り直す場合:

    mkdir -p assets/src && cd assets/src
    B=https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2/resolve/main
    curl -sSLO $B/model.safetensors && curl -sSLO $B/vocab.txt
    cd ../..
    dotnet run --project csharp/src/DeepCompare.ModelPrep/DeepCompare.ModelPrep.csproj -c Release \
        -- assets/src/model.safetensors assets/minilm.dcm

参照実装との突き合わせをやり直す場合は、加えて ONNX 版と検証用の環境が要る:

    curl -sSL -o assets/src/model.onnx $B/onnx/model.onnx
    python3 -m venv .venv-ref && .venv-ref/bin/pip install onnxruntime numpy tokenizers
    .venv-ref/bin/python tools/reference_embeddings.py

## 画面を開かずに使う

遠隔での検証や CI で、別環境の出力と機械的に突き合わせるために用意した。

    deepcompare --print 左 右 -o 出力    比較結果を 1 行 1 レコードのテキストで出す
    deepcompare --font-check             日本語を表示できる書体があるかを調べる

`-o` があるのは Windows の都合で、GUI サブシステムの exe には標準出力が繋がらないため
コンソールから実行しても何も見えない。

## 履歴

元は Python + PyQt6 + sentence-transformers（`master` ブランチ）。
配布サイズを理由に書き直し、途中 Rust + egui + candle の実装を経ている
（コミット `1ce2d16` に残っている。同じ重みを使い、比較結果は C# 版と完全に一致していた）。

## ライセンス

MIT。モデルは
[sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
（Apache-2.0）。
