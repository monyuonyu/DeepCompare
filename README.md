# DeepCompare

意味的な類似度で行を対応付けるコード比較ツール。

行を MiniLM で埋め込み、コサイン類似度を手がかりに Needleman-Wunsch で対応を決める。
変数名を一括で変えた行のように、文字列としては別物でも役割が同じ行を同じ行として並べられる。

C# + Avalonia。外部ランタイムを必要としないネイティブ実行ファイルとして配布する。

![テキスト比較](docs/images/text.png)

上の例では `settings` を `config` に、`f` を `handle` に一括で変えている。
**文字列としては別物だが、同じ役割の行として並んでいる**（右端の数字が類似度）。
`save_*` と `merge_*` の順序が入れ替わっていることも、消えた／増えたではなく
移動として扱われている。

---

## 画面

### 起動画面

比較するものを選ぶ。ファイルでもフォルダーでも、ドラッグ＆ドロップでも指定できる。

![起動画面](docs/images/home.png)

### フォルダー比較

再帰的に走査し、差異のあるファイルを一覧にする。行を開くとテキスト比較へ移る。

![フォルダー比較](docs/images/folder.png)

**差異を含むフォルダーだけが開く。** 差異の無い `tests/` は閉じたまま。
絞り込んでいるのに畳まれていると、探しているものに辿り着くまで
フォルダーを 1 つずつ開くことになる。

### 表として比較（CSV / TSV）

![表比較](docs/images/table.png)

キー列に `id` を指定してあるので、**並び順が違っても同じ行として照合される**
（左の 1 行目と右の 2 行目）。変わったセルだけが青い。`updated` は
「見ない列」に入れてあるので、日付が全行で動いていても差分にならない。

行を上から突き合わせる作りだと、1 行挿入されただけで以降が全部ずれる。

### 構造として比較（JSON / XML / TOML / YAML）

![構造比較](docs/images/structured.png)

キーの順序が違っても差分にしない。**型の変化は `!` で明示する**
（`8080` → `"8080"` は目で見て気づけない差の筆頭）。

### ノートブック（.ipynb）

![ノートブック比較](docs/images/notebook.png)

セル単位で比べ、**既定では出力と実行回数を見ない**。実行しただけで
出力の base64 が数千行動き、直した 1 行がその中に埋もれるため。
出力は「あるか」だけを示す。

### キーボード

| キー | すること |
|---|---|
| `F3` / `Shift+F3` | 次を探す / 前を探す |
| `Alt+↓` / `Alt+↑` | 次の差分へ / 前の差分へ |
| `Ctrl+Z` / `Ctrl+Y` | 取り消し / やり直し |
| `F5` | 比べ直す |
| `Ctrl+S` | 左を保存 |

検索の隣に置換もある。**読み取り専用の側は触らず、取り消しで戻せる。**

### Git

![Git](docs/images/git.png)

作業ツリーと履歴。**塊ごとに索引へ載せられる**（SourceTree の hunk 単位 stage
に当たるもの）。差分は git 本体ではなくこちらの比較エンジンに掛けるので、
意味的な行の対応付けが効く。

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

`dotnet test tests/DeepCompare.Engine.Tests/DeepCompare.Engine.Tests.csproj` で再現できる。
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
  [トークナイザを自前で実装した](src/DeepCompare.Engine/WordPieceTokenizer.cs)。
- **.NET のコードページ 20932 は不正な EUC-JP を受理する。** `0xC3 0x28` を「構」として
  読むためバイナリを EUC-JP と誤判定していた。正しい日本語はどちらのコードページでも同じ
  バイト列で往復するので、壊れた入力を与えたときにだけ表面化する。51932 が正しい。
- **`MathF.Round` の既定は偶数丸め。** Rust の `round()` は 0 から遠い側へ丸めるので、
  同じ重みから別のバイト列が出る。`MidpointRounding.AwayFromZero` を明示する。

## 日本語を比べるなら多言語モデルを置く

**既定のモデルは日本語では意味的な対応付けが効かない。** `paraphrase-MiniLM-L6-v2`
は英語中心の uncased モデルで、正規化が濁点を落とす（`だがぱ` → `たかは`）。
語彙 30,522 のうち、かなは 184 項目・漢字は 486 項目しかなく、**濁点つきのかなは
1 つも無い**。実装の問題ではない（Python の参照実装も同じ挙動）。

この機体で実測した差:

| | 英語 22MB | 多言語 114MB |
|---|---|---|
| 「バグを直す」vs「ハクを直す」（低いほど良い） | **1.0000** | 0.9429 |
| 「設定ファイルを読み込む」vs「コンフィグを読む」（高いほど良い） | 0.9290 | 0.5131 |
| 「設定を読む」vs「DB へ接続する」（低いほど良い） | 0.9110 | 0.4080 |
| 「設定ファイルを読み込む」vs「load the configuration file」（高いほど良い） | **-0.1756** | 0.5751 |
| 「今日はいい天気ですね」vs「設定を読む」（低いほど良い） | 0.8442 | 0.0557 |

英語モデルは日本語をすべて 0.84〜1.00 に潰していて、**何も区別していない**
（高い値は意味ではなく文字の重なりによる）。同義と別物の差は 0.018 しかない。
多言語モデルでは 0.105 と 6 倍に広がり、日本語と英語の訳も繋がる。

日本語が半分を超えるファイルを英語モデルで比べると、その旨を画面と CLI に出す。

### 置き方

    B=https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main
    curl -sSLO $B/model.safetensors
    curl -sSL -o unigram.json $B/unigram.json

    # 語彙を「トークン<TAB>スコア」の形へ
    python3 -c "import json;d=json.load(open('unigram.json'));print('\n'.join(f'{t}\t{s}' for t,s in d['vocab']))" \
        > multilingual.vocab

    dotnet run --project src/DeepCompare.ModelPrep/DeepCompare.ModelPrep.csproj -c Release \
        -- model.safetensors multilingual.dcm

`multilingual.dcm` と `multilingual.vocab` を**実行ファイルと同じ場所に、対で**置く。
名前を揃えるのは、そこで対応を取っているから。片方だけ差し替えると番号は付くのに
モデルが学習した番号とは別物になるので、使う前に語彙数を照合して断る。

    deepcompare --print 左 右 --model multilingual.dcm

GUI では設定から選べる（`.dcm` を並べて置けば一覧に出る）。既定を変えるなら
環境変数 `DEEPCOMPARE_MODEL` にフルパスを入れる。

**コスト**: 114MB（英語版は 22MB）、読み込みが 2.8 秒（同 0.8 秒）。
日本語を扱わないなら置く必要はない。

本家 sentence-transformers との照合はコサイン 0.9996 以上（155 行、平均 0.99983）で、
int8 量子化の誤差の範囲。トークン化は参照実装（tokenizers）と 6161 行で完全一致。

---

## 構成

    src/
      DeepCompare.Engine/     比較エンジン。**画面にも通信にも依存しない**
      DeepCompare.App/        Avalonia の画面と CLI
      DeepCompare.Assist/     LLM 支援（任意）。**Engine を参照しない**
      DeepCompare.ModelPrep/  モデルを int8 へ変換する開発用ツール
    tests/                    試験（789 件）
    tools/                    参照実装との突き合わせ・CLI の確認
    assets/                   埋め込みモデルの実体
    docs/images/              README のスクリーンショット

**依存の向きを構造で示している。** Engine は Assist を知らないので、
比較の経路に通信が紛れ込む余地が無い。App だけが両方を知る。

---

## ビルド

        dotnet test tests/DeepCompare.Engine.Tests/DeepCompare.Engine.Tests.csproj
    dotnet run --project src/DeepCompare.App/DeepCompare.App.csproj

### 発行

ネイティブ実行ファイル（NativeAOT）:

    dotnet publish src/DeepCompare.App/DeepCompare.App.csproj -c Release -r linux-x64 -o out
    dotnet publish src/DeepCompare.App/DeepCompare.App.csproj -c Release -r win-x64 -o out

Linux では `clang` と `zlib1g-dev`、Windows では MSVC（Visual Studio Build Tools の
C++ ワークロード）がリンクに要る。Skia と HarfBuzz はネイティブライブラリなので、
実行ファイルの隣に置かれる。

MSVC を用意できない場合は、非 AOT の単一ファイルで代替できる。こちらは Linux からでも作れる:

    dotnet publish src/DeepCompare.App/DeepCompare.App.csproj -c Release -r win-x64 \
        --self-contained true -p:PublishAot=false -p:PublishSingleFile=true \
        -p:IncludeNativeLibrariesForSelfExtract=true -p:EnableCompressionInSingleFile=true

### モデルアセットの再生成

`assets/minilm.dcm` は追跡しているので通常は不要。作り直す場合:

    mkdir -p assets/src && cd assets/src
    B=https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2/resolve/main
    curl -sSLO $B/model.safetensors && curl -sSLO $B/vocab.txt
    cd ../..
    dotnet run --project src/DeepCompare.ModelPrep/DeepCompare.ModelPrep.csproj -c Release \
        -- assets/src/model.safetensors assets/minilm.dcm

参照実装との突き合わせをやり直す場合は、加えて ONNX 版と検証用の環境が要る:

    curl -sSL -o assets/src/model.onnx $B/onnx/model.onnx
    python3 -m venv .venv-ref && .venv-ref/bin/pip install onnxruntime numpy tokenizers
    .venv-ref/bin/python tools/reference_embeddings.py

## LLM 支援（任意・既定で無効）

git の状態を平易な言葉で説明し、次にできることを挙げる。コミットメッセージの
草案も書く。**接続先を設定するまで機能そのものが現れない。**

**外部 API ではなくローカルの LLM を第一の経路にしている。** 業務コードを扱う
道具なので、中身が機械の外に出ないことは譲れない。接続先は OpenAI 互換の
エンドポイント（Ollama / LM Studio / llama.cpp）を URL で指定する。

    export DEEPCOMPARE_ASSIST_ENDPOINT=http://localhost:11434/v1
    export DEEPCOMPARE_ASSIST_MODEL=qwen2.5:7b

    deepcompare --assist-probe            繋がるかを確かめる
    deepcompare --assist-status .         いまの状態を説明し、次の一手を挙げる
    deepcompare --assist-commit . --staged コミットメッセージの草案

GUI では Git 画面に「いまの状態を説明」「草案をもらう」が出る。

**LLM に git を実行させない。** 提案として返せるのは決まった操作の一覧
（commit / pull / push / stash / …）からの選択だけで、実際に走るのはアプリ側の
決め打ちのコードパス。リポジトリの中身に「すべて削除せよ」と書いてあっても
命令にならないのは、信用しているからではなく**通す経路が無いから**。
force push・reset --hard・リベースは一覧にすら入れていない。

衝突の**解決案**だけは既定で出さない。説明や分類と違い、意味を取り違えると
害になる生成で、小さいモデルはもっともらしく間違える（ビルドが通るぶんだけ
発見が遅れる）。`--assist-allow-resolution` で明示的に許すまで通信もしない。

鍵は設定ファイルに置かない（平文で残り、バックアップにも同期にも乗る）。
外部 API を使うなら `DEEPCOMPARE_ASSIST_KEY` に入れる。

**どのくらいのモデルが要るか**（この機体で実測）:

| モデル | 速さ（CPU） | 状態の説明 |
|---|---|---|
| Qwen2.5 1.5B | 15.9 トークン/秒 | 「現在のリポジトリの状態は以下の通りです。」— **中身が無い** |
| Qwen2.5 7B | 2.2 トークン/秒 | 「`rewrite` という分岐にあり、6 つのファイルが変更されていますが、まだコミットとして保存されていません。」 |

**7B から実用になる。** 枝の名前もファイル数も正しく拾う。1.5B は形は守るが、
中身は要約でも分類でもなく言い換えにとどまる。CPU では 7B で 1 分ほどかかる。

小さいモデルは、形（JSON Schema）で縛ると**文法には沿ったまま同じ一文を
延々と書き続ける**ことがある。長さの上限と繰り返しの抑制を必ず入れている。

**解決案は 7B でも危ない。** 実際に試した例:

    元    : def total(items): return sum(i.price for i in items)
    こちら: docstring を足した
    あちら: def total(items, tax=0.1): にして * (1 + tax)

    7B の提案: def total(items):        ← tax 引数が消えている
                   """合計を出す。"""
                   return sum(i.price for i in items) * (1 + 0.1)

**構文は正しく、実行も通る。** 呼び出し側が `total(items, tax=0.08)` として
いたら壊れるが、ビルドは通ってしまう。これが解決案を既定で無効にし、
提案として選択肢に並べる形にしている理由。

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
