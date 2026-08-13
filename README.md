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

なぜその作りにしたかは [docs/design.md](docs/design.md) にまとめてある。
重みを int8 で自前量子化した理由、前向き計算の正しさをどう担保しているか、
対応付けを二段に分ける理由、実装中に見つけた道具側の問題（`BertTokenizer` が
記号を黙って捨てる、.NET のコードページ 20932 が不正な EUC-JP を受理する）など。

---

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

.NET 10 SDK が要る。それ以外の下準備は無い。

    dotnet run --project src/DeepCompare.App/DeepCompare.App.csproj

試験:

    dotnet test tests/DeepCompare.Engine.Tests/DeepCompare.Engine.Tests.csproj
    dotnet test tests/DeepCompare.Assist.Tests/DeepCompare.Assist.Tests.csproj

**画面を開かない経路**が壊れていないかの確認（終了コードを見る）:

    dotnet build src/DeepCompare.App/DeepCompare.App.csproj
    ./tools/cli-smoke.sh src/DeepCompare.App/bin/Debug/net10.0/deepcompare

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

元は Python + PyQt6 + sentence-transformers（タグ `python-legacy`）。
PyQt6 と sentence-transformers で配布が数百 MB になるのを理由に書き直し、
途中 Rust + egui + candle の実装を経ている（コミット `1ce2d16`。同じ重みを使い、
比較結果は C# 版と完全に一致していた）。

比較の考え方——**意味的な類似度で行を対応付ける**——は最初の実装で固まっており、
書き直しで変わったのは配布の形と、その周りに足した機能。

## ライセンス

MIT（`LICENSE`）。モデルは
[sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
（Apache-2.0）。
