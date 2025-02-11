# DeepCompare

## 概要

DeepCompare は、Python で実装されたシンプルなコード比較ツールです。MiniLM モデル（[sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)）を活用し、2 つのソースコードファイルの行ごとの意味的類似度を計算し、動的計画法を用いた最適な行アライメントを行います。GUI は PyQt6 を使用し、直感的に操作できるシンプルなインターフェースを提供します。

---

## 特徴

DeepCompare は、従来の文字ベースの比較ツールとは異なり、**AI を活用してコードの意味的な類似性を評価できる** 点が特徴です。大規模な比較ツールと比べるとシンプルな設計ですが、手軽にコードの差分を確認する用途に適しています。

### 1. 行単位の意味的類似性評価
各ファイルの行を MiniLM によりベクトル化し、コサイン類似度を計算することで、**既存のコンペアツールでは捉えにくい意味的な類似性を考慮した比較**が可能です。

### 2. 最適アライメントの採用
Needleman–Wunsch アルゴリズムを使用し、**コサイン類似度スコアを考慮しながら** 2 つのファイル間の最適な対応を決定。これにより、変更箇所を整理し、より分かりやすく表示します。

### 3. 文字レベルの差分ハイライト
差分がある行は背景色で強調表示され、行内の差分部分はオレンジ色で視認性を向上させています。

### 4. シンプルで使いやすい GUI
- ダークテーマ対応
- フレームレスウィンドウ
- ドラッグ＆ドロップによるファイル選択
- ウィンドウのリサイズ対応

---

## 必要な環境・依存ライブラリ

- Python 3.7 以降
- [PyTorch](https://pytorch.org/)
- [sentence-transformers](https://www.sbert.net/)
- [PyQt6](https://pypi.org/project/PyQt6/)

### インストール方法

1. リポジトリをクローンまたはダウンロードします。
2. 仮想環境を作成（任意）。

   ```bash
   python -m venv .venv
   ```
3. 仮想環境を有効化します。

   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
4. 依存ライブラリをインストールします。

   ```bash
   pip install -r requirements.txt
   ```

---

## 使い方

1. `DeepCompare.py` を実行します。

   ```bash
   python DeepCompare.py
   ```
2. GUI が起動し、2 つのコードファイルを指定（またはドラッグ＆ドロップ）します。
3. 「比較開始」ボタンをクリックすると、行ごとの類似度を計算し、結果が表示されます。
4. 差分がある行はハイライト表示され、行内の変更はオレンジ色で強調されます。

---

## プロジェクト構成

```
DeepCompare/
├── DeepCompare.py         # メインプログラム（GUI 版）
├── README.md              # このドキュメント
└── requirements.txt       # 依存ライブラリ一覧
```

