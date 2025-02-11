# DeepCompare

DeepCompare は、Python で実装されたシンプルなコード比較ツールです。MiniLM モデル（[sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)）を活用し、2 つのソースコードファイルの行ごとの意味的類似度を計算し、動的計画法を用いた最適な行アライメントを行います。GUI は PyQt6 を使用しており、直感的に操作できるシンプルなインターフェースを備えています。

---

## 特徴

DeepCompare は、従来の文字ベースの比較ツールとは異なり、**AI による意味的な類似度を考慮したコード比較**を行う点が特徴です。ただし、大規模なツールと比べるとシンプルな設計になっており、基本的なコード比較を手軽に行う用途に適しています。

- **行単位比較**  
  各ファイルの行を MiniLM によりベクトル化し、コサイン類似度を計算することで、単純な文字列比較では捉えきれない類似性を検出できます。

- **最適アライメント**  
  Needleman–Wunsch アルゴリズムを利用し、2 つのファイル間の最適な対応を決定。これにより、変更箇所が整理された形で表示されます。

- **文字レベルの差分ハイライト**  
  差分がある行は背景色で強調表示され、行内の差分部分はオレンジ色で視認性を向上させています。

- **シンプルな GUI**  
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
