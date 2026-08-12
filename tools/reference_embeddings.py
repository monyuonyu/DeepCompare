#!/usr/bin/env python3
"""自前実装の BERT を突き合わせるための参照値を作る。

ONNX Runtime を使うのは、torch を入れずに済むからというだけでなく、Rust 側とは
完全に独立した実装だから。両者が一致すれば、実装の取り違え（GELU の種類、
LayerNorm の位置、注意マスクの向き、プーリングの式）はまず起きていないと言える。

出力は JSON:
  {"texts": [...], "vectors": [[...], ...]}   # 平均プーリング後、L2 正規化済み
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# 比較対象がソースコードなので、素の英文だけでなくコードらしい行と、
# 似ているが同じではない対を混ぜる。
TEXTS = [
    "def main():",
    "def main(argv):",
    "    return x + 1",
    "    return x - 1",
    "}",
    "",
    "// 設定ファイルを読み込む",
    "// 設定ファイルを書き出す",
    "std::vector<int>& xs = a && b;",
    "class FileDropLineEdit(QLineEdit):",
    "The quick brown fox jumps over the lazy dog.",
    "import os",
    "import sys",
    "x" * 300,
]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    model_path = root / "assets" / "src" / "model.onnx"
    tokenizer_path = root / "assets" / "tokenizer.json"
    out_path = root / "tests" / "reference_embeddings.json"

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=512)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    expected_inputs = {i.name for i in session.get_inputs()}

    vectors = []
    for text in TEXTS:
        encoding = tokenizer.encode(text)
        ids = np.array([encoding.ids], dtype=np.int64)
        mask = np.array([encoding.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in expected_inputs:
            feed["token_type_ids"] = np.zeros_like(ids)

        hidden = session.run(None, feed)[0]  # [1, seq, hidden]

        # 平均プーリング。詰め物は無い（1 件ずつ流している）が、Rust 側と同じ式にする。
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        pooled = pooled / np.clip(
            np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12, None
        )
        vectors.append(pooled[0].astype(float).tolist())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"texts": TEXTS, "vectors": vectors}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{out_path} に {len(TEXTS)} 件 x {len(vectors[0])} 次元を書いた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
