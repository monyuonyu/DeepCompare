#!/usr/bin/env bash
# 画面の無い環境で GUI を起動し、実際に描画されたところを画像に落とす。
#
# 「コンパイルは通った」と「動いて見える」は別物なので、書体の読み込み、
# 差分の配色、行の並びまで目で確かめられるようにしておく。
#
# 使い方: tools/smoke.sh <左のファイル> <右のファイル> [出力.png]
set -euo pipefail

LEFT=${1:?左のファイルを指定}
RIGHT=${2:?右のファイルを指定}
OUT=${3:-/tmp/deepcompare.png}
BIN=${BIN:-target/release/deepcompare}
DISPLAY_NUM=${DISPLAY_NUM:-:99}

[ -x "$BIN" ] || { echo "実行ファイルが無い: $BIN" >&2; exit 1; }

cleanup() {
  [ -n "${APP_PID:-}" ] && kill "$APP_PID" 2>/dev/null || true
  [ -n "${XVFB_PID:-}" ] && kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT

Xvfb "$DISPLAY_NUM" -screen 0 1400x900x24 >/tmp/xvfb-smoke.log 2>&1 &
XVFB_PID=$!
# X サーバが受け付けるようになるまで待つ。
for _ in $(seq 1 50); do
  DISPLAY=$DISPLAY_NUM xdpyinfo >/dev/null 2>&1 && break
  sleep 0.2
done

# ソフトウェア描画を強制する（GPU が無い）。
export DISPLAY=$DISPLAY_NUM LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
"$BIN" "$LEFT" "$RIGHT" >/tmp/deepcompare-smoke.log 2>&1 &
APP_PID=$!

# モデルの読み込みと比較が終わるまで待つ。
sleep 12

if ! kill -0 "$APP_PID" 2>/dev/null; then
  echo "起動直後に落ちた。記録:" >&2
  cat /tmp/deepcompare-smoke.log >&2
  exit 1
fi

import -window root "$OUT"
echo "書き出した: $OUT"
echo "--- アプリの出力 ---"
cat /tmp/deepcompare-smoke.log
