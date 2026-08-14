#!/usr/bin/env bash
# 埋め込みモデルを Releases から取ってきて assets/ へ置く。
#
# **モデルは git に入れていない。** 59MB（日英）と 114MB（多言語）があり、
# 後者は GitHub の 100MiB 制限を超える。履歴に積むと差し替えるたびに
# その分だけ永久に残るので、Releases に置いて要るときに取る形にした。
#
# 使い方:
#   tools/fetch-model.sh            日英版（59MB）を取る
#   tools/fetch-model.sh full       多言語版（114MB）を取る
#
# **取れなくてもビルドは通る。** モデルが無ければ Myers で組む。
set -euo pipefail

RELEASE=models-v1
REPO=monyuonyu/DeepCompare
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/assets"

case "${1:-ja}" in
    ja)   NAME=multilingual-ja ;;
    full) NAME=multilingual ;;
    *)    echo "使い方: $0 [ja|full]" >&2; exit 2 ;;
esac

# 期待するハッシュ。**ここを消して「取れたから良し」にしない** ——
# 重みは中身を読めないので、壊れていても違う物でも動いてしまう。
declare -A SUMS=(
    [multilingual-ja.dcm]=838b71d3d0140ae252904e63fd4c14c05c392bf5d41099995fb569b4bd0179a8
    [multilingual-ja.vocab]=18c695b5064ac56919436f9c6e107c433b81e5364246553ec4dc2c93861fd2ca
    [multilingual.dcm]=30f3934f5a5a516eb426a98a6f626bf14a4a23783cb9ddec969623ae17af236e
    [multilingual.vocab]=4a5e1e0c56171db0ad3d46fcd98f4883aa81394f10aaec121f9351549b4cac35
)

mkdir -p "$DEST"

for file in "$NAME.dcm" "$NAME.vocab"; do
    target="$DEST/$file"
    expected="${SUMS[$file]}"

    if [ -f "$target" ] && echo "$expected  $target" | sha256sum -c --status; then
        echo "既にある: $file"
        continue
    fi

    echo "取得: $file"
    # **一時ファイルへ落としてから置き換える。** 途中で切れたものを
    # assets/ に残すと、次回「在る」と見なして壊れた物を読む。
    tmp="$(mktemp "$DEST/.$file.XXXXXX")"
    trap 'rm -f "$tmp"' EXIT
    curl -fsSL -o "$tmp" \
        "https://github.com/$REPO/releases/download/$RELEASE/$file"

    if ! echo "$expected  $tmp" | sha256sum -c --status; then
        echo "照合に失敗: $file" >&2
        echo "  期待 $expected" >&2
        echo "  実際 $(sha256sum "$tmp" | cut -d' ' -f1)" >&2
        exit 1
    fi

    mv "$tmp" "$target"
    trap - EXIT
    echo "  照合 OK"
done

echo "置いた: $DEST/$NAME.dcm ＋ .vocab"
