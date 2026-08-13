#!/usr/bin/env bash
# CLI の終了コードを確かめる。
#
# **画面を開かない経路だけを見る。** ここが壊れると CI から呼んだときに
# GUI が立ち上がり、画面を出せない場所では固まったように見える
# （Windows で `--print-structured` を渡して実際に踏んだ）。
#
# 使い方: tools/cli-smoke.sh [deepcompare のパス]
set -u

DC="${1:-src/DeepCompare.App/bin/Debug/net10.0/deepcompare}"
if [ ! -x "$DC" ]; then
    echo "実行ファイルが見つかりません: $DC" >&2
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

printf '{"a":1}\n' > "$WORK/x.json"
printf '{"a":2}\n' > "$WORK/y.json"
printf 'id,v\n1,a\n'  > "$WORK/t1.csv"
printf 'id,v\n1,b\n'  > "$WORK/t2.csv"

pass=0
fail=0

check() {   # 名前 期待コード コマンド...
    local name="$1" want="$2"
    shift 2
    timeout 60 "$@" > /dev/null 2>&1
    local got=$?
    if [ "$got" = "$want" ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "  ✗ $name: 期待 $want / 実際 $got"
    fi
}

# **知らないオプションは断る。** 綴りを間違えたまま画面が開くと、
# CI では描画基盤の初期化に失敗し続けて止まらない。
check "知らないオプションは 2"   2 "$DC" --print-structured "$WORK/x.json" "$WORK/y.json"
check "似た綴りも 2"             2 "$DC" --print-text "$WORK/x.json" "$WORK/y.json"

# 終了コードは差分の有無を表す（0 差異なし / 1 差異あり / 2 異常）。
check "正しい綴りは通る"         1 "$DC" --print-json "$WORK/x.json" "$WORK/y.json"
check "値つきオプション"         1 "$DC" --print-table "$WORK/t1.csv" "$WORK/t2.csv" \
                                      --key id --structural
check "同じ中身なら 0"           0 "$DC" --print "$WORK/x.json" "$WORK/x.json" --structural
check "使い方は 0"               0 "$DC" --help
check "-h も 0"                  0 "$DC" -h
check "無いファイルは 2"         2 "$DC" --print "$WORK/無い.txt" "$WORK/x.json" --structural

echo "通過 $pass / 失敗 $fail"
[ "$fail" = 0 ]
