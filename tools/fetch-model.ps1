# 埋め込みモデルを Releases から取ってきて assets\ へ置く（Windows 用）。
# 中身は tools/fetch-model.sh と同じ。使い方:
#   pwsh tools/fetch-model.ps1          日英版（59MB）
#   pwsh tools/fetch-model.ps1 full     多言語版（114MB）
param([ValidateSet('ja', 'full')][string]$Which = 'ja')

$ErrorActionPreference = 'Stop'

$release = 'models-v1'
$repo = 'monyuonyu/DeepCompare'
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root 'assets'

$name = if ($Which -eq 'ja') { 'multilingual-ja' } else { 'multilingual' }

# 期待するハッシュ。**取れたから良しにしない** —— 重みは中身を読めないので、
# 壊れていても違う物でも動いてしまう。
$sums = @{
    'multilingual-ja.dcm'   = '838b71d3d0140ae252904e63fd4c14c05c392bf5d41099995fb569b4bd0179a8'
    'multilingual-ja.vocab' = '18c695b5064ac56919436f9c6e107c433b81e5364246553ec4dc2c93861fd2ca'
    'multilingual.dcm'      = '30f3934f5a5a516eb426a98a6f626bf14a4a23783cb9ddec969623ae17af236e'
    'multilingual.vocab'    = '4a5e1e0c56171db0ad3d46fcd98f4883aa81394f10aaec121f9351549b4cac35'
}

New-Item -ItemType Directory -Force -Path $dest | Out-Null

foreach ($file in @("$name.dcm", "$name.vocab")) {
    $target = Join-Path $dest $file
    $expected = $sums[$file]

    if ((Test-Path $target) -and
        ((Get-FileHash $target -Algorithm SHA256).Hash -eq $expected.ToUpper())) {
        Write-Host "既にある: $file"
        continue
    }

    Write-Host "取得: $file"
    # **一時ファイルへ落としてから置き換える。** 途中で切れたものを
    # assets\ に残すと、次回「在る」と見なして壊れた物を読む。
    $tmp = "$target.partial"
    Invoke-WebRequest -Uri "https://github.com/$repo/releases/download/$release/$file" `
        -OutFile $tmp -UseBasicParsing

    $actual = (Get-FileHash $tmp -Algorithm SHA256).Hash
    if ($actual -ne $expected.ToUpper()) {
        Remove-Item $tmp -Force
        throw "照合に失敗: $file`n  期待 $($expected.ToUpper())`n  実際 $actual"
    }

    Move-Item $tmp $target -Force
    Write-Host "  照合 OK"
}

Write-Host "置いた: $dest\$name.dcm ＋ .vocab"
