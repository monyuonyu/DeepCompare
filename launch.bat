@echo off
:: ======================================================
:: バッチファイル名: launch.bat
::
:: 説明:
::   このバッチファイルは、仮想環境 (.venv) 内の Python インタープリターを使用して
::   DeepCompare.py を実行します。
::
::   可能であれば、ウィンドウを開かない pythonw.exe を使用し、
::   存在しない場合は python.exe にフォールバックします。
::
::   引数を渡すことで、比較対象のファイルを指定できます。
::
:: 使用方法:
::   launch.bat [file1] [file2]
::
:: 実行例:
::   launch.bat "C:\path\to\comptest1.py" "C:\path\to\comptest2.py"
::
:: 注意:
::   - バッチファイルは **UTF-8 (BOMなし)** で保存してください。
::   - コマンドプロンプトの文字コードを UTF-8 に変更して、文字化けを防ぎます。
:: ======================================================

:: コマンドプロンプトの文字コードを UTF-8 に設定（文字化け防止）
chcp 65001 >nul

setlocal

:: バッチファイルがあるディレクトリのパスを取得
set "SCRIPT_DIR=%~dp0"

:: .venv 内の pythonw.exe（コンソールを開かない実行ファイル）を使用
set "PYTHON_EXE=%SCRIPT_DIR%\.venv\Scripts\pythonw.exe"

:: pythonw.exe が存在しない場合は、通常の python.exe を使用
if not exist "%PYTHON_EXE%" (
    set "PYTHON_EXE=%SCRIPT_DIR%\.venv\Scripts\python.exe"
)

:: Python 実行ファイルが存在するか確認
if not exist "%PYTHON_EXE%" (
    echo [エラー] Python 実行ファイルが見つかりません: %PYTHON_EXE%
    pause
    exit /b 1
)

:: DeepCompare.py を新しいプロセスで起動し、引数を渡す
:: start コマンドを使用することで、バッチファイルのウィンドウを閉じる
start "" "%PYTHON_EXE%" "%SCRIPT_DIR%DeepCompare.py" %*

endlocal
exit /b
