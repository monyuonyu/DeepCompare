using Avalonia;

namespace DeepCompare.App;

internal static class Program
{
    private const string Usage = """
        DeepCompare - 意味的な類似度で行を対応付けるコード比較ツール

          deepcompare [左 右]        GUI を開く（引数 2 つならそのまま比較する）
          deepcompare --print 左 右  画面を開かず、比較結果をテキストで出す
          deepcompare --print-folder 左 右
                                     フォルダーを比較して一覧を出す。終了コードは
                                     0 差異なし / 1 差異あり / 2 異常（CI で使える）。
                                     zip / tar / tar.gz も中身をフォルダーとして扱う
          deepcompare --merge <向き> 左 右
                                     差分を反映する。向きは to-right か to-left。
                                     -o か --in-place を付けない限り何も書かない
          deepcompare --print-table 左 右
                                     CSV/TSV を列単位で比較する
          deepcompare --merge3 祖先 左 右
                                     3 方向マージ。競合は <<<<<<< で囲む。
                                     終了コードは 0 競合なし / 1 競合あり
          deepcompare --font-check   日本語を表示できる書体があるかを調べる

        オプション
          -o <パス>        結果をファイルへ書く。Windows の GUI 版 exe は標準出力が
                           繋がらないため、遠隔から結果を回収するときはこれを使う
          --threshold <値> 対応とみなす類似度の下限（既定 0.50）
          --structural     埋め込みを使わず文字列一致だけで対応付ける。GUI が最初に
                           描くのと同じ内容で、桁違いに速い
          -h, --help       この説明

        重要でない差分（指定した違いは一致として扱う）
          --ws <方式>      空白の扱い。respect（既定）/ trailing（行末）/
                           ends（行頭行末）/ collapse（連続空白を 1 つ）/ all（全部）
          -i, --ignore-case   大文字小文字を区別しない
          --ignore-pattern <正規表現>
                           一致した部分を比較から除く。複数回指定できる。
                           例: --ignore-pattern '\d{4}-\d{2}-\d{2}'

        表形式の比較のオプション（--print-table と併用）
          --key <列>       行の対応付けに使う列。番号（1 始まり）か見出し名。
                           複数回指定可。**並び順が違っても照合できる**
          --ignore-column <列>
                           比較から外す列。複数回指定可
          --delimiter <字> 区切り文字。tab も指定できる（既定は拡張子から推定）
          --no-header      1 行目を見出しとして扱わない

        書き出しのオプション
          --report <形式>  unified（patch で適用できる差分）か html（左右並記）。
                           --print と併用。-o でファイルへ
          --context <行数> unified で変更の前後に付ける文脈（既定 3）
          --csv            --print-folder の結果を CSV で出す

        差分の反映のオプション（--merge と併用）
          --block <番号>   反映する塊を番号で指定。複数回指定可。既定は全部
          --in-place       写される側のファイルを直接書き換える
          -o <パス>        結果を別のファイルへ書く（元は変えない）

        フォルダー比較のオプション（--print-folder と併用）
          --include <型>   対象にする名前。* と ? が使える。複数回指定可。
                           ファイルにだけ効く（ディレクトリの走査は止めない）
          --exclude <型>   除外する名前。複数回指定可
          --by-timestamp   中身を読まず、大きさと更新時刻で比べる。数が多いとき用
          --tolerance <秒> 更新時刻の差をどこまで同じとみなすか
          --ignore-dst     ちょうど 1 時間のずれを同じとみなす（夏時間）
          --min-size <数>  この大きさ未満のファイルを対象から外す
          --max-size <数>  この大きさを超えるファイルを対象から外す
          --no-recurse     直下だけを比べる
          --changes-only   一致した項目を一覧に出さない
          --detect-renames 名前が変わっただけのファイルを見つけて併記する。
                           BC は名前一致しか見ないので、ここは差になる部分

        """;

    [STAThread]
    public static int Main(string[] args)
    {
        // 画面を開かない経路を先に捌く。GUI を初期化する前に分岐しないと、
        // 表示できない環境で使えなくなる。
        if (Cli.TryRun(args, Usage) is { } exitCode)
        {
            return exitCode;
        }

        App.StartupFiles = Cli.Positional(args);
        return BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp()
        => AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();
}
