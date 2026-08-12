using Avalonia;

namespace DeepCompare.App;

internal static class Program
{
    private const string Usage = """
        DeepCompare - 意味的な類似度で行を対応付けるコード比較ツール

          deepcompare [左 右]        GUI を開く（引数 2 つならそのまま比較する）
          deepcompare --print 左 右  画面を開かず、比較結果をテキストで出す
          deepcompare --font-check   日本語を表示できる書体があるかを調べる

        オプション
          -o <パス>        結果をファイルへ書く。Windows の GUI 版 exe は標準出力が
                           繋がらないため、遠隔から結果を回収するときはこれを使う
          --threshold <値> 対応とみなす類似度の下限（既定 0.50）
          -h, --help       この説明

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
