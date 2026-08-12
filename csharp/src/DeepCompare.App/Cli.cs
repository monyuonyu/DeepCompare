using System.Text;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>
/// 画面を開かずに使える経路。
///
/// GUI しか出口が無いと、動作確認が「人が画面を見る」ことでしか行えない。実際に
/// Windows での検証がそこで詰まった。比較結果をテキストとして出せるようにして、
/// 遠隔からでも別環境の出力と機械的に突き合わせられるようにする。
/// </summary>
internal static class Cli
{
    private static readonly string[] TakesValue = ["-o", "--threshold"];

    /// <summary>画面を開かずに済む要求なら処理して終了コードを返す。GUI を開くなら null。</summary>
    public static int? TryRun(string[] args, string usage)
    {
        if (args.Contains("-h") || args.Contains("--help"))
        {
            Console.Write(usage);
            return 0;
        }

        var output = ValueOf(args, "-o");

        if (args.Contains("--font-check"))
        {
            return Report(() => RunFontCheck(output));
        }
        if (!args.Contains("--print"))
        {
            return null;
        }

        var positional = Positional(args);
        if (positional.Length < 2)
        {
            Console.Error.WriteLine("--print には比較する 2 つのファイルが必要です");
            Console.Error.Write(usage);
            return 2;
        }

        var threshold = float.TryParse(ValueOf(args, "--threshold"), out var parsed)
            ? parsed
            : Aligner.DefaultPairThreshold;
        return Report(() => RunCompare(positional[0], positional[1], threshold, output));
    }

    /// <summary>オプションとその値を取り除いた、位置引数だけの列。</summary>
    public static string[] Positional(string[] args)
    {
        var result = new List<string>();
        var skipNext = false;
        foreach (var arg in args)
        {
            if (skipNext)
            {
                skipNext = false;
                continue;
            }
            if (TakesValue.Contains(arg))
            {
                skipNext = true;
            }
            else if (!arg.StartsWith('-'))
            {
                result.Add(arg);
            }
        }
        return result.ToArray();
    }

    private static string? ValueOf(string[] args, string flag)
    {
        var index = Array.IndexOf(args, flag);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : null;
    }

    private static void RunCompare(string leftPath, string rightPath, float threshold, string? output)
    {
        var left = TextDecoder.Decode(File.ReadAllBytes(leftPath));
        var right = TextDecoder.Decode(File.ReadAllBytes(rightPath));
        var embedder = Embedder.CreateFromEmbeddedAssets();

        var started = DateTime.UtcNow;
        var result = DiffComparer.Compare(left, right, embedder, new CompareOptions(threshold));
        var elapsed = DateTime.UtcNow - started;

        var text = new StringBuilder();
        text.AppendLine($"left  {leftPath} encoding={TextDecoder.Label(left.Encoding)} "
            + $"line_ending={TextDecoder.Label(left.LineEnding)} lines={left.Lines.Count}");
        text.AppendLine($"right {rightPath} encoding={TextDecoder.Label(right.Encoding)} "
            + $"line_ending={TextDecoder.Label(right.LineEnding)} lines={right.Lines.Count}");
        text.AppendLine($"stats rows={result.Stats.Rows} identical={result.Stats.IdenticalLines} "
            + $"embedded={result.Stats.EmbeddedLines} skipped_blocks={result.Stats.SkippedBlocks} "
            + $"elapsed_ms={(long)elapsed.TotalMilliseconds}");
        text.AppendLine($"threshold {threshold:F2}");
        text.AppendLine("---");

        // 1 行 1 レコード。環境をまたいで diff で比べられるよう、桁を固定する。
        // 種別は = 一致 / ~ 変更あり / - 左のみ / + 右のみ。
        foreach (var row in result.Rows)
        {
            var kind = (row.Left, row.Right) switch
            {
                (not null, not null) when row.IsUnchanged => '=',
                (not null, not null) => '~',
                (not null, null) => '-',
                (null, not null) => '+',
                _ => '?',
            };
            var leftNumber = row.Left is { } l ? (l + 1).ToString() : string.Empty;
            var rightNumber = row.Right is { } r ? (r + 1).ToString() : string.Empty;
            var score = row.Score is { } s ? s.ToString("F4") : "-";
            var inlineChanges = row.LeftSpans.Concat(row.RightSpans).Count(x => x.Kind == SpanKind.Changed);
            var content = row.Left is { } li ? left.Lines[li]
                : row.Right is { } ri ? right.Lines[ri]
                : string.Empty;
            text.AppendLine($"{kind} {leftNumber,6} {rightNumber,6} {score,6} {inlineChanges,2}  {content}");
        }

        Emit(text.ToString(), output);
    }

    /// <summary>
    /// 日本語を表示できる書体があるかを調べる。
    ///
    /// 「豆腐になっていないか」は本来目で見るしかないが、書体ファイルの有無を
    /// 先に確かめられれば、遠隔でも当たりを付けられる。
    /// </summary>
    private static void RunFontCheck(string? output)
    {
        string[] candidates = OperatingSystem.IsWindows()
            ? [
                @"C:\Windows\Fonts\msgothic.ttc",
                @"C:\Windows\Fonts\YuGothM.ttc",
                @"C:\Windows\Fonts\meiryo.ttc",
                @"C:\Windows\Fonts\msmincho.ttc",
            ]
            : [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
            ];

        var text = new StringBuilder();
        text.AppendLine($"platform {(OperatingSystem.IsWindows() ? "windows" : "unix")}");
        text.AppendLine("---");
        var found = false;
        foreach (var path in candidates)
        {
            if (File.Exists(path))
            {
                text.AppendLine($"FOUND   {path}  {new FileInfo(path).Length} バイト");
                found = true;
            }
            else
            {
                text.AppendLine($"MISS    {path}");
            }
        }
        text.AppendLine("---");
        text.AppendLine(found
            ? "結果: 日本語を表示できる書体が見つかった"
            : "結果: 候補の書体が無い。表示は環境の既定に依存する");
        Emit(text.ToString(), output);
    }

    /// <summary>
    /// 出力先。`-o` を用意しているのは Windows の都合で、GUI サブシステムで作った exe には
    /// 標準出力が繋がらないため、コンソールから実行しても何も見えない。
    /// </summary>
    private static void Emit(string text, string? output)
    {
        if (output is not null)
        {
            File.WriteAllText(output, text, new UTF8Encoding(false));
        }
        else
        {
            Console.Out.Write(text);
            Console.Out.Flush();
        }
    }

    private static int Report(Action action)
    {
        try
        {
            action();
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"エラー: {error.Message}");
            return 1;
        }
    }
}
