using System.Text.Json;
using System.Text.Json.Serialization;

namespace DeepCompare.Engine;

public enum SessionKind
{
    Text,
    Folder,
}

/// <summary>
/// 保存した比較の組み合わせ。
///
/// 設定は「そのとき何を無視していたか」まで含める。パスだけ覚えても、
/// 空白の扱いや無視する正規表現を毎回入れ直すことになり、結局手作業が残る。
/// </summary>
public sealed record Session
{
    public string Name { get; init; } = string.Empty;
    public SessionKind Kind { get; init; }
    public string LeftPath { get; init; } = string.Empty;
    public string RightPath { get; init; } = string.Empty;

    /// <summary>最後に開いた日時。「最近使った項目」の並べ替えに使う。</summary>
    public DateTime LastUsed { get; init; }

    // --- テキスト比較の設定 ---
    public float PairThreshold { get; init; } = Aligner.DefaultPairThreshold;
    public WhitespaceMode Whitespace { get; init; }
    public bool IgnoreCase { get; init; }
    public List<string> IgnoredPatterns { get; init; } = [];

    // --- フォルダー比較の設定 ---
    public List<string> IncludeNames { get; init; } = [];
    public List<string> ExcludeNames { get; init; } = [];
    public FolderComparisonMode FolderMode { get; init; }
    public double TimestampToleranceSeconds { get; init; }
    public bool IgnoreDaylightSavingOffset { get; init; }

    /// <summary>保存した設定から比較の指定を組み立てる。</summary>
    public CompareOptions ToCompareOptions() => new(
        PairThreshold,
        Importance: new Importance(Whitespace, IgnoreCase, IgnoredPatterns));

    public FolderCompareOptions ToFolderOptions() => new()
    {
        Filter = new NameFilter(IncludeNames, ExcludeNames),
        Mode = FolderMode,
        TimestampToleranceSeconds = TimestampToleranceSeconds,
        IgnoreDaylightSavingOffset = IgnoreDaylightSavingOffset,
    };
}

public sealed record SessionFile
{
    public List<Session> Sessions { get; init; } = [];

    /// <summary>明るいテーマを使うか。既定は明るい方。</summary>
    public bool LightTheme { get; init; } = true;

    /// <summary>
    /// LLM 支援の接続先（OpenAI 互換）。**空なら機能を出さない。**
    /// Ollama なら http://localhost:11434/v1。
    /// </summary>
    public string AssistEndpoint { get; init; } = string.Empty;

    /// <summary>使うモデルの名前。</summary>
    public string AssistModel { get; init; } = string.Empty;

    /// <summary>
    /// 衝突の解決案まで出してよいか。**既定は false。**
    /// 弱いモデルはもっともらしく間違え、ビルドが通るぶんだけ発見が遅れる。
    /// </summary>
    public bool AssistAllowResolution { get; init; }

    /// <summary>
    /// 表示に使う言語。**いまは日本語だけ。**
    /// 置き場所を先に決めておく（後から場所を変えると、覚えた設定が読めなくなる）。
    /// </summary>
    public string Language { get; init; } = "ja";

    /// <summary>
    /// 前に閉じたときのウィンドウ。**0 なら「まだ知らない」。**
    /// 最大化していたかどうかも覚える（大きさだけ戻すと、最大化していた人が
    /// 毎回小さな窓で始まることになる）。
    /// </summary>
    public double WindowWidth { get; init; }
    public double WindowHeight { get; init; }
    public bool WindowMaximized { get; init; }

    // **鍵はここに置かない。** 設定ファイルは平文で、バックアップにも
    // 同期にも乗る。外部の API を使うなら環境変数から読む
    // （DEEPCOMPARE_ASSIST_KEY）。
}

/// <summary>
/// NativeAOT では実行時の型情報が削られるため、反射に頼る直列化は動かない。
/// 生成された文脈を通す。
/// </summary>
[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(SessionFile))]
internal sealed partial class SessionJsonContext : JsonSerializerContext;

/// <summary>
/// セッションの保存先。
///
/// 置き場所は環境の設定ディレクトリ。実行ファイルの隣に書くと、書き込めない場所へ
/// 置かれた場合に動かなくなる。
///
/// 読み込みは壊れたファイルでも落ちない。設定が壊れているせいで道具ごと起動しない
/// のは避ける。
/// </summary>
public sealed class SessionStore
{
    private readonly string _path;

    public SessionStore(string? path = null)
    {
        _path = path ?? DefaultPath();
    }

    public static string DefaultPath()
    {
        var directory = Path.Combine(
            Environment.GetFolderPath(
                Environment.SpecialFolder.ApplicationData,
                Environment.SpecialFolderOption.DoNotVerify),
            "DeepCompare");
        return Path.Combine(directory, "sessions.json");
    }

    /// <summary>保存されているものを、最近使った順で返す。</summary>
    public List<Session> Load()
    {
        try
        {
            if (!File.Exists(_path))
            {
                return [];
            }
            var json = File.ReadAllText(_path);
            var file = JsonSerializer.Deserialize(json, SessionJsonContext.Default.SessionFile);
            return file is null
                ? []
                : [.. file.Sessions.OrderByDescending(s => s.LastUsed)];
        }
        catch (Exception)
        {
            // 壊れた設定で起動できなくなる方が困る。空として扱う。
            return [];
        }
    }

    /// <summary>
    /// 設定ファイルを丸ごと読む。無ければ、あるいは壊れていれば既定。
    ///
    /// **落とさない。** 設定が壊れているせいで道具ごと起動しないのは避ける。
    /// </summary>
    public SessionFile LoadFile()
    {
        try
        {
            return File.Exists(_path)
                ? JsonSerializer.Deserialize(
                    File.ReadAllText(_path), SessionJsonContext.Default.SessionFile)
                  ?? new SessionFile()
                : new SessionFile();
        }
        catch (Exception)
        {
            return new SessionFile();
        }
    }

    /// <summary>
    /// 既定の置き場所を指すもの。
    /// **毎回 new しても同じ場所を見る**ので、使い回しのために置いてある。
    /// </summary>
    public static SessionStore Default { get; } = new();

    /// <summary>テーマの選択。設定が読めなければ既定（明るい方）。</summary>
    public bool LoadLightTheme()
    {
        try
        {
            if (!File.Exists(_path))
            {
                return true;
            }
            var file = JsonSerializer.Deserialize(
                File.ReadAllText(_path), SessionJsonContext.Default.SessionFile);
            return file?.LightTheme ?? true;
        }
        catch (Exception)
        {
            return true;
        }
    }

    /// <summary>テーマの選択だけを書き換える。保存済みの比較は残す。</summary>
    public void SaveLightTheme(bool light)
    {
        Save(Load(), light);
    }

    /// <summary>
    /// ウィンドウの大きさだけを書き換える。
    ///
    /// **0 を渡されたら前の値を保つ。** 最大化したまま閉じると
    /// 「戻したときの大きさ」が取れないが、それで前の値を消してしまうと、
    /// 最大化を解除した瞬間に初期値の窓に戻ってしまう。
    /// </summary>
    public void SaveWindow(double width, double height, bool maximized)
    {
        var file = LoadFile();
        WriteFile(file with
        {
            WindowWidth = width > 0 ? width : file.WindowWidth,
            WindowHeight = height > 0 ? height : file.WindowHeight,
            WindowMaximized = maximized,
        });
    }

    /// <summary>言語だけを書き換える。</summary>
    public void SaveLanguage(string code)
    {
        var file = LoadFile();
        WriteFile(file with { Language = code });
    }

    /// <summary>LLM 支援の設定だけを書き換える。**鍵は保存しない。**</summary>
    public void SaveAssist(string endpoint, string model, bool allowResolution)
    {
        var file = LoadFile();
        WriteFile(file with
        {
            AssistEndpoint = endpoint.Trim(),
            AssistModel = model.Trim(),
            AssistAllowResolution = allowResolution,
        });
    }

    /// <summary>
    /// 設定ファイルをまるごと書く。
    ///
    /// **書けなくても落とさない。** 設定が保存できないことと、
    /// 道具が使えないことは別。
    /// </summary>
    private void WriteFile(SessionFile file)
    {
        try
        {
            var directory = Path.GetDirectoryName(_path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            File.WriteAllText(_path,
                JsonSerializer.Serialize(file, SessionJsonContext.Default.SessionFile));
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
        }
    }

    public void Save(IReadOnlyList<Session> sessions) => Save(sessions, LoadLightTheme());

    private void Save(IReadOnlyList<Session> sessions, bool lightTheme)
    {
        var directory = Path.GetDirectoryName(_path);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var json = JsonSerializer.Serialize(
            new SessionFile { Sessions = [.. sessions], LightTheme = lightTheme },
            SessionJsonContext.Default.SessionFile);

        // 直接上書きすると、書いている途中で落ちたときに設定を失う。
        var temporary = _path + ".tmp";
        File.WriteAllText(temporary, json);
        File.Move(temporary, _path, overwrite: true);
    }

    /// <summary>
    /// 同じ名前があれば置き換え、無ければ足す。
    ///
    /// **合言葉は保存しない。** 場所の文字列に書かれた合言葉をそのまま
    /// 設定ファイルへ落とすと、平文で残り続ける（そして本人も忘れる）。
    /// 伏せた形で保存し、使うときに改めて聞く。
    /// </summary>
    public List<Session> Upsert(Session session)
    {
        var sessions = Load();
        sessions.RemoveAll(s => string.Equals(s.Name, session.Name, StringComparison.OrdinalIgnoreCase));
        session = session with
        {
            LeftPath = RemoteLocation.Redact(session.LeftPath),
            RightPath = RemoteLocation.Redact(session.RightPath),
        };
        sessions.Insert(0, session with { LastUsed = DateTime.UtcNow });
        Save(sessions);
        return sessions;
    }

    public List<Session> Remove(string name)
    {
        var sessions = Load();
        sessions.RemoveAll(s => string.Equals(s.Name, name, StringComparison.OrdinalIgnoreCase));
        Save(sessions);
        return sessions;
    }
}
