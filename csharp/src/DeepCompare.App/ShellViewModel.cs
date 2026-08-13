using System.Collections.ObjectModel;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

public enum CompareKindId
{
    Text,
    Folder,
    Structured,
    Merge,
    Git,
    Image,
    VersionInfo,
    Snapshot,
}

/// <summary>
/// 起動画面に並べる「比較の種類」1 つ。
///
/// Beyond Compare の Home view と同じ形にする。大きな絵と短い名前を並べ、
/// そこへファイルを落とせばその種類で始まる。
///
/// **画面そのものは持たない。** 選ぶたびに新しいタブを作るので、
/// ここが持つのは「どの種類か」だけ。
/// </summary>
public sealed class CompareKind(string label, string hint, string iconKey, CompareKindId id)
{
    public string Label { get; } = label;

    /// <summary>指したときに出す一言。</summary>
    public string Hint { get; } = hint;

    public Geometry? Icon => Icons.Get(iconKey);

    public CompareKindId Id { get; } = id;
}

/// <summary>
/// 開いている比較 1 枚分。タブの見出しと中身を持つ。
/// </summary>
public sealed class CompareTab(string title, object content, bool canClose = true) : ViewModelBase
{
    private string _title = title;

    /// <summary>タブの見出し。比較するものが決まったら、そのファイル名に変わる。</summary>
    public string Title
    {
        get => _title;
        set => Set(ref _title, value);
    }

    private string _tooltip = string.Empty;

    /// <summary>見出しだけでは分からないので、指したら全体を出す。</summary>
    public string Tooltip
    {
        get => _tooltip;
        set => Set(ref _tooltip, value);
    }

    public object Content { get; } = content;

    /// <summary>起動画面は閉じられない。全部閉じると何もできなくなる。</summary>
    public bool CanClose { get; } = canClose;
}

/// <summary>
/// 画面全体の入れ物。
///
/// **タブで複数の比較を同時に開く。** Beyond Compare と同じ。フォルダー比較の
/// 一覧から次々にファイルを開くとき、1 枚しか持てないと前の比較が消える。
/// 見比べながら直す作業では、開いたまま行き来できることが効く。
///
/// タブは閉じるまで生き続けるので、切り替えても入力もスクロール位置も残る。
/// </summary>
public sealed class ShellViewModel : ViewModelBase
{
    private readonly SessionStore _settings = new();
    private bool _lightTheme;

    /// <summary>
    /// モデルは読み込みに数秒かかるうえ 90MB 近い実体を持つので、
    /// 画面を移るたびに作り直さず、ここで一度だけ用意して使い回す。
    /// </summary>
    private Embedder? _embedder;

    public ShellViewModel(
        Func<string, bool, Task<string?>> pickPath,
        Func<string, string, Task<string?>> pickSavePath)
    {
        PickPath = pickPath;
        PickSavePath = pickSavePath;
        _lightTheme = _settings.LoadLightTheme();
        Palette.Use(_lightTheme);
        ToggleThemeCommand = new RelayCommand(() => { LightTheme = !LightTheme; return Task.CompletedTask; });
        GoHomeCommand = new RelayCommand(() => { GoHome(); return Task.CompletedTask; });

        Home = new HomeViewModel(this);
        HomeTab = new CompareTab("ホーム", Home, canClose: false);
        Tabs = [HomeTab];
        _selected = HomeTab;

        Kinds =
        [
            new CompareKind("テキスト比較", "2 つのファイルを行で突き合わせる", "IconText",
                CompareKindId.Text),
            new CompareKind("フォルダー比較", "2 つのフォルダーを再帰的に比べる", "IconFolder",
                CompareKindId.Folder),
            new CompareKind("構造として比較", "JSON を構造として比べる。キーの順序は差分にしない",
                "IconStructure", CompareKindId.Structured),
            new CompareKind("3 方向マージ", "共通の元から分かれた 2 つの変更を合わせる", "IconMerge",
                CompareKindId.Merge),
            new CompareKind("画像比較", "画素で比べる。大きさが違っても重なる範囲は比べる",
                "IconImage", CompareKindId.Image),
            new CompareKind("版の比較", "実行ファイルの版・会社名・説明を並べる",
                "IconVersion", CompareKindId.VersionInfo),
            new CompareKind("写しと比べる", "いまの状態を保存し、後の時点と比べる",
                "IconCamera", CompareKindId.Snapshot),
            new CompareKind("Git", "作業ツリーと履歴", "IconGit", CompareKindId.Git),
        ];

        CloseTabCommand = new RelayCommand<CompareTab>(
            tab => { Close(tab); return Task.CompletedTask; }, tab => tab.CanClose);
        CloseOthersCommand = new RelayCommand<CompareTab>(
            tab => { CloseOthers(tab); return Task.CompletedTask; });
        NewTabCommand = new RelayCommand(() => { Select(HomeTab); return Task.CompletedTask; });
    }

    public Func<string, bool, Task<string?>> PickPath { get; }

    /// <summary>書き出し先を選ばせる。（題名、既定のファイル名）を受ける。</summary>
    public Func<string, string, Task<string?>> PickSavePath { get; }

    public ObservableCollection<CompareKind> Kinds { get; }

    public HomeViewModel Home { get; }
    public CompareTab HomeTab { get; }
    public ObservableCollection<CompareTab> Tabs { get; }

    private CompareTab _selected;
    public CompareTab Selected
    {
        get => _selected;
        set
        {
            // TabControl は入れ替えの途中で選択を外すことがある。null は無視する。
            if (value is not null)
            {
                Set(ref _selected, value);
            }
        }
    }

    public System.Windows.Input.ICommand ToggleThemeCommand { get; }
    public System.Windows.Input.ICommand GoHomeCommand { get; }
    public RelayCommand<CompareTab> CloseTabCommand { get; }
    public RelayCommand<CompareTab> CloseOthersCommand { get; }
    public System.Windows.Input.ICommand NewTabCommand { get; }

    private void Select(CompareTab tab) => Selected = tab;

    /// <summary>起動画面のタブへ移る。**開いている比較は閉じない。**</summary>
    public void GoHome() => Select(HomeTab);

    private CompareTab Add(string title, object content, string tooltip = "")
    {
        var tab = new CompareTab(title, content) { Tooltip = tooltip };
        Tabs.Add(tab);
        Selected = tab;
        return tab;
    }

    private void Close(CompareTab tab)
    {
        if (!tab.CanClose)
        {
            return;
        }
        var index = Tabs.IndexOf(tab);
        Tabs.Remove(tab);

        // 閉じた後は隣へ移る。何も選ばれていない状態を残さない。
        if (ReferenceEquals(_selected, tab) || !Tabs.Contains(_selected))
        {
            Selected = Tabs[Math.Clamp(index - 1, 0, Tabs.Count - 1)];
        }
    }

    private void CloseOthers(CompareTab keep)
    {
        foreach (var tab in Tabs.Where(t => t.CanClose && !ReferenceEquals(t, keep)).ToList())
        {
            Tabs.Remove(tab);
        }
        Selected = Tabs.Contains(keep) ? keep : HomeTab;
    }

    /// <summary>左右のファイル名から見出しを作る。同じ名前なら 1 つで足りる。</summary>
    private static string TitleFor(string left, string right)
    {
        var l = System.IO.Path.GetFileName(left.TrimEnd('/', '\\'));
        var r = System.IO.Path.GetFileName(right.TrimEnd('/', '\\'));
        if (l.Length == 0 && r.Length == 0)
        {
            return "比較";
        }
        return string.Equals(l, r, StringComparison.Ordinal) ? l : $"{l} ↔ {r}";
    }

    /// <summary>
    /// 明るいテーマを使うか。切り替えたら、色を持っている行を作り直す必要がある。
    /// C# 側で組み立てた文字色は、テーマが変わっても勝手には追随しない。
    /// </summary>
    public bool LightTheme
    {
        get => _lightTheme;
        set
        {
            if (!Set(ref _lightTheme, value))
            {
                return;
            }
            Palette.Use(value);
            _settings.SaveLightTheme(value);
            OnPropertyChanged(nameof(ThemeIcon));
            OnPropertyChanged(nameof(ThemeTooltip));
            ThemeChanged?.Invoke();
        }
    }

    /// <summary>いま押すと何になるか。押した先を示す方が、迷わない。</summary>
    public Geometry? ThemeIcon => Icons.Get(_lightTheme ? "IconMoon" : "IconSun");

    public string ThemeTooltip => _lightTheme ? "暗い配色にする" : "明るい配色にする";

    /// <summary>テーマが変わったときに、開いている画面が行を作り直すための知らせ。</summary>
    public event Action? ThemeChanged;

    /// <summary>初回だけ読む。呼び出し側は必ず作業スレッドから呼ぶこと。</summary>
    public Embedder GetEmbedder() => _embedder ??= Embedder.CreateFromDefaultAssets();

    // --- 画面を開く。**どれも新しいタブを作る** ---

    public TextCompareViewModel ShowText(string left, string right)
    {
        var model = new TextCompareViewModel(this) { LeftPath = left, RightPath = right };
        var tab = Add(TitleFor(left, right), model, $"{left}\n{right}");
        model.Tab = tab;
        model.CompareCommand.Execute(null);
        return model;
    }

    public void ShowFolders(string left, string right)
    {
        var model = new FolderCompareViewModel(this) { LeftRoot = left, RightRoot = right };
        var tab = Add(TitleFor(left, right), model, $"{left}\n{right}");
        model.Tab = tab;
        model.RefreshCommand.Execute(null);
    }

    public void ShowStructured(string left, string right)
    {
        var model = new StructuredCompareViewModel(this) { LeftPath = left, RightPath = right };
        var tab = Add(TitleFor(left, right), model, $"{left}\n{right}");
        model.Tab = tab;
        if (left.Length > 0 && right.Length > 0)
        {
            model.CompareCommand.Execute(null);
        }
    }

    public void ShowMerge(string basePath, string left, string right)
    {
        var model = new MergeViewModel(this)
        {
            BasePath = basePath,
            LeftPath = left,
            RightPath = right,
        };
        var tab = Add(TitleFor(left, right), model, $"{basePath}\n{left}\n{right}");
        model.Tab = tab;
        if (basePath.Length > 0 && left.Length > 0 && right.Length > 0)
        {
            model.MergeCommand.Execute(null);
        }
    }

    /// <summary>
    /// git の競合を解く。
    ///
    /// 索引に積まれた 3 つ（祖先・こちら・むこう）をそのまま渡す。
    /// **一時ファイルは作らない。** 中身の供給を差し替えられるので要らない。
    /// </summary>
    public void ShowGitConflict(
        string relativePath,
        Func<string, byte[]> loader,
        Func<IReadOnlyList<string>, Task> save)
    {
        // 名前は「どの側か」が分かる形にする。パスだけだと 3 つとも同じになる。
        var model = new MergeViewModel(this)
        {
            BasePath = $"共通の祖先:{relativePath}",
            LeftPath = $"こちら:{relativePath}",
            RightPath = $"むこう:{relativePath}",
            ContentLoader = loader,
            SaveHandler = save,
            SaveLabel = "解決したことにする",
        };
        var tab = Add($"競合 {System.IO.Path.GetFileName(relativePath)}", model, relativePath);
        model.Tab = tab;
        model.MergeCommand.Execute(null);
    }

    /// <summary>
    /// クリップボードの中身と比べる（BC の File &gt; Open Clipboard）。
    ///
    /// **一時ファイルを作らない。** 読み込み方を差し替える仕掛けが既にあるので、
    /// 中身をそのまま渡せる。後始末も要らず、画面に一時パスも出ない。
    /// </summary>
    public void ShowTextAgainstClipboard(string path, string clipboard)
    {
        var bytes = new System.Text.UTF8Encoding(false).GetBytes(clipboard);
        ShowTextWith(
            path, "（クリップボード）",
            p => p == "（クリップボード）" ? bytes : File.ReadAllBytes(p),
            leftReadOnly: false, rightReadOnly: true);
    }

    /// <summary>写しの画面。フォルダーは空でも開ける（画面で指定できる）。</summary>
    public void ShowSnapshot(string folder, string snapshotFile = "")
    {
        var model = new SnapshotViewModel(this)
        {
            FolderPath = folder,
            SnapshotPath = snapshotFile,
        };
        var tab = Add(folder.Length > 0
            ? System.IO.Path.GetFileName(folder.TrimEnd(System.IO.Path.DirectorySeparatorChar)) + "（写し）"
            : "写し", model, folder);
        model.Tab = tab;

        // 写しが指定されていれば、そのまま比べる。**開いてから押させない。**
        if (snapshotFile.Length > 0)
        {
            model.CompareCommand.Execute(null);
        }
    }

    /// <summary>
    /// ノートブックや Office 文書の**本文だけ**を取り出して、テキスト比較で見る。
    ///
    /// **新しい画面を作らない。** 本文さえ取り出せれば、意味的な行の対応付けも
    /// 折り畳みも検索も、既存の画面のものがそのまま効く。
    /// 取り出したものなので**両側とも読み取り専用**（書き戻せない）。
    /// </summary>
    public void ShowExtractedText(string left, string right)
    {
        ShowTextWith(left, right, ExtractText, leftReadOnly: true, rightReadOnly: true);
    }

    /// <summary>その形式から読める本文を取り出す。読めない形式ならそのまま返す。</summary>
    public static byte[] ExtractText(string path)
    {
        if (Notebook.LooksLikeNotebook(path))
        {
            var document = Notebook.Read(File.ReadAllText(path));
            var text = new System.Text.StringBuilder();
            foreach (var cell in document.Cells)
            {
                // どの種類のセルかを頭に付ける。**本文だけ並べると、
                // markdown とコードの境目が分からない。**
                text.Append("# --- ").Append(cell.Kind switch
                {
                    CellKind.Code => "コード",
                    CellKind.Markdown => "Markdown",
                    _ => "生",
                }).Append(" ---\n");
                text.Append(cell.Source.TrimEnd('\n')).Append("\n\n");
            }
            return new System.Text.UTF8Encoding(false).GetBytes(text.ToString());
        }

        if (OfficeDocument.LooksLikeOffice(path))
        {
            var content = OfficeDocument.Read(path);
            return new System.Text.UTF8Encoding(false)
                .GetBytes(string.Join('\n', content.ToLines()) + "\n");
        }

        return File.ReadAllBytes(path);
    }

    /// <summary>本文を取り出して見るべき形式か。</summary>
    public static bool CanExtractText(string path)
        => Notebook.LooksLikeNotebook(path) || OfficeDocument.LooksLikeOffice(path);

    public void ShowVersionInfo(string left, string right)
    {
        var model = new VersionCompareViewModel(this) { LeftPath = left, RightPath = right };
        var tab = Add(TitleFor(left, right) + "（版）", model, $"{left}\n{right}");
        model.Tab = tab;
        if (left.Length > 0 && right.Length > 0)
        {
            model.CompareCommand.Execute(null);
        }
    }

    public void ShowImage(string left, string right)
    {
        var model = new ImageCompareViewModel(this) { LeftPath = left, RightPath = right };
        var tab = Add(TitleFor(left, right) + "（画像）", model, $"{left}\n{right}");
        model.Tab = tab;
        if (left.Length > 0 && right.Length > 0)
        {
            model.CompareCommand.Execute(null);
        }
    }

    /// <summary>バイト列として比べる。テキストとして読めないファイル向け。</summary>
    public void ShowBinary(string left, string right)
    {
        var model = new BinaryCompareViewModel(this) { LeftPath = left, RightPath = right };
        var tab = Add(TitleFor(left, right) + "（16 進）", model, $"{left}\n{right}");
        model.Tab = tab;
        if (left.Length > 0 && right.Length > 0)
        {
            model.CompareCommand.Execute(null);
        }
    }

    public void ShowGit(string path)
    {
        var model = new GitViewModel(this, path);
        var tab = Add("Git", model, path);
        model.Tab = tab;
        _ = model.RefreshAsync();
    }

    /// <summary>起動画面の大きな絵から。左右が空でも画面は開く。</summary>
    public void Open(CompareKind kind, string left, string right, string basePath = "")
    {
        switch (kind.Id)
        {
            case CompareKindId.Text:
                ShowText(left, right);
                break;
            case CompareKindId.Folder:
                ShowFolders(left, right);
                break;
            case CompareKindId.Structured:
                ShowStructured(left, right);
                break;
            case CompareKindId.Merge:
                ShowMerge(basePath, left, right);
                break;
            case CompareKindId.Image:
                ShowImage(left, right);
                break;
            case CompareKindId.VersionInfo:
                ShowVersionInfo(left, right);
                break;
            case CompareKindId.Snapshot:
                ShowSnapshot(left, right);
                break;
            case CompareKindId.Git:
                ShowGit(left.Length > 0 ? left : Environment.CurrentDirectory);
                break;
        }
    }

    /// <summary>
    /// 中身の取り出し方を差し替えたテキスト比較を、新しいタブで出す。
    /// git のある時点の中身を比べるときに使う。
    /// </summary>
    public void ShowTextWith(
        string leftLabel, string rightLabel, Func<string, byte[]> loader,
        bool leftReadOnly, bool rightReadOnly)
    {
        var model = new TextCompareViewModel(this)
        {
            ContentLoader = loader,
            LeftReadOnly = leftReadOnly,
            RightReadOnly = rightReadOnly,
            LeftPath = leftLabel,
            RightPath = rightLabel,
        };
        var tab = Add(TitleFor(leftLabel, rightLabel), model, $"{leftLabel}\n{rightLabel}");
        model.Tab = tab;
        model.CompareCommand.Execute(null);
    }

}
