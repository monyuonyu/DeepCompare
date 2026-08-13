using System.Collections.ObjectModel;
using Avalonia.Media;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>写しの比較の 1 行。</summary>
public sealed class SnapshotRowView(FolderEntry entry)
{
    public FolderEntry Entry { get; } = entry;

    public string Path => Entry.RelativePath;

    /// <summary>何が起きたか。**言葉にする。** 記号だけだと向きを取り違える。</summary>
    public string What => Entry.Status switch
    {
        EntryStatus.Different => "変わった",
        EntryStatus.LeftOnly => "消えた",
        EntryStatus.RightOnly => "増えた",
        _ => "同じ",
    };

    public IBrush WhatBrush => Palette.Brush(Entry.Status switch
    {
        EntryStatus.Different => "GitChanged",
        EntryStatus.LeftOnly => "GitRemoved",
        EntryStatus.RightOnly => "GitAdded",
        _ => "FgDim",
    });

    public IBrush Background => Palette.Brush(Entry.Status switch
    {
        EntryStatus.Different => "BgChanged",
        EntryStatus.LeftOnly => "BgRemoved",
        EntryStatus.RightOnly => "BgAdded",
        _ => "CardBg",
    });

    /// <summary>大きさの変化。**増減を符号で出す。** 数字を 2 つ並べても差は頭で引くことになる。</summary>
    public string SizeChange => (Entry.LeftSize, Entry.RightSize) switch
    {
        (null, { } after) => $"+{after:N0} B",
        ({ } before, null) => $"-{before:N0} B",
        ({ } before, { } after) when before != after =>
            $"{after - before:+#,0;-#,0} B（{before:N0} → {after:N0}）",
        _ => string.Empty,
    };

    public bool IsDirectory => Entry.IsDirectory;
}

/// <summary>
/// 写しを取り、後の姿と比べる画面（BC の Snapshot に当たる）。
///
/// **写しの読み書きは engine に置く。** ここは指示を受けて結果を並べるだけ。
/// </summary>
public sealed class SnapshotViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;

    public CompareTab? Tab { get; set; }
    public ShellViewModel Shell => _shell;

    public SnapshotViewModel(ShellViewModel shell)
    {
        _shell = shell;

        TakeCommand = new RelayCommand(TakeAsync,
            () => !_busy && FolderPath.Trim().Length > 0);
        CompareCommand = new RelayCommand(CompareAsync,
            () => !_busy && SnapshotPath.Trim().Length > 0);
        BrowseFolderCommand = new RelayCommand(async () =>
        {
            if (await _shell.PickPath("写し取るフォルダー", true) is { } picked)
            {
                FolderPath = picked;
            }
        });
        BrowseSnapshotCommand = new RelayCommand(async () =>
        {
            if (await _shell.PickPath("比べる写し", false) is { } picked)
            {
                SnapshotPath = picked;
            }
        });
    }

    public RelayCommand TakeCommand { get; }
    public RelayCommand CompareCommand { get; }
    public RelayCommand BrowseFolderCommand { get; }
    public RelayCommand BrowseSnapshotCommand { get; }

    public ObservableCollection<SnapshotRowView> Rows { get; } = [];

    private string _folderPath = string.Empty;
    public string FolderPath
    {
        get => _folderPath;
        set { if (Set(ref _folderPath, value)) { TakeCommand.Raise(); } }
    }

    private string _snapshotPath = string.Empty;

    /// <summary>比べる相手の写し。取ったばかりのものは自動で入る。</summary>
    public string SnapshotPath
    {
        get => _snapshotPath;
        set { if (Set(ref _snapshotPath, value)) { CompareCommand.Raise(); } }
    }

    private bool _withHashes = true;

    /// <summary>
    /// 指紋も取るか。
    ///
    /// **既定で取る。** 取らないと、同じ秒の内の書き換えを見分けられない。
    /// 数万ファイルで時間がかかるときだけ切る、という順序にする。
    /// </summary>
    public bool WithHashes
    {
        get => _withHashes;
        set => Set(ref _withHashes, value);
    }

    private bool _showIdentical;

    /// <summary>変化のないものも出すか。既定は隠す。</summary>
    public bool ShowIdentical
    {
        get => _showIdentical;
        set { if (Set(ref _showIdentical, value)) { Rebuild(); } }
    }

    private string _summary = string.Empty;
    public string Summary
    {
        get => _summary;
        private set => Set(ref _summary, value);
    }

    private string _message = string.Empty;
    public string Message
    {
        get => _message;
        private set => Set(ref _message, value);
    }

    /// <summary>
    /// 指紋なしの写しで比べている。
    ///
    /// **黙っていられない。** 「変化なし」が「大きさと時刻に変化なし」の
    /// 意味になっているのを伝えないのは、嘘に近い。
    /// </summary>
    private bool _withoutHashes;
    public bool WithoutHashes
    {
        get => _withoutHashes;
        private set => Set(ref _withoutHashes, value);
    }

    private bool _busy;
    public bool Busy
    {
        get => _busy;
        private set
        {
            if (Set(ref _busy, value))
            {
                TakeCommand.Raise();
                CompareCommand.Raise();
            }
        }
    }

    private FolderComparison? _comparison;

    private async Task TakeAsync()
    {
        var root = FolderPath.Trim();
        if (!Directory.Exists(root))
        {
            Message = $"{root} はフォルダーではありません。";
            return;
        }

        var target = await _shell.PickSavePath("写しの書き出し先",
            System.IO.Path.GetFileName(root.TrimEnd(System.IO.Path.DirectorySeparatorChar))
            + ".dcsnap");
        if (target is null)
        {
            return;
        }

        Busy = true;
        Message = string.Empty;
        try
        {
            var hashes = WithHashes;
            var snapshot = await Task.Run(() => Snapshots.Take(root, withHashes: hashes));
            await File.WriteAllTextAsync(target, Snapshots.Save(snapshot));

            // 取った写しをそのまま比較の相手に据える。**次にやることが決まって
            // いるなら、入れ直させない。**
            SnapshotPath = target;
            Summary = $"{snapshot.FileCount} ファイル / {snapshot.DirectoryCount} フォルダーを写しました"
                + (snapshot.HasHashes ? "（指紋あり）" : "（指紋なし）");
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Message = error.Message;
        }
        finally
        {
            Busy = false;
        }
    }

    private async Task CompareAsync()
    {
        Busy = true;
        Message = string.Empty;
        Rows.Clear();

        try
        {
            var path = SnapshotPath.Trim();
            var before = Snapshots.Load(await File.ReadAllTextAsync(path));

            // 比べる先は、画面で指定されていればそこ。空なら写しに書かれた元の場所。
            var root = FolderPath.Trim().Length > 0 ? FolderPath.Trim() : before.Root;
            if (!Directory.Exists(root))
            {
                Message = $"{root} が見つかりません。比べる先のフォルダーを指定してください。";
                return;
            }

            // **写しと同じ条件で取り直す。** 片方だけ指紋があると、中身の変化を
            // 見分けられたり見分けられなかったりして食い違う。
            var hashes = before.HasHashes;
            var after = await Task.Run(() => Snapshots.Take(root, withHashes: hashes));

            _comparison = Snapshots.Compare(before, after);
            WithoutHashes = !hashes;
            FolderPath = root;
            Rebuild();

            var stats = _comparison.Stats;
            Summary = $"{before.TakenAt:yyyy-MM-dd HH:mm} と今 — "
                + $"変わった {stats.Different} / 消えた {stats.LeftOnly} / 増えた {stats.RightOnly}"
                + $"（同じ {stats.Identical}）";

            if (Tab is { } tab)
            {
                tab.Title = System.IO.Path.GetFileName(root) + "（写し）";
            }
        }
        catch (Exception error) when (error is IOException or InvalidDataException
                                        or UnauthorizedAccessException)
        {
            Message = error.Message;
            _comparison = null;
        }
        finally
        {
            Busy = false;
        }
    }

    private void Rebuild()
    {
        Rows.Clear();
        if (_comparison is null)
        {
            return;
        }
        foreach (var entry in _comparison.Entries)
        {
            if (!ShowIdentical && entry.Status == EntryStatus.Identical)
            {
                continue;
            }
            Rows.Add(new SnapshotRowView(entry));
        }
    }
}
