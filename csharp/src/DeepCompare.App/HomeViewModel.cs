using System.Windows.Input;

namespace DeepCompare.App;

/// <summary>
/// 起動画面。何を比べるかをここで決める。
///
/// いきなりファイル比較の画面を出すと「フォルダーを比べたい」ときに行き場が無くなる。
/// Beyond Compare が最初にセッションの種類を選ばせるのと同じ考え方。
/// </summary>
public sealed class HomeViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;
    private string _leftPath = string.Empty;
    private string _rightPath = string.Empty;
    private string _message = string.Empty;

    public HomeViewModel(ShellViewModel shell)
    {
        _shell = shell;
        BrowseLeftFileCommand = new RelayCommand(() => PickAsync(isFolder: false, left: true));
        BrowseRightFileCommand = new RelayCommand(() => PickAsync(isFolder: false, left: false));
        BrowseLeftFolderCommand = new RelayCommand(() => PickAsync(isFolder: true, left: true));
        BrowseRightFolderCommand = new RelayCommand(() => PickAsync(isFolder: true, left: false));
        CompareTextCommand = new RelayCommand(() => { StartText(); return Task.CompletedTask; });
        CompareFoldersCommand = new RelayCommand(() => { StartFolders(); return Task.CompletedTask; });
    }

    public string LeftPath
    {
        get => _leftPath;
        set => Set(ref _leftPath, value);
    }

    public string RightPath
    {
        get => _rightPath;
        set => Set(ref _rightPath, value);
    }

    public string Message
    {
        get => _message;
        private set => Set(ref _message, value);
    }

    public ICommand BrowseLeftFileCommand { get; }
    public ICommand BrowseRightFileCommand { get; }
    public ICommand BrowseLeftFolderCommand { get; }
    public ICommand BrowseRightFolderCommand { get; }
    public ICommand CompareTextCommand { get; }
    public ICommand CompareFoldersCommand { get; }

    private async Task PickAsync(bool isFolder, bool left)
    {
        var title = (isFolder, left) switch
        {
            (true, true) => "左のフォルダーを選択",
            (true, false) => "右のフォルダーを選択",
            (false, true) => "左のファイルを選択",
            (false, false) => "右のファイルを選択",
        };
        if (await _shell.PickPath(title, isFolder) is { } path)
        {
            if (left)
            {
                LeftPath = path;
            }
            else
            {
                RightPath = path;
            }
        }
    }

    private void StartText()
    {
        if (!Validate(out var left, out var right))
        {
            return;
        }
        if (!File.Exists(left) || !File.Exists(right))
        {
            Message = "テキスト比較にはファイルを指定してください。フォルダーなら「フォルダーを比較」を使ってください。";
            return;
        }
        _shell.ShowText(left, right);
    }

    private void StartFolders()
    {
        if (!Validate(out var left, out var right))
        {
            return;
        }
        if (!Directory.Exists(left) || !Directory.Exists(right))
        {
            Message = "フォルダー比較にはフォルダーを指定してください。";
            return;
        }
        _shell.ShowFolders(left, right);
    }

    private bool Validate(out string left, out string right)
    {
        left = LeftPath.Trim();
        right = RightPath.Trim();
        if (left.Length == 0 || right.Length == 0)
        {
            Message = "左右の両方を指定してください。";
            return false;
        }
        Message = string.Empty;
        return true;
    }

    /// <summary>
    /// 落とされたものを受け取る。ファイルかフォルダーかは中身を見て判断できるので、
    /// 利用者に選ばせる必要はない。
    /// </summary>
    public void AcceptDropped(IReadOnlyList<string> paths)
    {
        if (paths.Count == 0)
        {
            return;
        }
        if (paths.Count >= 2)
        {
            LeftPath = paths[0];
            RightPath = paths[1];
        }
        else if (LeftPath.Trim().Length == 0)
        {
            LeftPath = paths[0];
        }
        else if (RightPath.Trim().Length == 0)
        {
            RightPath = paths[0];
        }
        else
        {
            LeftPath = paths[0];
        }

        // 両方揃っていて、どちらもフォルダーなら、そのまま始めてよい。
        var left = LeftPath.Trim();
        var right = RightPath.Trim();
        if (left.Length > 0 && right.Length > 0)
        {
            Message = Directory.Exists(left) && Directory.Exists(right)
                ? "フォルダーが 2 つ揃いました。「フォルダーを比較」を押してください。"
                : File.Exists(left) && File.Exists(right)
                    ? "ファイルが 2 つ揃いました。「テキストを比較」を押してください。"
                    : string.Empty;
        }
    }
}
