using DeepCompare.Assist;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>選べる表示言語 1 つ。</summary>
public sealed record UiLanguage(string Code, string Label);

/// <summary>
/// 設定の画面。
///
/// **散らばっていた設定を 1 か所に集める。** これまでテーマは画面の隅、
/// モデルは起動画面、LLM の接続先は設定ファイルと環境変数にしか無く、
/// 「どこで変えるのか」が分からなかった。
///
/// **変えたらすぐ保存する。** 「適用」を押し忘れて戻る、が起きない。
/// </summary>
public sealed class SettingsViewModel : ViewModelBase
{
    private readonly ShellViewModel _shell;
    private readonly SessionStore _store = SessionStore.Default;

    public SettingsViewModel(ShellViewModel shell)
    {
        _shell = shell;

        var saved = _store.LoadFile();
        _assistEndpoint = saved.AssistEndpoint;
        _assistModel = saved.AssistModel;
        _assistAllowResolution = saved.AssistAllowResolution;
        _language = Languages.FirstOrDefault(l => l.Code == saved.Language) ?? Languages[0];

        ProbeAssistCommand = new RelayCommand(ProbeAssistAsync, () => AssistConfigured);
        SetUpModels();
    }

    public ShellViewModel Shell => _shell;
    public CompareTab? Tab { get; set; }

    // ---- 見た目 ----

    /// <summary>明るいテーマか。**殻が持っている値をそのまま見せる。**</summary>
    public bool LightTheme
    {
        get => _shell.LightTheme;
        set
        {
            if (_shell.LightTheme != value)
            {
                _shell.LightTheme = value;
                OnPropertyChanged();
            }
        }
    }

    /// <summary>
    /// 選べる表示言語。
    ///
    /// **いまは日本語だけ。** 文言をコードに直接書いているので、
    /// 他の言語を足すには全部を資源へ移す作業が要る。
    /// ここに枠だけ置いておくのは、**設定の置き場所を先に決めておく**ため
    /// （後から場所を変えると、覚えた設定が読めなくなる）。
    /// </summary>
    public IReadOnlyList<UiLanguage> Languages { get; } =
    [
        new("ja", "日本語"),
    ];

    private UiLanguage _language;
    public UiLanguage Language
    {
        get => _language;
        set
        {
            if (Set(ref _language, value))
            {
                _store.SaveLanguage(value.Code);
            }
        }
    }

    /// <summary>**まだ 1 つしか無いことを言う。** 黙って選べないより分かる。</summary>
    public bool HasOtherLanguages => Languages.Count > 1;

    // ---- モデル ----

    /// <summary>実行ファイルの隣にあるモデル。</summary>
    public IReadOnlyList<string> AvailableModels => _shell.AvailableModels;

    public string ModelName
    {
        get => _shell.ModelName;
        set
        {
            if (_shell.ModelName != value)
            {
                _shell.ModelName = value;
                OnPropertyChanged();
            }
        }
    }

    public bool CanChooseModel => _shell.CanChooseModel;

    /// <summary>モデルが 1 つでも在るか。</summary>
    public bool HasModel => _shell.AvailableModels.Count > 0;

    /// <summary>
    /// いまの状態を 1 行で。**「効いていない」を黙らない。**
    ///
    /// モデルは配布物に含めていないので、無いのが初期状態。そのとき
    /// 比較は動くが、行の対応付けは文字の一致で決まっている。
    /// 何も言わないと、意味で並んだ結果だと受け取られる。
    /// </summary>
    public string ModelStatus
    {
        get
        {
            if (!HasModel)
            {
                return "**モデルがありません。** 行の対応付けは文字の一致で"
                    + "決めています（普通の diff と同じ）。";
            }
            var path = Embedder.ResolveModelPath();
            if (!File.Exists(path))
            {
                return $"選ばれている {Path.GetFileName(path)} が見つかりません。";
            }
            var size = new FileInfo(path).Length / (1024.0 * 1024.0);
            return $"{Path.GetFileName(path)}（{size:0.#}MB）で意味的に対応付けています。";
        }
    }

    /// <summary>置き場所。**どこへ置けばいいか分からない、を無くす。**</summary>
    public string ModelLocation => AppContext.BaseDirectory;

    /// <summary>モデルが無いときに出す入手方法。</summary>
    public string ModelHint
        => "tools/fetch-model.sh を実行するか、Releases の models-v1 から"
           + $" .dcm と .vocab を対で落として、上の場所へ置いてください。";

    /// <summary>置いた後に押す。**動かしている最中に置かれるので要る。**</summary>
    public RelayCommand RefreshModelsCommand { get; private set; } = null!;

    private void SetUpModels()
    {
        RefreshModelsCommand = new RelayCommand(() =>
        {
            _shell.RefreshModels();
            return Task.CompletedTask;
        });
        _shell.ModelsChanged += () =>
        {
            OnPropertyChanged(nameof(HasModel));
            OnPropertyChanged(nameof(ModelStatus));
            OnPropertyChanged(nameof(CanChooseModel));
            OnPropertyChanged(nameof(AvailableModels));
            OnPropertyChanged(nameof(ModelName));
        };
    }

    // ---- LLM 支援 ----

    private string _assistEndpoint;

    /// <summary>接続先。**空なら支援の機能そのものが出ない。**</summary>
    public string AssistEndpoint
    {
        get => _assistEndpoint;
        set
        {
            if (Set(ref _assistEndpoint, value))
            {
                _store.SaveAssist(value, AssistModel, AssistAllowResolution);
                OnPropertyChanged(nameof(AssistConfigured));
                ProbeAssistCommand.Raise();
            }
        }
    }

    private string _assistModel;
    public string AssistModel
    {
        get => _assistModel;
        set
        {
            if (Set(ref _assistModel, value))
            {
                _store.SaveAssist(AssistEndpoint, value, AssistAllowResolution);
                OnPropertyChanged(nameof(AssistConfigured));
                ProbeAssistCommand.Raise();
            }
        }
    }

    private bool _assistAllowResolution;

    /// <summary>
    /// 衝突の解決案まで出してよいか。
    ///
    /// **既定は切ってある。** 説明と違い、意味を取り違えると害になる生成で、
    /// 7B でも引数を落としたまま構文としては正しいコードを出してくる。
    /// </summary>
    public bool AssistAllowResolution
    {
        get => _assistAllowResolution;
        set
        {
            if (Set(ref _assistAllowResolution, value))
            {
                _store.SaveAssist(AssistEndpoint, AssistModel, value);
            }
        }
    }

    public bool AssistConfigured
        => AssistEndpoint.Trim().Length > 0 && AssistModel.Trim().Length > 0;

    public RelayCommand ProbeAssistCommand { get; }

    private string _assistStatus = string.Empty;
    public string AssistStatus
    {
        get => _assistStatus;
        private set => Set(ref _assistStatus, value);
    }

    /// <summary>繋がるかを確かめる。**短い時限で諦める。**</summary>
    private async Task ProbeAssistAsync()
    {
        AssistStatus = "試しています…";
        try
        {
            var settings = new AssistSettings
            {
                Endpoint = AssistEndpoint.Trim(),
                Model = AssistModel.Trim(),
                ApiKey = Environment.GetEnvironmentVariable(AssistCli.ApiKeyEnvironmentVariable),
            };
            using var client = new ChatClient(settings);
            AssistStatus = await client.ProbeAsync()
                ? "繋がりました。"
                : "繋がりません。相手が動いているか確かめてください。";
        }
        catch (Exception error) when (error is AssistException or ArgumentException)
        {
            AssistStatus = error.Message;
        }
    }

    /// <summary>設定ファイルの置き場所。**どこを見ればいいか分かるように出す。**</summary>
    public string SettingsPath => SessionStore.DefaultPath();
}
