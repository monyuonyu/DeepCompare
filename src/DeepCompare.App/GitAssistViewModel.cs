using System.Collections.ObjectModel;
using DeepCompare.Assist;
using DeepCompare.Engine;

namespace DeepCompare.App;

/// <summary>画面に出す提案 1 件。</summary>
public sealed class AssistSuggestionRow(AssistSuggestion suggestion)
{
    public AssistSuggestion Suggestion { get; } = suggestion;

    public string Display => Suggestion.Display;
    public string Reason => Suggestion.Reason;
    public bool Recommended => Suggestion.Recommended;

    /// <summary>作業ツリーに触るか。**触るものは押す前に確かめる。**</summary>
    public bool IsDestructive => AssistActions.IsDestructive(Suggestion.Action);
}

/// <summary>
/// Git 画面の LLM 支援。
///
/// **接続先が無ければ何も出さない。** 起動時に繋ぎに行くこともしない。
/// 出す・出さないの判断をここ 1 か所に集める。
/// </summary>
public sealed class GitAssistViewModel : ViewModelBase
{
    private readonly Func<GitRepository?> _repository;

    public GitAssistViewModel(Func<GitRepository?> repository)
    {
        _repository = repository;
        Settings = LoadSettings();

        ExplainCommand = new RelayCommand(ExplainAsync, () => IsAvailable && !IsBusy);
        DraftCommitCommand = new RelayCommand(DraftAsync, () => IsAvailable && !IsBusy);
    }

    /// <summary>設定を読む。**鍵は設定ファイルではなく環境変数から。**</summary>
    private static AssistSettings LoadSettings()
    {
        var saved = SessionStore.Default.LoadFile();
        return new AssistSettings
        {
            Endpoint = Environment.GetEnvironmentVariable(
                AssistCli.EndpointEnvironmentVariable) ?? saved.AssistEndpoint,
            Model = Environment.GetEnvironmentVariable(
                AssistCli.ModelEnvironmentVariable) ?? saved.AssistModel,
            ApiKey = Environment.GetEnvironmentVariable(AssistCli.ApiKeyEnvironmentVariable),
            AllowResolutionProposals = saved.AssistAllowResolution,
        };
    }

    public AssistSettings Settings { get; private set; }

    /// <summary>
    /// 使えるか。**これが false なら画面に何も出さない。**
    /// 設定していない人にとっては、この機能は存在しないのと同じでよい。
    /// </summary>
    public bool IsAvailable => Settings.IsConfigured;

    private bool _isBusy;
    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (Set(ref _isBusy, value))
            {
                ExplainCommand.Raise();
                DraftCommitCommand.Raise();
            }
        }
    }

    private string _explanation = string.Empty;
    public string Explanation
    {
        get => _explanation;
        private set
        {
            if (Set(ref _explanation, value))
            {
                OnPropertyChanged(nameof(HasExplanation));
            }
        }
    }

    public bool HasExplanation => Explanation.Length > 0;

    public ObservableCollection<AssistSuggestionRow> Suggestions { get; } = [];

    public RelayCommand ExplainCommand { get; }
    public RelayCommand DraftCommitCommand { get; }

    /// <summary>
    /// 草案ができたら呼ぶ。**入力欄へ入れるだけ。**
    /// そのまま記録はしない — 人が読んで直す前提。
    /// </summary>
    public Action<string>? CommitDraftHandler { get; set; }

    private async Task ExplainAsync()
    {
        var repository = _repository();
        if (repository is null)
        {
            return;
        }

        IsBusy = true;
        Explanation = "考えています…";
        Suggestions.Clear();

        try
        {
            // **git を呼ぶのは画面側。** Assist に副作用を持たせない。
            var status = await Task.Run(() => AssistCli.FormatStatus(repository));

            using var client = new ChatClient(Settings);
            var advice = await new GitAssistant(client).ExplainStatusAsync(
                status, maxTokens: Settings.ExplainMaxTokens);

            Explanation = advice.Explanation;
            foreach (var suggestion in advice.Suggestions)
            {
                Suggestions.Add(new AssistSuggestionRow(suggestion));
            }
        }
        catch (Exception error) when (error is AssistException or GitException
                                        or ArgumentException)
        {
            // **比較の操作まで止めない。** 支援はあると助かるもので、
            // 失敗しても道具そのものは使える。
            Explanation = error.Message;
        }
        finally
        {
            IsBusy = false;
        }
    }

    private async Task DraftAsync()
    {
        var repository = _repository();
        if (repository is null)
        {
            return;
        }

        IsBusy = true;
        try
        {
            var diff = await Task.Run(() => repository.Diff(staged: true));
            if (diff.Trim().Length == 0)
            {
                // **stage されていない状態で書かせない。** 何か出てくるが中身は無い。
                Explanation = "記録の準備ができているファイルがありません。"
                    + "先に stage してください。";
                return;
            }

            using var client = new ChatClient(Settings);
            var assistant = new GitAssistant(client);

            var draft = new System.Text.StringBuilder();
            await foreach (var piece in assistant.DraftCommitMessageAsync(diff))
            {
                draft.Append(piece);
                // **少しずつ入れる。** ローカルのモデルは遅いので、
                // 出来上がるまで黙っていると止まって見える。
                CommitDraftHandler?.Invoke(draft.ToString());
            }
        }
        catch (Exception error) when (error is AssistException or GitException
                                        or ArgumentException)
        {
            Explanation = error.Message;
        }
        finally
        {
            IsBusy = false;
        }
    }
}
