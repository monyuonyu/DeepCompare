using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Markup.Xaml;
using Avalonia.Interactivity;
using Avalonia.VisualTree;

namespace DeepCompare.App.Views;

public partial class TextCompareView : UserControl
{
    private ScrollViewer? _scroll;

    public TextCompareView()
    {
        // **生成される初期化を呼ぶ。** AvaloniaXamlLoader.Load を直に呼ぶと
        // x:Name のフィールドが埋まらない。この画面はこれまで x:Name を
        // 使っていなかったので表に出ていなかった。
        InitializeComponent();

        // 選んだ行を ViewModel へ渡す。**ListBox の SelectedItems は
        // 画面側にしかない。** ViewModel から画面に触らせず、ここで写す。
        if (this.FindControl<ListBox>("RowList") is { } list)
        {
            list.SelectionChanged += (_, _) =>
            {
                if (DataContext is not TextCompareViewModel model)
                {
                    return;
                }
                model.SelectedRows.Clear();
                foreach (var item in list.SelectedItems ?? Array.Empty<object>())
                {
                    if (item is not null)
                    {
                        model.SelectedRows.Add(item);
                    }
                }
                model.SelectionChanged();
            };
        }

        // 本文列の幅は「内容に必要な幅」と「画面を分け合った幅」の大きい方にする。
        // 画面幅は表示側にしか分からないので、変わるたびに知らせる。
        this.GetObservable(BoundsProperty).Subscribe(new AnonymousObserver<Rect>(bounds =>
        {
            if (DataContext is TextCompareViewModel model)
            {
                model.ViewportWidth = bounds.Width;
            }
        }));

        // 書き込み先は表示側にしかない。ViewModel から画面に触らせず、ここで差し込む。
        // **エディタ側へ中身を流す。** 比較が終わるたびに入れ直す。
        DataContextChanged += (_, _) =>
        {
            if (DataContext is TextCompareViewModel model)
            {
                model.AlignedChanged -= OnAlignedChanged;
                model.AlignedChanged += OnAlignedChanged;
                OnAlignedChanged(model, EventArgs.Empty);

                // 矢印を押したら、その塊を写す。
                Editors.ApplyBlock = async (block, toRight) =>
                {
                    if (DataContext is TextCompareViewModel current)
                    {
                        await current.ApplyBlockAsync(block, toRight);
                    }
                };

                // いる行を殻へ伝える。**下の帯と地図がこれで追う。**
                Editors.CaretLineChanged -= OnCaretLine;
                Editors.CaretLineChanged += OnCaretLine;
                Editors.ViewportChanged -= OnViewport;
                Editors.ViewportChanged += OnViewport;

                // 行が変わったら本文をそこへ動かす（地図・次の差分へ・Ctrl+G）。
                model.GoToLineRequested -= OnGoToLine;
                model.GoToLineRequested += OnGoToLine;

                // **打った内容を本文へ戻す。** 詰め物は除いてから渡す。
                Editors.LeftChanged -= OnLeftEdited;
                Editors.LeftChanged += OnLeftEdited;
                Editors.RightChanged -= OnRightEdited;
                Editors.RightChanged += OnRightEdited;
            }
        };

        DataContextChanged += (_, _) =>
        {
            if (DataContext is TextCompareViewModel model)
            {
                model.ReadClipboard = async () =>
                {
                    var clipboard = TopLevel.GetTopLevel(this)?.Clipboard;
                    return clipboard is null
                        ? null
                        : await clipboard.TryGetValueAsync(DataFormat.Text);
                };
                // 段階 2 で行が入れ替わったとき、読んでいた場所へ戻す。
                model.ScrollToRow = index =>
                {
                    var list = this.GetVisualDescendants().OfType<ListBox>()
                        .FirstOrDefault(l => l.ItemsSource == model.VisibleRows);
                    if (list is not null && index >= 0 && index < model.VisibleRows.Count)
                    {
                        list.ScrollIntoView(index);
                    }
                };
                model.Clipboard = text =>
                {
                    var clipboard = TopLevel.GetTopLevel(this)?.Clipboard;
                    if (clipboard is not null)
                    {
                        _ = clipboard.SetValueAsync(DataFormat.Text, text);
                    }
                };
            }
        };

        // 行の中の入力欄で Enter を押したら確定、Esc で取り消す。
        // **入力欄ごとに仕掛けない。** 仮想化で作り直されるので、
        // ここで一括して拾う方が漏れない。
        // **押した側を覚える。** まとめて消す・貼り替えるとき、
        // どちらの本文を書き換えるかがこれで決まる。
        AddHandler(PointerPressedEvent, OnPointerPressedForSide, handledEventsToo: true);

        AddHandler(KeyDownEvent, OnKeyDown, RoutingStrategies.Tunnel);

        // 差分の地図に「いま見えている範囲」を出すために、スクロール位置を拾う。
        // ListBox の中の ScrollViewer は、テンプレートが当たるまで存在しない。
        AttachedToVisualTree += (_, _) => HookScroll();

        // **見つかるまで探し続ける。** 本文の一覧は比較が終わるまで
        // 非表示で、そのあいだ中の ScrollViewer は作られてすらいない。
        // 木に付いたときの一度きりで諦めていたため、地図の「いま見えている
        // 範囲」がずっと初期値のまま（＝全体）になっていた。
        LayoutUpdated += (_, _) => HookScroll();
    }

    /// <summary>
    /// 押した場所から、左右のどちらを触ったかを決める。
    ///
    /// **目印は親をたどって探す。** 押されたのは中の文字や余白なので、
    /// その要素自身は左右を知らない。
    /// </summary>
    private void OnPointerPressedForSide(object? sender, PointerPressedEventArgs e)
    {
        if (DataContext is not TextCompareViewModel model)
        {
            return;
        }
        for (var at = e.Source as Visual; at is not null; at = at.GetVisualParent())
        {
            if (at is Control { Tag: string side })
            {
                if (side == "left")
                {
                    model.ActiveIsLeft = true;
                    return;
                }
                if (side == "right")
                {
                    model.ActiveIsLeft = false;
                    return;
                }
            }
        }
    }

    /// <summary>
    /// 揃えた本文をエディタへ流す。
    ///
    /// **読み取り専用かどうかもここで渡す。** 片側が git の履歴や
    /// クリップボードなら、そちらは打てない。
    /// </summary>
    private void OnAlignedChanged(object? sender, EventArgs e)
    {
        if (DataContext is not TextCompareViewModel model)
        {
            return;
        }
        Editors.Fill(
            model.AlignedLeft, model.AlignedRight,
            model.LeftReadOnly, model.RightReadOnly,
            model.CurrentLanguage);
    }

    /// <summary>
    /// エディタで打たれた内容を本文へ戻す。
    ///
    /// **打つたびに比べ直さない。** 一文字ごとに組み直すと、書いている
    /// 途中で行が動いてカーソルを見失う。少し待ってから比べる。
    /// </summary>
    private void OnLeftEdited(object? sender, EventArgs e) => ScheduleApply(left: true);

    private void OnRightEdited(object? sender, EventArgs e) => ScheduleApply(left: false);

    private CancellationTokenSource? _editDelay;

    private void ScheduleApply(bool left)
    {
        _editDelay?.Cancel();
        _editDelay = new CancellationTokenSource();
        var token = _editDelay.Token;

        _ = Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(async () =>
        {
            try
            {
                // 打鍵が止まるのを待つ。**止まらないうちは比べない。**
                await Task.Delay(400, token);
            }
            catch (TaskCanceledException)
            {
                return;
            }
            if (DataContext is TextCompareViewModel model)
            {
                await model.ApplyEditedLinesAsync(
                    left ? Editors.LeftLines() : Editors.RightLines(), left);
            }
        });
    }

    private void OnCaretLine(object? sender, int line)
    {
        if (DataContext is TextCompareViewModel model)
        {
            model.FollowCaret(line);
        }
    }

    private void OnGoToLine(object? sender, int line) => Editors.GoToLine(line);

    private void OnViewport(object? sender, (double Start, double Size) range)
    {
        if (DataContext is TextCompareViewModel model)
        {
            model.MapViewStart = range.Start;
            model.MapViewSize = range.Size;
        }
    }


    private void OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Source is not TextBox { Classes: var classes } box || !classes.Contains("inline"))
        {
            return;
        }
        if (DataContext is not TextCompareViewModel model
            || box.DataContext is not RowView row)
        {
            return;
        }

        if (e.Key == Key.Enter)
        {
            _ = model.CommitRowEditAsync(row);
            e.Handled = true;
        }
        else if (e.Key == Key.Escape)
        {
            // 打った内容を捨てて元へ戻す。**確定と取り消しは別の鍵にする。**
            row.EditLeft = row.LeftText;
            row.EditRight = row.RightText;
            e.Handled = true;
        }
    }

    private void HookScroll()
    {
        if (_scroll is not null)
        {
            return;
        }
        // **本文の一覧のものに限る。** 上から順に拾うと、先に現れた別の
        // 入れ物の ScrollViewer を掴んでしまう。
        var list = this.FindControl<ListBox>("RowList");
        _scroll = list?.GetVisualDescendants().OfType<ScrollViewer>().FirstOrDefault();
        if (_scroll is null)
        {
            return;
        }

        void Update()
        {
            if (DataContext is not TextCompareViewModel model)
            {
                return;
            }
            // Extent は中身の全体、Viewport は見えている分。差が 0 なら全部見えている。
            var scrollable = _scroll.Extent.Height - _scroll.Viewport.Height;
            model.MapViewStart = scrollable > 0 ? _scroll.Offset.Y / _scroll.Extent.Height : 0;
            model.MapViewSize = _scroll.Extent.Height > 0
                ? Math.Min(1, _scroll.Viewport.Height / _scroll.Extent.Height)
                : 1;
        }

        _scroll.GetObservable(ScrollViewer.OffsetProperty)
            .Subscribe(new AnonymousObserver<Vector>(_ => Update()));
        // 行が入れ替わると中身の高さが変わる。そのときも出し直す。
        _scroll.GetObservable(ScrollViewer.ExtentProperty)
            .Subscribe(new AnonymousObserver<Size>(_ => Update()));
    }

    /// <summary>
    /// 値が来たときだけ何かする観測者。Avalonia は System.Reactive を要求しないので、
    /// この程度は自前で持つ方が依存を増やさずに済む。
    /// </summary>
    private sealed class AnonymousObserver<T>(Action<T> onNext) : IObserver<T>
    {
        public void OnCompleted() { }

        public void OnError(Exception error) { }

        public void OnNext(T value) => onNext(value);
    }
}
