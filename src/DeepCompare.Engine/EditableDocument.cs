namespace DeepCompare.Engine;

/// <summary>
/// 編集できる行の並びと、その取り消し履歴。
///
/// 履歴に上限は設けない。BC の「無制限の undo」に相当する。保存しても履歴は捨てない
/// ので、保存した後でも遡れる。差分を反映する道具では「入れてみて、やっぱり戻す」が
/// 日常的に起きるため、ここを削ると使い勝手が大きく落ちる。
///
/// 保持するのは取り消しに必要な最小限（消えた行と入れた行数）だけで、全体の複製では
/// ない。1 万行のファイルに 100 回操作しても、控えるのは触った範囲だけで済む。
/// </summary>
public sealed class EditableDocument
{
    private readonly record struct Change(int Start, string[] Removed, string[] Inserted);

    private readonly List<string> _lines;
    private readonly List<Change> _history = [];

    /// <summary>適用済みの操作数。取り消すと減り、やり直すと増える。</summary>
    private int _position;

    /// <summary>最後に保存した時点の <see cref="_position"/>。</summary>
    private int _savedPosition;

    public EditableDocument(IReadOnlyList<string> lines)
    {
        _lines = [.. lines];
    }

    public IReadOnlyList<string> Lines => _lines;

    /// <summary>保存した時点から変わっているか。取り消して戻れば偽に戻る。</summary>
    public bool IsModified => _position != _savedPosition;

    public bool CanUndo => _position > 0;

    public bool CanRedo => _position < _history.Count;

    /// <summary>
    /// 範囲を置き換える。<paramref name="count"/> が 0 なら挿入、
    /// <paramref name="replacement"/> が空なら削除。
    /// </summary>
    public void Replace(int start, int count, IReadOnlyList<string> replacement)
    {
        if (start < 0 || count < 0 || start + count > _lines.Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(start),
                $"置き換える範囲が並びの外にある: 開始 {start} 数 {count} 全体 {_lines.Count}");
        }

        var removed = new string[count];
        _lines.CopyTo(start, removed, 0, count);
        var inserted = replacement.ToArray();

        if (removed.AsSpan().SequenceEqual(inserted))
        {
            // 何も変わらない操作を履歴に積むと、取り消しても見た目が動かず
            // 「効かない undo」に見える。
            return;
        }

        Apply(start, count, inserted);

        // やり直せる分を捨ててから積む。分岐した履歴は持たない。
        if (_position < _history.Count)
        {
            _history.RemoveRange(_position, _history.Count - _position);

            // 分岐前の保存時点へはもう戻れない。以降 IsModified は真のままになる。
            if (_savedPosition > _position)
            {
                _savedPosition = -1;
            }
        }
        _history.Add(new Change(start, removed, inserted));
        _position++;
    }

    public void Undo()
    {
        if (!CanUndo)
        {
            return;
        }
        var change = _history[_position - 1];
        Apply(change.Start, change.Inserted.Length, change.Removed);
        _position--;
    }

    public void Redo()
    {
        if (!CanRedo)
        {
            return;
        }
        var change = _history[_position];
        Apply(change.Start, change.Removed.Length, change.Inserted);
        _position++;
    }

    /// <summary>保存した印を付ける。履歴は捨てないので、保存後も遡れる。</summary>
    public void MarkSaved() => _savedPosition = _position;

    private void Apply(int start, int count, IReadOnlyList<string> replacement)
    {
        _lines.RemoveRange(start, count);
        _lines.InsertRange(start, replacement);
    }
}
