using Xunit;
using DeepCompare.Engine;

namespace DeepCompare.Engine.Tests;

/// <summary>
/// 編集と取り消し。ROADMAP の「無制限の undo（保存後も遡れる）」を固定する。
/// </summary>
public sealed class EditableDocumentTests
{
    [Fact]
    public void StartsCleanAndWithNothingToUndo()
    {
        var document = new EditableDocument(["a", "b"]);

        Assert.False(document.IsModified);
        Assert.False(document.CanUndo);
        Assert.False(document.CanRedo);
    }

    [Fact]
    public void ReplaceChangesLinesAndMarksModified()
    {
        var document = new EditableDocument(["a", "b", "c"]);

        document.Replace(1, 1, ["B1", "B2"]);

        Assert.Equal(["a", "B1", "B2", "c"], document.Lines);
        Assert.True(document.IsModified);
        Assert.True(document.CanUndo);
    }

    [Fact]
    public void UndoAndRedoWalkTheHistory()
    {
        var document = new EditableDocument(["a"]);
        document.Replace(1, 0, ["b"]);
        document.Replace(2, 0, ["c"]);
        Assert.Equal(["a", "b", "c"], document.Lines);

        document.Undo();
        Assert.Equal(["a", "b"], document.Lines);
        document.Undo();
        Assert.Equal(["a"], document.Lines);
        Assert.False(document.CanUndo);

        document.Redo();
        document.Redo();
        Assert.Equal(["a", "b", "c"], document.Lines);
        Assert.False(document.CanRedo);
    }

    /// <summary>取り消して元の状態に戻れば、変更ありの印も消えること。</summary>
    [Fact]
    public void UndoingBackToTheStartClearsTheModifiedMark()
    {
        var document = new EditableDocument(["a"]);
        document.Replace(0, 1, ["z"]);
        Assert.True(document.IsModified);

        document.Undo();

        Assert.False(document.IsModified);
    }

    /// <summary>保存しても履歴は残り、保存前まで遡れること。</summary>
    [Fact]
    public void HistorySurvivesSaving()
    {
        var document = new EditableDocument(["a"]);
        document.Replace(0, 1, ["b"]);
        document.MarkSaved();

        Assert.False(document.IsModified);
        Assert.True(document.CanUndo);

        document.Undo();

        Assert.Equal(["a"], document.Lines);
        // 保存した状態から戻ったので、また「変更あり」になる。
        Assert.True(document.IsModified);
    }

    [Fact]
    public void EditingAfterUndoDropsTheRedoBranch()
    {
        var document = new EditableDocument(["a"]);
        document.Replace(1, 0, ["b"]);
        document.Undo();
        Assert.True(document.CanRedo);

        document.Replace(1, 0, ["c"]);

        Assert.False(document.CanRedo);
        Assert.Equal(["a", "c"], document.Lines);
    }

    /// <summary>
    /// 内容が変わらない置き換えは履歴に積まない。積むと「押しても何も起きない undo」
    /// ができてしまう。
    /// </summary>
    [Fact]
    public void ReplacingWithTheSameContentIsNotRecorded()
    {
        var document = new EditableDocument(["a", "b"]);

        document.Replace(0, 1, ["a"]);

        Assert.False(document.IsModified);
        Assert.False(document.CanUndo);
    }

    [Fact]
    public void ManyEditsCanAllBeUndone()
    {
        var document = new EditableDocument(["0"]);
        for (var i = 1; i <= 500; i++)
        {
            document.Replace(i, 0, [i.ToString()]);
        }
        Assert.Equal(501, document.Lines.Count);

        for (var i = 0; i < 500; i++)
        {
            document.Undo();
        }

        Assert.Equal(["0"], document.Lines);
        Assert.False(document.CanUndo);
    }

    [Fact]
    public void ReplaceRejectsRangesOutsideTheDocument()
    {
        var document = new EditableDocument(["a"]);

        Assert.Throws<ArgumentOutOfRangeException>(() => document.Replace(0, 5, []));
        Assert.Throws<ArgumentOutOfRangeException>(() => document.Replace(-1, 0, []));
    }
}
