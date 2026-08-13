using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;

namespace DeepCompare.App;

public abstract class ViewModelBase : INotifyPropertyChanged
{
    public event PropertyChangedEventHandler? PropertyChanged;

    protected bool Set<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }
        field = value;
        OnPropertyChanged(name);
        return true;
    }

    protected void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}

/// <summary>非同期処理を起動できる最小限のコマンド。</summary>
public sealed class RelayCommand(Func<Task> execute, Func<bool>? canExecute = null) : ICommand
{
    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter) => canExecute?.Invoke() ?? true;

    public async void Execute(object? parameter)
    {
        try
        {
            await execute();
        }
        catch (Exception error)
        {
            // ここで漏らすとプロセスごと落ちる。
            Console.Error.WriteLine($"コマンドが失敗: {error}");
        }
    }

    public void Raise() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}

/// <summary>
/// 対象を伴うコマンド。差分の塊ごとに置くコピーボタンのように、
/// 「どれに対して」が要る操作に使う。
/// </summary>
public sealed class RelayCommand<T>(Func<T, Task> execute, Func<T, bool>? canExecute = null) : ICommand
{
    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter)
        => parameter is T value && (canExecute?.Invoke(value) ?? true);

    public async void Execute(object? parameter)
    {
        if (parameter is not T value)
        {
            return;
        }
        try
        {
            await execute(value);
        }
        catch (Exception error)
        {
            // ここで漏らすとプロセスごと落ちる。
            Console.Error.WriteLine($"コマンドが失敗: {error}");
        }
    }

    public void Raise() => CanExecuteChanged?.Invoke(this, EventArgs.Empty);
}
