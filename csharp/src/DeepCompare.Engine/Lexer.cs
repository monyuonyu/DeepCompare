namespace DeepCompare.Engine;

public enum TokenKind
{
    Text,
    Keyword,
    Identifier,
    Number,
    String,
    Comment,
    Punctuation,
    Whitespace,
}

/// <summary>行の一部を指すトークン。Start/Length は元の行への文字単位の範囲。</summary>
public readonly record struct Token(TokenKind Kind, int Start, int Length);

/// <summary>
/// 言語の決まり。1 つの字句解析器を設定で使い回す。
///
/// 言語ごとに解析器を書くと、対応言語を増やすたびに同じ間違いを繰り返す。ここで
/// 扱うのは色分けと行内差分の粒度だけなので、構文木までは要らない。
/// </summary>
public sealed record Language(
    string Name,
    IReadOnlySet<string> Keywords,
    IReadOnlyList<string> LineComments,
    string? BlockCommentStart = null,
    string? BlockCommentEnd = null,
    IReadOnlyList<char>? StringDelimiters = null)
{
    public IReadOnlyList<char> Strings => StringDelimiters ?? ['"', '\''];
}

/// <summary>行をまたぐ状態。ブロックコメントの途中かどうか。</summary>
public readonly record struct LexState(bool InBlockComment)
{
    public static readonly LexState Start = new(false);
}

/// <summary>
/// 字句解析。色分けと、行内差分をトークン単位にするために使う。
///
/// 文字単位の行内差分は `Update` と `Refresh` の共通部分を拾って断片的に光らせるが、
/// トークン単位なら語ごとに光る。読みやすさが目に見えて変わる。
/// </summary>
public static class Lexer
{
    private static readonly HashSet<string> CFamily = new(StringComparer.Ordinal)
    {
        "if", "else", "for", "while", "do", "switch", "case", "default", "break", "continue",
        "return", "goto", "class", "struct", "enum", "interface", "namespace", "using", "import",
        "package", "public", "private", "protected", "internal", "static", "const", "readonly",
        "final", "abstract", "virtual", "override", "sealed", "new", "delete", "this", "base",
        "super", "null", "nullptr", "true", "false", "void", "int", "long", "short", "float",
        "double", "bool", "boolean", "char", "string", "var", "let", "fn", "func", "def", "type",
        "try", "catch", "finally", "throw", "throws", "async", "await", "yield", "in", "is", "as",
        "impl", "trait", "mut", "pub", "use", "mod", "match", "where", "extends", "implements",
        "function", "export", "extern", "unsafe", "operator", "template", "typename", "auto",
    };

    private static readonly HashSet<string> Python = new(StringComparer.Ordinal)
    {
        "def", "class", "return", "if", "elif", "else", "for", "while", "break", "continue",
        "pass", "import", "from", "as", "try", "except", "finally", "raise", "with", "lambda",
        "global", "nonlocal", "assert", "del", "yield", "async", "await", "None", "True", "False",
        "and", "or", "not", "in", "is", "self",
    };

    private static readonly HashSet<string> Shell = new(StringComparer.Ordinal)
    {
        "if", "then", "else", "elif", "fi", "for", "while", "do", "done", "case", "esac",
        "function", "return", "export", "local", "readonly", "shift", "exit", "echo", "cd",
    };

    private static readonly HashSet<string> Lua = new(StringComparer.Ordinal)
    {
        "and", "break", "do", "else", "elseif", "end", "false", "for", "function", "goto",
        "if", "in", "local", "nil", "not", "or", "repeat", "return", "then", "true", "until",
        "while", "self",
    };

    private static readonly HashSet<string> PowerShell = new(StringComparer.OrdinalIgnoreCase)
    {
        "if", "else", "elseif", "switch", "foreach", "for", "while", "do", "until", "break",
        "continue", "return", "function", "param", "begin", "process", "end", "try", "catch",
        "finally", "throw", "class", "enum", "filter", "in", "true", "false", "null",
    };

    private static readonly HashSet<string> Docker = new(StringComparer.OrdinalIgnoreCase)
    {
        "from", "run", "cmd", "label", "maintainer", "expose", "env", "add", "copy",
        "entrypoint", "volume", "user", "workdir", "arg", "onbuild", "stopsignal",
        "healthcheck", "shell", "as",
    };

    private static readonly HashSet<string> Make = new(StringComparer.Ordinal)
    {
        "ifeq", "ifneq", "ifdef", "ifndef", "else", "endif", "include", "define", "endef",
        "export", "unexport", "override", "vpath", ".PHONY", ".DEFAULT", ".SUFFIXES",
    };

    private static readonly HashSet<string> Haskell = new(StringComparer.Ordinal)
    {
        "module", "where", "import", "data", "type", "newtype", "class", "instance", "deriving",
        "do", "case", "of", "let", "in", "if", "then", "else", "infix", "infixl", "infixr",
    };

    private static readonly HashSet<string> Elixir = new(StringComparer.Ordinal)
    {
        "def", "defp", "defmodule", "defstruct", "defprotocol", "defimpl", "do", "end", "fn",
        "if", "unless", "else", "case", "cond", "with", "for", "when", "and", "or", "not",
        "in", "alias", "import", "require", "use", "true", "false", "nil", "receive", "after",
    };

    private static readonly HashSet<string> None = new(StringComparer.Ordinal);

    /// <summary>
    /// 道具の設定ファイルなど、**拡張子を持たないもの**。
    ///
    /// Makefile や Dockerfile は拡張子で判定できない。実際に比べる機会が多い
    /// ファイルなので、名前で拾う。
    /// </summary>
    private static Language? ForFileName(string name) => name.ToLowerInvariant() switch
    {
        "makefile" or "gnumakefile" or "makefile.am" or "makefile.in"
            => new Language("Makefile", Make, ["#"]),
        "dockerfile" or "containerfile" => new Language("Dockerfile", Docker, ["#"]),
        ".gitignore" or ".dockerignore" or ".npmignore" or ".gitattributes"
            => new Language("無視の指定", None, ["#"]),
        ".env" or ".envrc" => new Language("環境変数", None, ["#"]),
        "cmakelists.txt" => new Language("CMake", CFamily, ["#"]),
        _ => null,
    };

    /// <summary>拡張子から言語を決める。分からなければ null（色分けなし）。</summary>
    public static Language? ForPath(string path)
        => ForFileName(Path.GetFileName(path)) ?? Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".c" or ".h" or ".cpp" or ".hpp" or ".cc" or ".cs" or ".java" or ".js" or ".jsx"
                or ".ts" or ".tsx" or ".go" or ".rs" or ".swift" or ".kt" or ".scala" or ".php"
                => new Language("C 系", CFamily, ["//"], "/*", "*/"),
            ".py" or ".pyi" => new Language("Python", Python, ["#"], "\"\"\"", "\"\"\""),
            ".sh" or ".bash" or ".zsh" => new Language("シェル", Shell, ["#"]),
            ".rb" => new Language("Ruby", Python, ["#"]),
            ".sql" => new Language("SQL", CFamily, ["--"], "/*", "*/"),
            ".json" => new Language("JSON", None, [], StringDelimiters: ['"']),
            ".yaml" or ".yml" => new Language("YAML", None, ["#"], StringDelimiters: ['"', '\'']),
            ".toml" or ".ini" or ".cfg" or ".conf" => new Language("設定", None, ["#", ";"]),
            ".xml" or ".html" or ".htm" or ".xaml" or ".axaml" or ".svg"
                => new Language("XML", None, [], "<!--", "-->", ['"', '\'']),
            ".css" or ".scss" or ".less" => new Language("CSS", None, ["//"], "/*", "*/"),
            ".lua" => new Language("Lua", Lua, ["--"], "--[[", "]]"),
            ".ps1" or ".psm1" or ".psd1" => new Language("PowerShell", PowerShell, ["#"], "<#", "#>"),
            ".pl" or ".pm" => new Language("Perl", Python, ["#"]),
            ".r" => new Language("R", Python, ["#"]),
            ".dart" => new Language("Dart", CFamily, ["//"], "/*", "*/"),
            ".zig" or ".nim" or ".v" => new Language("C 系", CFamily, ["//"], "/*", "*/"),
            ".hs" => new Language("Haskell", Haskell, ["--"], "{-", "-}"),
            ".ex" or ".exs" => new Language("Elixir", Elixir, ["#"]),
            ".vim" => new Language("Vim script", None, ["\""]),
            ".tf" or ".hcl" => new Language("HCL", None, ["#", "//"], "/*", "*/", ['"']),
            ".proto" => new Language("Protocol Buffers", CFamily, ["//"], "/*", "*/"),
            ".gradle" or ".groovy" => new Language("Groovy", CFamily, ["//"], "/*", "*/"),
            ".md" or ".markdown" => new Language("Markdown", None, [], StringDelimiters: ['`']),
            ".vue" or ".svelte" => new Language("XML", None, [], "<!--", "-->", ['"', '\'']),
            _ => null,
        };

    /// <summary>
    /// 1 行を分解する。<paramref name="state"/> は前の行から引き継ぐ状態で、
    /// 戻り値の状態を次の行へ渡す。
    /// </summary>
    public static List<Token> Tokenize(string line, Language? language, ref LexState state)
    {
        var tokens = new List<Token>();
        if (line.Length == 0)
        {
            return tokens;
        }
        if (language is null)
        {
            tokens.Add(new Token(TokenKind.Text, 0, line.Length));
            return tokens;
        }

        var at = 0;
        while (at < line.Length)
        {
            // ブロックコメントの途中なら、終わりを探すまで全部コメント。
            if (state.InBlockComment)
            {
                var close = language.BlockCommentEnd is { } end
                    ? line.IndexOf(end, at, StringComparison.Ordinal)
                    : -1;
                if (close < 0)
                {
                    tokens.Add(new Token(TokenKind.Comment, at, line.Length - at));
                    return tokens;
                }
                var length = close + language.BlockCommentEnd!.Length - at;
                tokens.Add(new Token(TokenKind.Comment, at, length));
                at += length;
                state = new LexState(false);
                continue;
            }

            var c = line[at];

            if (char.IsWhiteSpace(c))
            {
                var start = at;
                while (at < line.Length && char.IsWhiteSpace(line[at]))
                {
                    at++;
                }
                tokens.Add(new Token(TokenKind.Whitespace, start, at - start));
                continue;
            }

            // 行コメント。以降は行末まで全部。
            var lineComment = language.LineComments
                .FirstOrDefault(marker => Matches(line, at, marker));
            if (lineComment is not null)
            {
                tokens.Add(new Token(TokenKind.Comment, at, line.Length - at));
                return tokens;
            }

            if (language.BlockCommentStart is { } open && Matches(line, at, open)
                && language.BlockCommentEnd is { } blockEnd)
            {
                var start = at;
                // 開始記号の分を飛ばしてから終わりを探す。飛ばさないと、開始と終了が
                // 同じ綴り（Python の """）のときに自分自身で閉じたことになる。
                var close = line.IndexOf(blockEnd, at + open.Length, StringComparison.Ordinal);
                if (close < 0)
                {
                    tokens.Add(new Token(TokenKind.Comment, start, line.Length - start));
                    state = new LexState(true);
                    return tokens;
                }
                at = close + blockEnd.Length;
                tokens.Add(new Token(TokenKind.Comment, start, at - start));
                continue;
            }

            if (language.Strings.Contains(c))
            {
                var start = at;
                at++;
                while (at < line.Length)
                {
                    // 逃がし文字の次は中身として飛ばす。これを見ないと "\"" で閉じたと誤る。
                    if (line[at] == '\\' && at + 1 < line.Length)
                    {
                        at += 2;
                        continue;
                    }
                    if (line[at] == c)
                    {
                        at++;
                        break;
                    }
                    at++;
                }
                tokens.Add(new Token(TokenKind.String, start, at - start));
                continue;
            }

            if (char.IsDigit(c))
            {
                var start = at;
                while (at < line.Length
                       && (char.IsLetterOrDigit(line[at]) || line[at] == '.' || line[at] == '_'))
                {
                    at++;
                }
                tokens.Add(new Token(TokenKind.Number, start, at - start));
                continue;
            }

            if (char.IsLetter(c) || c == '_' || c == '$')
            {
                var start = at;
                while (at < line.Length
                       && (char.IsLetterOrDigit(line[at]) || line[at] == '_' || line[at] == '$'))
                {
                    at++;
                }
                var word = line[start..at];
                tokens.Add(new Token(
                    language.Keywords.Contains(word) ? TokenKind.Keyword : TokenKind.Identifier,
                    start, at - start));
                continue;
            }

            // 記号は 1 文字ずつ。まとめると `!=` と `!` の区別のために表が要る。
            tokens.Add(new Token(TokenKind.Punctuation, at, 1));
            at++;
        }

        return tokens;
    }

    /// <summary>状態を引き継がない単発の呼び出し。</summary>
    public static List<Token> Tokenize(string line, Language? language)
    {
        var state = LexState.Start;
        return Tokenize(line, language, ref state);
    }

    private static bool Matches(string line, int at, string marker)
        => marker.Length > 0
           && at + marker.Length <= line.Length
           && line.AsSpan(at, marker.Length).SequenceEqual(marker);
}
