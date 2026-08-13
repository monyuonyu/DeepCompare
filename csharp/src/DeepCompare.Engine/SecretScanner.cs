using System.Text.RegularExpressions;

namespace DeepCompare.Engine;

/// <summary>見つけたものの確からしさ。</summary>
public enum SecretConfidence
{
    /// <summary>形が特定の発行元のものと一致する。ほぼ確実。</summary>
    High,

    /// <summary>「秘密らしい名前」に値が付いている。文脈次第。</summary>
    Medium,

    /// <summary>統計的にそれらしいだけ。誤りも多い。</summary>
    Low,
}

/// <summary>見つけた 1 件。</summary>
public sealed record SecretFinding(
    string Kind,
    SecretConfidence Confidence,
    /// <summary>1 始まりの行番号。</summary>
    int Line,
    int Column,
    int Length,
    /// <summary>見つけた部分。**そのままは出さない**ので、伏せた形で持つ。</summary>
    string Masked)
{
    public string Describe() => $"{Line}:{Column} {Kind}（{Label(Confidence)}）{Masked}";

    public static string Label(SecretConfidence confidence) => confidence switch
    {
        SecretConfidence.High => "ほぼ確実",
        SecretConfidence.Medium => "たぶん",
        SecretConfidence.Low => "かもしれない",
        _ => string.Empty,
    };
}

/// <summary>
/// 秘密が混ざっていないか調べる。
///
/// **差分ツールは、変更が外へ出る直前の最後の関所という良い位置にいる。**
/// コミットの前に必ず差分を見るなら、そこで気づけるのが一番早い。
///
/// **見つからないことより、誤って騒ぐことの方が害が大きい。**
/// 毎回でたらめを出す道具は、そのうち誰も読まなくなる。読まれなくなった
/// 警告は無いのと同じなので、確からしさを 3 段階に分けて、弱いものは
/// 弱いと分かる形で出す。
/// </summary>
public static class SecretScanner
{
    /// <summary>
    /// 発行元の形が決まっているもの。**ここに当たれば、ほぼ確実。**
    ///
    /// 形が固定なので誤検出はほとんど無い。逆に、形の決まっていない秘密
    /// （自社の API キーなど）はここでは拾えないので、下の 2 段が要る。
    /// </summary>
    private static readonly (string Kind, Regex Pattern)[] KnownShapes =
    [
        ("AWS のアクセスキー", new Regex(@"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA)[0-9A-Z]{16}\b")),
        ("GitHub のトークン", new Regex(@"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
        ("GitHub のアプリトークン", new Regex(@"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
        ("Slack のトークン", new Regex(@"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
        ("Google の API キー", new Regex(@"\bAIza[0-9A-Za-z_\-]{35}\b")),
        ("Stripe の鍵", new Regex(@"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
        ("OpenAI の鍵", new Regex(@"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
        ("Anthropic の鍵", new Regex(@"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
        ("npm のトークン", new Regex(@"\bnpm_[A-Za-z0-9]{36}\b")),
        ("秘密鍵", new Regex(@"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|PGP)?\s*PRIVATE KEY")),
        ("JSON Web Token", new Regex(@"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
        // **値の形まで縛る。** 緩いままだと `password = os.environ["X"]` のような
        // 参照や、正規表現そのものを書いた行に当たる。実際、この道具を自分の
        // コードに掛けたら**この説明文が「ほぼ確実」で引っかかった。**
        // 接続文字列の値は `;` か行末で終わり、括弧や記号を含まない。
        ("接続文字列のパスワード", new Regex(
            @"(?i)\b(?:password|pwd)\s*=\s*(?<value>[^;\s""'<>\[\](){}$\\|]{6,})\s*(?:;|$)")),
        ("URL に埋めた資格情報", new Regex(@"[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s:@]+@")),
    ];

    /// <summary>
    /// 「秘密らしい名前」に値が付いている形。
    ///
    /// 名前で拾うので、値の形が自由なものも見つかる。ただし
    /// <c>password = os.environ["X"]</c> のような**参照**まで拾ってしまうので、
    /// 値が引用符で囲まれた即値のときだけにする。
    /// </summary>
    private static readonly Regex NamedSecret = new(
        @"(?i)\b(api[_\-]?key|secret|token|passwd|password|credential|private[_\-]?key|auth)\b"
        + @"\s*[:=]\s*(?<q>[""'])(?<value>[^""'\r\n]{6,})\k<q>");

    /// <summary>
    /// 明らかに本物ではないもの。**ここを外すと、例のコードで毎回騒ぐ。**
    ///
    /// 説明書やテストには必ず「それらしい値」が書いてあるので、これが無いと
    /// 警告だらけになって読まれなくなる。
    /// </summary>
    private static readonly Regex Placeholder = new(
        @"(?i)^(?:x{3,}|\*{3,}|\.{3,}|<.*>|\{\{.*\}\}|\$\{.*\}|%.*%|"
        + @"(?:your|my|the)[_\-]?\w*|example\w*|sample\w*|dummy\w*|test\w*|fake\w*|"
        + @"changeme|placeholder|todo|none|null|undefined|redacted|secret|password|"
        + @"abc123|123456\d*|xxxx+)$");

    /// <summary>
    /// 統計的にそれらしい文字列。
    ///
    /// **これだけでは判断しない。** 乱数のような文字列はハッシュ値、識別子、
    /// 符号化した画像など、秘密でないものにも山ほどある。名前や形の手がかりが
    /// 無い場合の最後の網として、弱い印で出すだけにする。
    /// </summary>
    private const double EntropyThreshold = 4.2;
    private const int MinimumEntropyLength = 24;

    /// <summary>
    /// その行を調べない印。
    ///
    /// **意図して書いた値を黙らせる手段が要る。** 試験や説明書には本物と
    /// 同じ形の値を書くことがあり、そこで毎回騒がれると、道具ごと切られる。
    ///
    /// 他の道具と同じ綴りも受ける（gitleaks / nosec）。**印の付け方を
    /// 覚え直させない**方が、実際に使ってもらえる。
    /// </summary>
    private static readonly Regex AllowMark = new(
        @"(?i)(?:deepcompare|gitleaks|secret)[:\-]?allow|\bnosec\b|\bnosecret\b");

    /// <summary>
    /// そのファイル全体を調べないか。**先頭の数行に印があれば飛ばす。**
    ///
    /// 試験や説明書のように、本物と同じ形の値が何十個も並ぶファイルがある。
    /// 1 行ずつ印を付けるのは現実的でないし、付け忘れた行だけ騒がれる。
    /// ファイルの頭で「ここは意図的」と言えるようにする。
    /// </summary>
    private static bool FileIsAllowed(IReadOnlyList<string> lines)
    {
        // 先頭の数行だけ見る。**全体を見ると、途中の 1 行の印でファイルごと
        // 黙ることになり、意図しない見逃しが起きる。**
        var head = Math.Min(8, lines.Count);
        for (var i = 0; i < head; i++)
        {
            if (lines[i].Contains("deepcompare:allow-file", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    public static IReadOnlyList<SecretFinding> Scan(IReadOnlyList<string> lines)
    {
        var findings = new List<SecretFinding>();
        if (FileIsAllowed(lines))
        {
            return findings;
        }

        for (var i = 0; i < lines.Count; i++)
        {
            var line = lines[i];
            if (line.Length == 0 || AllowMark.IsMatch(line))
            {
                continue;
            }

            var claimed = new List<(int Start, int End)>();

            foreach (var (kind, pattern) in KnownShapes)
            {
                foreach (Match match in pattern.Matches(line))
                {
                    // 値を取れる形なら、それが置き換え用でないかも見る。
                    // 形が決まっているものでも、説明のために書かれることはある。
                    var captured = match.Groups["value"];
                    if (captured.Success && Placeholder.IsMatch(captured.Value.Trim()))
                    {
                        continue;
                    }

                    findings.Add(new SecretFinding(
                        kind, SecretConfidence.High, i + 1, match.Index + 1,
                        match.Length, Mask(match.Value)));
                    claimed.Add((match.Index, match.Index + match.Length));
                }
            }

            foreach (Match match in NamedSecret.Matches(line))
            {
                var value = match.Groups["value"].Value;
                if (Placeholder.IsMatch(value.Trim()) || Overlaps(claimed, match.Index, match.Length))
                {
                    continue;
                }
                findings.Add(new SecretFinding(
                    "秘密らしい名前に値が付いている", SecretConfidence.Medium,
                    i + 1, match.Index + 1, match.Length, Mask(value)));
                claimed.Add((match.Index, match.Index + match.Length));
            }

            foreach (var (start, length) in HighEntropyRuns(line))
            {
                if (Overlaps(claimed, start, length))
                {
                    continue;
                }
                var value = line.Substring(start, length);
                if (Placeholder.IsMatch(value))
                {
                    continue;
                }
                findings.Add(new SecretFinding(
                    "乱数のような文字列", SecretConfidence.Low,
                    i + 1, start + 1, length, Mask(value)));
            }
        }

        return findings;
    }

    /// <summary>
    /// 差分のうち**増えた側だけ**を調べる。
    ///
    /// 既にあるものを毎回言われても直しようがない（消すなら履歴からも消す
    /// 必要があり、それはこの道具の仕事ではない）。**これから外へ出る分**に
    /// 絞る方が、警告の数が減って読んでもらえる。
    /// </summary>
    public static IReadOnlyList<SecretFinding> ScanAdded(Comparison comparison, DecodedText right)
    {
        var added = new List<string>();
        var numbers = new List<int>();

        foreach (var row in comparison.Rows)
        {
            if (row.Right is not { } index)
            {
                continue;
            }
            if (row.Left is null || !row.IsUnchanged)
            {
                added.Add(right.Lines[index]);
                numbers.Add(index + 1);
            }
        }

        // 行番号を元のものへ戻す。詰めた並びのままだと場所を示せない。
        return [.. Scan(added).Select(f => f with { Line = numbers[f.Line - 1] })];
    }

    private static bool Overlaps(List<(int Start, int End)> claimed, int start, int length)
        => claimed.Any(c => start < c.End && start + length > c.Start);

    /// <summary>
    /// 値を伏せる。**そのままは出さない。**
    ///
    /// 警告を画面や報告に出す道具なので、そこに秘密を書くと、秘密が
    /// 別の場所へ増えるだけになる。長さと前後だけ分かれば場所は特定できる。
    /// </summary>
    private static string Mask(string value)
    {
        if (value.Length <= 8)
        {
            return new string('*', value.Length);
        }
        return $"{value[..3]}…{value[^2..]}（{value.Length} 文字）";
    }

    /// <summary>乱雑さ（シャノンのエントロピー）。1 文字あたりの情報量。</summary>
    internal static double Entropy(string text)
    {
        if (text.Length == 0)
        {
            return 0;
        }
        var counts = new Dictionary<char, int>();
        foreach (var c in text)
        {
            counts[c] = counts.GetValueOrDefault(c) + 1;
        }

        var entropy = 0.0;
        foreach (var count in counts.Values)
        {
            var p = (double)count / text.Length;
            entropy -= p * Math.Log2(p);
        }
        return entropy;
    }

    /// <summary>乱雑さの高い連なりを探す。</summary>
    private static IEnumerable<(int Start, int Length)> HighEntropyRuns(string line)
    {
        // 秘密は「区切り記号を含まない、そこそこ長い塊」として現れる。
        // 単語の切れ目で分け、塊ごとに見る。
        var start = -1;
        for (var i = 0; i <= line.Length; i++)
        {
            var part = i < line.Length && (char.IsLetterOrDigit(line[i])
                || line[i] is '+' or '/' or '=' or '_' or '-');

            if (part)
            {
                if (start < 0)
                {
                    start = i;
                }
                continue;
            }

            if (start >= 0)
            {
                var length = i - start;
                if (length >= MinimumEntropyLength)
                {
                    var text = line.Substring(start, length);
                    // **数字だけ・英字だけは除く。** 電話番号や長い識別子、
                    // 英単語の連なりを毎回拾うと使いものにならない。
                    var hasDigit = text.Any(char.IsDigit);
                    var hasLetter = text.Any(char.IsLetter);
                    if (hasDigit && hasLetter && Entropy(text) >= EntropyThreshold)
                    {
                        yield return (start, length);
                    }
                }
                start = -1;
            }
        }
    }

    /// <summary>人が読む形に整える。</summary>
    public static string Format(IReadOnlyList<SecretFinding> findings)
    {
        if (findings.Count == 0)
        {
            return "秘密らしいものは見つかりませんでした。" + Environment.NewLine;
        }

        var text = new System.Text.StringBuilder();
        foreach (var finding in findings.OrderBy(f => f.Line).ThenBy(f => f.Column))
        {
            text.AppendLine(finding.Describe());
        }
        text.AppendLine();
        foreach (var group in findings.GroupBy(f => f.Confidence).OrderBy(g => g.Key))
        {
            text.AppendLine($"{SecretFinding.Label(group.Key)}: {group.Count()} 件");
        }
        return text.ToString();
    }
}
