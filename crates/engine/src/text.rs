//! ファイルのバイト列を行の並びへ落とすところ。
//!
//! 旧 Python 実装は `open(path, encoding="utf-8")` 固定だったので、Windows で保存した
//! Shift_JIS のソースを渡すと比較開始と同時に例外ダイアログが出て終わっていた。
//! ここで符号化を推定し、BOM と改行コードは復元できる形で剥がしておく。

use std::fmt;

/// 推定された文字符号化。表示してユーザーが誤りに気付けるよう、名前を持たせている。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Utf8,
    Utf8Bom,
    Utf16Le,
    Utf16Be,
    ShiftJis,
    EucJp,
    /// どれとしても妥当に解釈できず、UTF-8 として不正な箇所を潰して読んだ場合。
    Utf8Lossy,
}

impl Encoding {
    pub fn label(self) -> &'static str {
        match self {
            Encoding::Utf8 => "UTF-8",
            Encoding::Utf8Bom => "UTF-8 (BOM)",
            Encoding::Utf16Le => "UTF-16LE",
            Encoding::Utf16Be => "UTF-16BE",
            Encoding::ShiftJis => "Shift_JIS",
            Encoding::EucJp => "EUC-JP",
            Encoding::Utf8Lossy => "UTF-8 (不正バイトを置換)",
        }
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// 支配的な改行コード。書き戻しは今のところしないが、比較結果の見出しに出すと
/// 「中身は同じなのに全行差分になる」場合の原因がすぐ分かる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEnding {
    Lf,
    CrLf,
    Cr,
    /// 混在。全行差分の典型的な原因なので独立した値にしてある。
    Mixed,
    /// 改行が一つも無い（空ファイル、または1行のみ）。
    None,
}

impl LineEnding {
    pub fn label(self) -> &'static str {
        match self {
            LineEnding::Lf => "LF",
            LineEnding::CrLf => "CRLF",
            LineEnding::Cr => "CR",
            LineEnding::Mixed => "混在",
            LineEnding::None => "-",
        }
    }
}

impl fmt::Display for LineEnding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone)]
pub struct DecodedText {
    /// 改行で分割済みの行。終端の改行文字は含まない。
    pub lines: Vec<String>,
    pub encoding: Encoding,
    pub line_ending: LineEnding,
}

/// バイト列を復号し、行へ分割する。
pub fn decode_file(bytes: &[u8]) -> DecodedText {
    let (text, encoding) = decode(bytes);
    let line_ending = detect_line_ending(&text);
    DecodedText {
        lines: split_lines(&text),
        encoding,
        line_ending,
    }
}

fn decode(bytes: &[u8]) -> (String, Encoding) {
    // 1. BOM があればそれが最も確かな根拠なので優先する。
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        return (
            String::from_utf8_lossy(rest).into_owned(),
            Encoding::Utf8Bom,
        );
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        return (decode_utf16(rest, true), Encoding::Utf16Le);
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        return (decode_utf16(rest, false), Encoding::Utf16Be);
    }

    // 2. UTF-8 として厳密に妥当なら UTF-8。多バイト列の妥当性は偶然には成立しにくく、
    //    Shift_JIS のテキストが UTF-8 として通ることはまず無いので、この順序で良い。
    if let Ok(s) = std::str::from_utf8(bytes) {
        return (s.to_owned(), Encoding::Utf8);
    }

    // 3. 日本語のレガシー符号化を、誤り無く復号できるものだけ順に試す。
    //    Windows で書かれたソースを想定して Shift_JIS を先に見る。
    for (enc, tag) in [
        (encoding_rs::SHIFT_JIS, Encoding::ShiftJis),
        (encoding_rs::EUC_JP, Encoding::EucJp),
    ] {
        let (text, _, had_errors) = enc.decode(bytes);
        // 「復号でエラーが出なかった」だけでは根拠として弱い。Shift_JIS は単バイトで
        // 受け付ける範囲が広く、latin-1 のテキストやバイナリでもほぼエラー無しに
        // 通ってしまうため、結果が日本語文として妥当かどうかまで見る。
        if !had_errors && looks_like_japanese_text(&text) {
            return (text.into_owned(), tag);
        }
    }

    // 4. どれでもない。読めるところまで読む方が、開けないより有用。
    (
        String::from_utf8_lossy(bytes).into_owned(),
        Encoding::Utf8Lossy,
    )
}

/// レガシー符号化として復号した結果が、本当にその符号化の文章だったかを判定する。
///
/// 判断材料は二つ。
///
/// - 制御文字が混ざっていれば、テキストではなくバイナリを読んでいる。
/// - 非 ASCII の大半が半角カタカナなら、それは日本語ではなく誤復号の兆候。
///   latin-1 の上位バイトや任意のバイナリを Shift_JIS として読むと、0xA1..0xDF が
///   すべて半角カタカナに化けるため、この形になる。
///
/// 代償として、本当に半角カタカナばかりの Shift_JIS ファイルは取りこぼす。
/// ソースコードの比較という用途ではまず出てこない形なので、誤検出を減らす側を採る。
fn looks_like_japanese_text(text: &str) -> bool {
    let mut non_ascii = 0usize;
    let mut halfwidth_katakana = 0usize;
    for c in text.chars() {
        if c.is_control() && !matches!(c, '\t' | '\n' | '\r') {
            return false;
        }
        if !c.is_ascii() {
            non_ascii += 1;
            if ('\u{FF61}'..='\u{FF9F}').contains(&c) {
                halfwidth_katakana += 1;
            }
        }
    }
    if non_ascii == 0 {
        // 非 ASCII が無いなら、そもそも UTF-8 として通っていたはず。ここへ来る時点で
        // 何か食い違っているので採用しない。
        return false;
    }
    halfwidth_katakana * 2 <= non_ascii
}

fn decode_utf16(bytes: &[u8], little_endian: bool) -> String {
    let units: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|p| {
            if little_endian {
                u16::from_le_bytes([p[0], p[1]])
            } else {
                u16::from_be_bytes([p[0], p[1]])
            }
        })
        .collect();
    String::from_utf16_lossy(&units)
}

fn detect_line_ending(text: &str) -> LineEnding {
    let bytes = text.as_bytes();
    let (mut crlf, mut lf, mut cr) = (0usize, 0usize, 0usize);
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\r' => {
                if bytes.get(i + 1) == Some(&b'\n') {
                    crlf += 1;
                    i += 2;
                    continue;
                }
                cr += 1;
            }
            b'\n' => lf += 1,
            _ => {}
        }
        i += 1;
    }
    match (crlf > 0, lf > 0, cr > 0) {
        (false, false, false) => LineEnding::None,
        (true, false, false) => LineEnding::CrLf,
        (false, true, false) => LineEnding::Lf,
        (false, false, true) => LineEnding::Cr,
        _ => LineEnding::Mixed,
    }
}

/// CRLF / LF / CR のいずれでも分割する。末尾の改行は空行を生まない
/// （Python の `str.splitlines()` と同じ扱い）。
fn split_lines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                lines.push(std::mem::take(&mut current));
            }
            '\n' => lines.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_utf8_is_detected() {
        let d = decode_file("あいう\nかきく\n".as_bytes());
        assert_eq!(d.encoding, Encoding::Utf8);
        assert_eq!(d.lines, vec!["あいう", "かきく"]);
        assert_eq!(d.line_ending, LineEnding::Lf);
    }

    #[test]
    fn utf8_bom_is_stripped_not_left_on_the_first_line() {
        // BOM が本文に残ると 1 行目だけが必ず差分として出てしまう。
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"fn main() {}\n");
        let d = decode_file(&bytes);
        assert_eq!(d.encoding, Encoding::Utf8Bom);
        assert_eq!(d.lines, vec!["fn main() {}"]);
    }

    #[test]
    fn shift_jis_is_read_instead_of_failing() {
        // 旧実装がここで例外を投げて比較そのものを諦めていた入力。
        let (bytes, _, _) = encoding_rs::SHIFT_JIS.encode("日本語のコメント\nprint(1)\n");
        let d = decode_file(&bytes);
        assert_eq!(d.encoding, Encoding::ShiftJis);
        assert_eq!(d.lines, vec!["日本語のコメント", "print(1)"]);
    }

    /// encoding_rs の UTF-16 は復号専用（`encode()` は UTF-8 を返す）ので、
    /// 検査用のバイト列は自前で組み立てる。
    fn utf16_bytes(s: &str, little_endian: bool) -> Vec<u8> {
        s.encode_utf16()
            .flat_map(|u| {
                if little_endian {
                    u.to_le_bytes()
                } else {
                    u.to_be_bytes()
                }
            })
            .collect()
    }

    #[test]
    fn utf16le_with_bom_is_read() {
        let mut bytes = vec![0xFF, 0xFE];
        bytes.extend_from_slice(&utf16_bytes("abc\ndef\n", true));
        let d = decode_file(&bytes);
        assert_eq!(d.encoding, Encoding::Utf16Le);
        assert_eq!(d.lines, vec!["abc", "def"]);
    }

    #[test]
    fn utf16be_with_bom_is_read() {
        let mut bytes = vec![0xFE, 0xFF];
        bytes.extend_from_slice(&utf16_bytes("abc\ndef\n", false));
        let d = decode_file(&bytes);
        assert_eq!(d.encoding, Encoding::Utf16Be);
        assert_eq!(d.lines, vec!["abc", "def"]);
    }

    #[test]
    fn crlf_and_lf_produce_identical_lines() {
        // 改行コードだけが違うファイルが全行差分にならないことの担保。
        let lf = decode_file(b"a\nb\nc\n");
        let crlf = decode_file(b"a\r\nb\r\nc\r\n");
        assert_eq!(lf.lines, crlf.lines);
        assert_eq!(lf.line_ending, LineEnding::Lf);
        assert_eq!(crlf.line_ending, LineEnding::CrLf);
    }

    #[test]
    fn mixed_line_endings_are_reported() {
        assert_eq!(decode_file(b"a\r\nb\nc").line_ending, LineEnding::Mixed);
    }

    #[test]
    fn trailing_newline_does_not_create_an_empty_line() {
        assert_eq!(decode_file(b"a\nb\n").lines, vec!["a", "b"]);
        assert_eq!(decode_file(b"a\nb").lines, vec!["a", "b"]);
    }

    #[test]
    fn blank_lines_in_the_middle_are_kept() {
        assert_eq!(decode_file(b"a\n\nb\n").lines, vec!["a", "", "b"]);
    }

    #[test]
    fn empty_input_yields_no_lines() {
        let d = decode_file(b"");
        assert!(d.lines.is_empty());
        assert_eq!(d.line_ending, LineEnding::None);
    }

    #[test]
    fn undecodable_bytes_still_yield_text() {
        // 開けずに終わるより、読めるところまで見せる方が使える。
        let d = decode_file(&[b'a', 0xC3, 0x28, b'b']);
        assert_eq!(d.encoding, Encoding::Utf8Lossy);
        assert_eq!(d.lines.len(), 1);
    }

    #[test]
    fn latin1_is_not_mistaken_for_shift_jis() {
        // Shift_JIS は 0xA1..0xDF を単バイトの半角カタカナとして受け入れるので、
        // 「復号エラーが出ない」だけを根拠にすると latin-1 が丸ごと化けて通る。
        let latin1: Vec<u8> = b"caf\xE9 na\xEFve r\xE9sum\xE9".to_vec();
        assert_ne!(decode_file(&latin1).encoding, Encoding::ShiftJis);
    }

    #[test]
    fn binary_input_is_not_mistaken_for_text_encoding() {
        let binary: Vec<u8> = vec![0x00, 0x01, 0xB1, 0xB2, 0x02, 0xC0];
        assert_ne!(decode_file(&binary).encoding, Encoding::ShiftJis);
        assert_ne!(decode_file(&binary).encoding, Encoding::EucJp);
    }

    #[test]
    fn japanese_shift_jis_survives_the_stricter_check() {
        // 誤検出を潰す過程で、本来通すべき日本語まで弾いていないことの確認。
        let (bytes, _, _) =
            encoding_rs::SHIFT_JIS.encode("// 設定を読み込む\nlet 名前 = \"太郎\";\n");
        assert_eq!(decode_file(&bytes).encoding, Encoding::ShiftJis);
    }

    #[test]
    fn euc_jp_is_detected() {
        let (bytes, _, _) = encoding_rs::EUC_JP.encode("日本語の文章です\n二行目\n");
        let d = decode_file(&bytes);
        assert_eq!(d.encoding, Encoding::EucJp);
        assert_eq!(d.lines, vec!["日本語の文章です", "二行目"]);
    }
}
