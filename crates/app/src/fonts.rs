//! 日本語を表示できるようにする。
//!
//! egui の既定フォントに CJK は入っていないので、何もしないと日本語がすべて豆腐に
//! なる。UI のラベルだけの話ではなく、Shift_JIS のファイルを読めるようにした以上、
//! 比較対象そのものに日本語が含まれる。
//!
//! フォントを同梱すれば確実だが、日本語の書体は最小の部分集合でも数 MiB あり、
//! 「軽量な単一 exe」という前提を削ってしまう。ここでは OS が持っている書体を探す。
//! 見つからない場合でも起動はし、ASCII の比較は問題なくできる。

/// 探す順。等幅を優先する（コードの桁が揃う）が、無ければ本文用でも表示はできる。
#[cfg(target_os = "windows")]
const CANDIDATES: &[&str] = &[
    r"C:\Windows\Fonts\msgothic.ttc", // MS ゴシック（等幅）
    r"C:\Windows\Fonts\YuGothM.ttc",  // 游ゴシック Medium
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\msmincho.ttc",
];

#[cfg(not(target_os = "windows"))]
const CANDIDATES: &[&str] = &[
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
    "/usr/share/fonts/OTF/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
];

/// `.ttc`（フォントコレクション）の何番目を使うか。日本語の書体は複数の
/// ウェイトを 1 ファイルに束ねていることが多く、先頭が通常の字面になっている。
pub const FONT_INDEX: u32 = 0;

/// 探索する書体の一覧。診断（`--font-check`）と実際の読み込みで同じ列を使う。
pub fn candidates() -> &'static [&'static str] {
    CANDIDATES
}

/// 見つかった書体の名前。見つからなければ `None`。
pub fn install_cjk_font(ctx: &egui::Context) -> Option<String> {
    let (path, data) = CANDIDATES
        .iter()
        .find_map(|path| std::fs::read(path).ok().map(|data| (*path, data)))?;

    let mut fonts = egui::FontDefinitions::default();
    // `.ttc` は複数の書体を束ねたファイルなので、どれを使うかを明示する。
    // 診断（`--font-check`）が調べるのと同じ index でなければ意味がない。
    let font_data = egui::FontData {
        index: FONT_INDEX,
        ..egui::FontData::from_owned(data)
    };
    fonts
        .font_data
        .insert("system-cjk".to_owned(), std::sync::Arc::new(font_data));

    // 末尾に足す。ASCII は既定の書体のままにして、既定に無い字だけを引き受けさせる。
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("system-cjk".to_owned());
    }
    ctx.set_fonts(fonts);
    Some(path.to_owned())
}
