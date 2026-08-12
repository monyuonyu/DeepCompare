//! 描画。
//!
//! 旧 Python 実装は差分行ごとに `QLabel` を生成してテーブルへ差し込んでいた。行数分の
//! ウィジェットが実体として並ぶため、数千行のファイルで操作できなくなる。ここでは
//! 画面に映っている行だけを描く（`show_rows` が担当する）。行が何万あっても、
//! 1 フレームで触るのは見えている数十行だけ。
//!
//! 左右を別々のスクロール領域にしていないのは、同期を自前で書かずに済ませるため。
//! 1 つの領域の中に左右の列を並べれば、ずれようがない。

use crate::{App, State};
use deepcompare_engine::compare::Row;
use deepcompare_engine::inline::{Span, SpanKind};
use deepcompare_engine::text::DecodedText;
use egui::text::LayoutJob;
use egui::{Color32, FontId, RichText, TextFormat, TextStyle};

/// 行番号の桁が変わっても列がずれないよう、幅は固定で確保する。
const GUTTER_WIDTH: f32 = 56.0;
const SCORE_WIDTH: f32 = 52.0;

// 暗い背景に載せる前提の配色。
const BG_CHANGED: Color32 = Color32::from_rgb(46, 51, 80);
const BG_REMOVED: Color32 = Color32::from_rgb(64, 38, 42);
const BG_ADDED: Color32 = Color32::from_rgb(34, 58, 42);
const FG_NORMAL: Color32 = Color32::from_rgb(214, 216, 222);
const FG_DIM: Color32 = Color32::from_rgb(126, 131, 143);
/// 行内で変わった部分。旧実装と同じオレンジ。
const FG_INLINE: Color32 = Color32::from_rgb(255, 165, 0);

pub fn draw(app: &mut App, ui: &mut egui::Ui) {
    egui::Panel::top("controls").show(ui, |ui| controls(app, ui));
    egui::Panel::bottom("status").show(ui, |ui| status(app, ui));
    egui::CentralPanel::default().show(ui, |ui| match &app.state {
        State::LoadingModel => centered(ui, "モデルを読み込んでいます…"),
        State::Idle => centered(
            ui,
            "比較する 2 つのファイルを指定してください（ドラッグ＆ドロップ可）",
        ),
        State::Working(phase) => centered(ui, phase_label(*phase)),
        State::Failed(message) => {
            centered_colored(ui, message, Color32::from_rgb(240, 120, 120));
        }
        State::Done(finished) => {
            let finished = finished.as_ref();
            table(
                ui,
                &finished.comparison.rows,
                &app.visible_rows,
                &finished.left,
                &finished.right,
            );
        }
    });
}

fn controls(app: &mut App, ui: &mut egui::Ui) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("ファイル1:");
        ui.add(egui::TextEdit::singleline(&mut app.left_path).desired_width(320.0));
        if ui.button("参照").clicked() {
            if let Some(path) = pick_file() {
                app.left_path = path;
            }
        }
        ui.separator();
        ui.label("ファイル2:");
        ui.add(egui::TextEdit::singleline(&mut app.right_path).desired_width(320.0));
        if ui.button("参照").clicked() {
            if let Some(path) = pick_file() {
                app.right_path = path;
            }
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let busy = matches!(app.state, State::Working(_) | State::LoadingModel);
        if ui
            .add_enabled(!busy, egui::Button::new("比較開始"))
            .clicked()
        {
            app.request_compare();
        }
        if ui
            .checkbox(&mut app.changes_only, "変更のある行だけ表示")
            .changed()
        {
            app.rebuild_visible_rows();
        }
        ui.separator();
        ui.label("対応とみなす類似度:").on_hover_text(
            "これを下回る行同士は、対にせず削除と追加として別々に並べる。\n\
                 短いコード行は似ていても値が伸びにくいので、対応が取れないときは下げる。",
        );
        ui.add(
            egui::Slider::new(&mut app.pair_threshold, 0.0..=0.95)
                .fixed_decimals(2)
                .step_by(0.01),
        );
        if busy {
            ui.spinner();
        }
    });
    ui.add_space(4.0);
}

fn status(app: &App, ui: &mut egui::Ui) {
    ui.add_space(2.0);
    ui.horizontal(|ui| match &app.state {
        State::Done(finished) => {
            let stats = &finished.comparison.stats;
            ui.label(
                RichText::new(format!(
                    "{} 行  /  一致 {} 行  /  埋め込み {} 行  /  {:.2} 秒",
                    stats.rows,
                    stats.identical_lines,
                    stats.embedded_lines,
                    finished.elapsed.as_secs_f32()
                ))
                .small(),
            );
            ui.separator();
            // 符号化と改行コードを出すのは、「中身は同じなのに全行差分になる」
            // という混乱の原因がここで一目で分かるから。
            ui.label(
                RichText::new(format!(
                    "左: {} / {}   右: {} / {}",
                    finished.left.encoding,
                    finished.left.line_ending,
                    finished.right.encoding,
                    finished.right.line_ending,
                ))
                .small(),
            );
            if stats.skipped_blocks > 0 {
                ui.separator();
                ui.label(
                    RichText::new(format!(
                        "{} 箇所は大きすぎるため意味的な対応付けを省略",
                        stats.skipped_blocks
                    ))
                    .small()
                    .color(Color32::from_rgb(220, 180, 100)),
                );
            }
        }
        _ => {
            ui.label(RichText::new("DeepCompare").small().color(FG_DIM));
        }
    });
    ui.add_space(2.0);
}

fn phase_label(phase: deepcompare_engine::compare::Phase) -> &'static str {
    use deepcompare_engine::compare::Phase::*;
    match phase {
        Segmenting => "差分を切り分けています…",
        Embedding => "変更のある行を解析しています…",
        Aligning => "行を対応付けています…",
        Done => "仕上げています…",
    }
}

fn centered(ui: &mut egui::Ui, text: &str) {
    centered_colored(ui, text, FG_DIM);
}

fn centered_colored(ui: &mut egui::Ui, text: &str, color: Color32) {
    ui.vertical_centered(|ui| {
        ui.add_space(ui.available_height() * 0.4);
        ui.label(RichText::new(text).color(color));
    });
}

fn table(
    ui: &mut egui::Ui,
    rows: &[Row],
    visible: &[usize],
    left: &DecodedText,
    right: &DecodedText,
) {
    let font = ui
        .style()
        .text_styles
        .get(&TextStyle::Monospace)
        .cloned()
        .unwrap_or_else(|| FontId::monospace(12.0));
    // 行の高さは全行で同じでなければならない。見えている行だけを描く方式は、
    // 「n 行目は n * 高さ の位置にある」という前提で成り立っている。
    let (row_height, char_width) = ui.ctx().fonts_mut(|f| {
        (
            f.row_height(&font) + 2.0,
            f.glyph_width(&font, ' ').max(1.0),
        )
    });

    // 横スクロールできるよう、最長行に合わせて内容の幅を決める。折り返して
    // しまうと行の高さが揃わず、見えている行だけを描く方式が使えなくなる。
    let longest = left
        .lines
        .iter()
        .chain(&right.lines)
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0);
    let text_width = (longest as f32 * char_width).max(240.0);
    let content_width = (GUTTER_WIDTH + text_width) * 2.0 + SCORE_WIDTH + 24.0;

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show_rows(ui, row_height, visible.len(), |ui, range| {
            ui.set_width(content_width.max(ui.available_width()));
            for index in range {
                draw_row(
                    ui,
                    &rows[visible[index]],
                    left,
                    right,
                    &font,
                    row_height,
                    text_width,
                );
            }
        });
}

fn draw_row(
    ui: &mut egui::Ui,
    row: &Row,
    left: &DecodedText,
    right: &DecodedText,
    font: &FontId,
    row_height: f32,
    text_width: f32,
) {
    let background = match (row.left, row.right) {
        (Some(_), Some(_)) if row.is_unchanged() => None,
        (Some(_), Some(_)) => Some(BG_CHANGED),
        (Some(_), None) => Some(BG_REMOVED),
        (None, Some(_)) => Some(BG_ADDED),
        (None, None) => None,
    };

    let full = egui::vec2(ui.available_width().max(1.0), row_height);
    let (rect, _) = ui.allocate_exact_size(full, egui::Sense::hover());
    if let Some(color) = background {
        ui.painter().rect_filled(rect, 0.0, color);
    }

    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    child.spacing_mut().item_spacing.x = 0.0;

    cell_number(&mut child, row.left, row_height);
    cell_text(
        &mut child,
        row.left.map(|i| left.lines[i].as_str()),
        &row.left_spans,
        font,
        text_width,
        row_height,
    );
    cell_number(&mut child, row.right, row_height);
    cell_text(
        &mut child,
        row.right.map(|i| right.lines[i].as_str()),
        &row.right_spans,
        font,
        text_width,
        row_height,
    );
    cell_score(&mut child, row, row_height);
}

fn cell_number(ui: &mut egui::Ui, line: Option<usize>, height: f32) {
    let text = line.map(|i| (i + 1).to_string()).unwrap_or_default();
    ui.allocate_ui_with_layout(
        egui::vec2(GUTTER_WIDTH, height),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            ui.add_space(6.0);
            ui.label(RichText::new(text).monospace().small().color(FG_DIM));
        },
    );
}

fn cell_text(
    ui: &mut egui::Ui,
    line: Option<&str>,
    spans: &[Span],
    font: &FontId,
    width: f32,
    height: f32,
) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, height),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.add_space(6.0);
            match line {
                // 空きの側。何も無いことが分かればよい。
                None => {
                    ui.label(RichText::new("").monospace());
                }
                Some(text) => {
                    ui.label(build_job(text, spans, font));
                }
            }
        },
    );
}

fn cell_score(ui: &mut egui::Ui, row: &Row, height: f32) {
    let text = match row.score {
        // 文字列として完全一致した行。数値を出しても意味がないので記号にする。
        Some(score) if row.is_unchanged() => {
            let _ = score;
            "=".to_owned()
        }
        Some(score) => format!("{score:.2}"),
        None => String::new(),
    };
    ui.allocate_ui_with_layout(
        egui::vec2(SCORE_WIDTH, height),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| {
            ui.add_space(6.0);
            ui.label(RichText::new(text).monospace().small().color(FG_DIM));
        },
    );
}

/// 行を、変更部分だけ色を変えた 1 つのテキストとして組み立てる。
///
/// 旧実装はここで HTML を作っていたため、`<` や `&` を含む行が壊れていた。
/// 色は書式として与え、本文は一切加工しない。
fn build_job(text: &str, spans: &[Span], font: &FontId) -> LayoutJob {
    let mut job = LayoutJob::default();
    // 折り返さない。行の高さを一定に保つことが、見えている行だけを描く前提になる。
    job.wrap.max_width = f32::INFINITY;

    if spans.is_empty() {
        job.append(text, 0.0, format(font, FG_NORMAL));
        return job;
    }
    for span in spans {
        let color = match span.kind {
            SpanKind::Equal => FG_NORMAL,
            SpanKind::Changed => FG_INLINE,
        };
        job.append(&text[span.range.clone()], 0.0, format(font, color));
    }
    job
}

fn format(font: &FontId, color: Color32) -> TextFormat {
    TextFormat {
        font_id: font.clone(),
        color,
        ..Default::default()
    }
}

fn pick_file() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("比較するファイルを選択")
        .pick_file()
        .map(|path| path.display().to_string())
}
