// Windows でコンソールウィンドウを出さない（旧実装が launch.bat で pythonw.exe を
// 探していたのと同じ目的を、バイナリ側の属性ひとつで済ませる）。
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod cli;
mod fonts;
mod screenshot;
mod view;
mod worker;

use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

use deepcompare_engine::compare::{CompareOptions, Phase};
use worker::{Event, Finished, Request};

const USAGE: &str = "\
DeepCompare - 意味的な類似度で行を対応付けるコード比較ツール

  deepcompare [左 右]              GUI を開く（引数 2 つならそのまま比較する）
  deepcompare --print 左 右        画面を開かず、比較結果をテキストで出す
  deepcompare --font-check         日本語を表示できる書体があるかを調べる
  deepcompare --screenshot <png> 左 右
                                 GUI を開いて比較し、描画結果を PNG に保存して終了

オプション
  -o <パス>        結果をファイルへ書く。Windows の GUI 版 exe は標準出力が
                   繋がらないため、遠隔から結果を回収するときはこれを使う
  --threshold <値> 対応とみなす類似度の下限（既定 0.50）
  -h, --help       この説明
";

fn main() -> eframe::Result {
    let raw: Vec<String> = std::env::args().skip(1).collect();

    // 画面を開かない経路を先に捌く。GUI を初期化する前に分岐しないと、
    // 表示できない環境で使えなくなる。
    if let Some(code) = run_headless(&raw) {
        std::process::exit(code);
    }

    // 撮影して終わる指定。GUI は普通に立ち上げるので、写るのは実物と同じ描画。
    let capture = value_of(&raw, "--screenshot").map(PathBuf::from);
    // 旧実装と同じく、引数 2 つで比較対象を渡せる。オプションとその値は除く。
    let args: Vec<PathBuf> = positional(&raw).into_iter().map(PathBuf::from).collect();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 760.0])
            .with_min_inner_size([720.0, 420.0])
            .with_title("DeepCompare"),
        ..Default::default()
    };
    eframe::run_native(
        "DeepCompare",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_theme(egui::Theme::Dark);
            if fonts::install_cjk_font(&cc.egui_ctx).is_none() {
                eprintln!("日本語を表示できる書体が見つからなかった。日本語は表示されない。");
            }
            Ok(Box::new(App::new(&cc.egui_ctx, args, capture)))
        }),
    )
}

/// オプションとその値を取り除いた、位置引数だけの列。
fn positional(args: &[String]) -> Vec<String> {
    const TAKES_VALUE: &[&str] = &["-o", "--threshold", "--screenshot"];
    let mut out = Vec::new();
    let mut skip_next = false;
    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if TAKES_VALUE.contains(&arg.as_str()) {
            skip_next = true;
        } else if !arg.starts_with('-') {
            out.push(arg.clone());
        }
    }
    out
}

/// 画面を開かずに済む要求なら処理して終了コードを返す。GUI を開くなら `None`。
fn run_headless(args: &[String]) -> Option<i32> {
    let wants = |flag: &str| args.iter().any(|a| a == flag);

    if wants("-h") || wants("--help") {
        print!("{USAGE}");
        return Some(0);
    }

    let output = value_of(args, "-o").map(PathBuf::from);

    if wants("--font-check") {
        return Some(report(cli::run_font_check(output.as_deref())));
    }

    if !wants("--print") {
        return None;
    }

    let positional = positional(args);
    if positional.len() < 2 {
        eprintln!("--print には比較する 2 つのファイルが必要です\n\n{USAGE}");
        return Some(2);
    }

    let threshold = value_of(args, "--threshold")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or_else(|| CompareOptions::default().pair_threshold);

    Some(report(cli::run_compare(
        cli::CliOptions {
            left: PathBuf::from(&positional[0]),
            right: PathBuf::from(&positional[1]),
            output,
            pair_threshold: threshold,
        },
        worker::WEIGHTS,
        worker::TOKENIZER,
    )))
}

fn value_of(args: &[String], flag: &str) -> Option<String> {
    let index = args.iter().position(|a| a == flag)?;
    args.get(index + 1).cloned()
}

fn report(result: anyhow::Result<()>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("エラー: {error:#}");
            1
        }
    }
}

/// 撮影の進行。比較が終わってから撮らないと、読み込み中の画面が写る。
struct Capture {
    path: PathBuf,
    /// 結果が出てから撮るまでに置く猶予（フレーム数）。
    ///
    /// 要求した次のフレームで画像が返るとは限らず、また最初のフレームは
    /// レイアウトが定まっていないことがあるので、数フレーム描かせてから撮る。
    settle: u32,
    requested: bool,
}

/// 比較の進み具合。UI はこれだけを見て描く。
enum State {
    LoadingModel,
    Idle,
    Working(Phase),
    Done(Box<Finished>),
    Failed(String),
}

struct App {
    left_path: String,
    right_path: String,
    state: State,
    requests: Sender<Request>,
    events: Receiver<Event>,
    /// モデルの準備ができる前に来た依頼を保持しておく。
    /// 旧実装はここで待たずに走り出し、結果としてスプラッシュ画面ごと固まっていた。
    queued: Option<Request>,
    /// 完全一致の行を畳んで、変更点だけを見る。
    changes_only: bool,
    /// 対応付けると判断する類似度の下限。
    ///
    /// 既定の 0.5 を動かせるようにしているのは、短いコード行では MiniLM の類似度が
    /// 伸びにくく、`self.config = config` と `self.settings = settings` のように
    /// 明らかに対応する行でも境界付近に落ちることがあるため。適切な値は比較する
    /// コードの性質によって変わるので、決め打ちにせず手元で動かせる形にした。
    pair_threshold: f32,
    /// `--screenshot` の保存先と進行状態。
    capture: Option<Capture>,
    /// 実際に描く行の番号。絞り込みの結果をここに持っておく。
    ///
    /// 毎フレーム絞り直すと、行数に比例した処理が描画のたびに走る。見えている行だけを
    /// 描く工夫が台無しになるので、結果が変わったときにだけ作り直す。
    visible_rows: Vec<usize>,
}

impl App {
    fn new(ctx: &egui::Context, args: Vec<PathBuf>, capture: Option<PathBuf>) -> Self {
        let (request_tx, request_rx) = channel();
        let (event_tx, event_rx) = channel();
        let ctx = ctx.clone();
        worker::spawn(request_rx, event_tx, move || ctx.request_repaint());

        let mut app = Self {
            left_path: String::new(),
            right_path: String::new(),
            state: State::LoadingModel,
            requests: request_tx,
            events: event_rx,
            queued: None,
            changes_only: false,
            pair_threshold: CompareOptions::default().pair_threshold,
            capture: capture.map(|path| Capture {
                path,
                settle: 3,
                requested: false,
            }),
            visible_rows: Vec::new(),
        };
        if args.len() >= 2 {
            app.left_path = args[0].display().to_string();
            app.right_path = args[1].display().to_string();
            app.request_compare();
        }
        app
    }

    fn request_compare(&mut self) {
        if self.left_path.trim().is_empty() || self.right_path.trim().is_empty() {
            self.state = State::Failed("両方のファイルを指定してください。".to_owned());
            return;
        }
        let request = Request::Compare {
            left: PathBuf::from(self.left_path.trim()),
            right: PathBuf::from(self.right_path.trim()),
            options: CompareOptions {
                pair_threshold: self.pair_threshold,
                ..CompareOptions::default()
            },
        };
        match self.state {
            // モデルがまだ来ていないなら、投げずに預かる。
            State::LoadingModel => self.queued = Some(request),
            _ => {
                self.state = State::Working(Phase::Segmenting);
                let _ = self.requests.send(request);
            }
        }
    }

    fn drain_events(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                Event::Ready => {
                    self.state = State::Idle;
                    if let Some(request) = self.queued.take() {
                        self.state = State::Working(Phase::Segmenting);
                        let _ = self.requests.send(request);
                    }
                }
                Event::Progress(phase) => self.state = State::Working(phase),
                Event::Finished(finished) => {
                    self.state = State::Done(finished);
                    self.rebuild_visible_rows();
                }
                Event::Failed(message) => self.state = State::Failed(message),
            }
        }
    }

    /// 絞り込みの結果を作り直す。結果が届いたときと、絞り込みを切り替えたときだけ。
    fn rebuild_visible_rows(&mut self) {
        let State::Done(finished) = &self.state else {
            self.visible_rows.clear();
            return;
        };
        let rows = &finished.comparison.rows;
        self.visible_rows = if self.changes_only {
            (0..rows.len())
                .filter(|&i| !rows[i].is_unchanged())
                .collect()
        } else {
            (0..rows.len()).collect()
        };
    }

    /// ウィンドウへ落とされたファイルを、空いている方から順に入れる。
    fn accept_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        if dropped.is_empty() {
            return;
        }
        let mut iter = dropped.into_iter();
        // 2 つまとめて落とされたら左右へ割り振る。
        if let Some(path) = iter.next() {
            if self.left_path.trim().is_empty() {
                self.left_path = path.display().to_string();
            } else if self.right_path.trim().is_empty() {
                self.right_path = path.display().to_string();
            } else {
                self.left_path = path.display().to_string();
            }
        }
        if let Some(path) = iter.next() {
            self.right_path = path.display().to_string();
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_events();
        self.accept_dropped_files(ui.ctx());
        view::draw(self, ui);
        self.advance_capture(ui.ctx());
    }
}

impl App {
    /// `--screenshot` の進行。比較が終わったら数フレーム待って撮影を要求し、
    /// 返ってきた画像を保存して終了する。
    fn advance_capture(&mut self, ctx: &egui::Context) {
        let Some(capture) = &mut self.capture else {
            return;
        };

        // 撮影を要求した後は、画像が届くのを待つだけ。
        if capture.requested {
            let shot = ctx.input(|i| {
                i.raw.events.iter().find_map(|event| match event {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(image) = shot {
                let [width, height] = image.size;
                let rgba: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                let result = screenshot::write_png(&capture.path, width, height, &rgba);
                match result {
                    Ok(()) => eprintln!(
                        "書き出した: {} ({width}x{height})",
                        capture.path.display()
                    ),
                    Err(error) => eprintln!("撮影に失敗: {error:#}"),
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                // 画像はまだ来ていない。描き続けさせる。
                ctx.request_repaint();
            }
            return;
        }

        // 結果が出るまでは撮らない。読み込み中の画面を保存しても意味がない。
        match &self.state {
            State::Done(_) | State::Failed(_) => {}
            _ => {
                ctx.request_repaint();
                return;
            }
        }

        if capture.settle > 0 {
            capture.settle -= 1;
            ctx.request_repaint();
            return;
        }
        capture.requested = true;
        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        ctx.request_repaint();
    }
}
