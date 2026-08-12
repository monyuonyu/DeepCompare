//! 比較を別スレッドで回す。
//!
//! 旧 Python 実装は GUI スレッドで直接 `compare_files()` を呼んでいたため、比較中は
//! ウィンドウが固まった。さらに起動引数つきで開くとモデル読み込み前に比較が走り、
//! `get_line_embeddings` が同期的にモデルを読むので、スプラッシュ画面ごと固まっていた。
//!
//! ここではモデルの読み込みも比較もすべてこのスレッドが持ち、UI とはメッセージだけを
//! やり取りする。UI 側が待つことはない。

use anyhow::{Context, Result};
use deepcompare_engine::compare::{Comparison, Phase};
use deepcompare_engine::text::DecodedText;
use deepcompare_engine::{compare, CompareOptions, Embedder};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};

/// exe に埋め込むモデル。ここが唯一の埋め込み地点。
pub static WEIGHTS: &[u8] = include_bytes!("../../../assets/minilm.dcm");
pub static TOKENIZER: &[u8] = include_bytes!("../../../assets/tokenizer.json");

pub enum Request {
    Compare {
        left: PathBuf,
        right: PathBuf,
        options: CompareOptions,
    },
}

pub enum Event {
    /// モデルの読み込みが終わり、比較を受け付けられる状態になった。
    Ready,
    Progress(Phase),
    Finished(Box<Finished>),
    Failed(String),
}

pub struct Finished {
    pub left: DecodedText,
    pub right: DecodedText,
    pub left_path: PathBuf,
    pub right_path: PathBuf,
    pub comparison: Comparison,
    pub elapsed: std::time::Duration,
}

/// 作業スレッドを起こす。`repaint` は UI を起こし直すための呼び出し。
pub fn spawn(
    requests: Receiver<Request>,
    events: Sender<Event>,
    repaint: impl Fn() + Send + 'static,
) {
    std::thread::spawn(move || {
        let notify = |event: Event| {
            // UI が閉じていれば送信は失敗する。作業スレッドはそこで畳む。
            let ok = events.send(event).is_ok();
            repaint();
            ok
        };

        let embedder = match Embedder::from_bytes(WEIGHTS, TOKENIZER) {
            Ok(embedder) => embedder,
            Err(error) => {
                notify(Event::Failed(format!("モデルを読み込めない: {error:#}")));
                return;
            }
        };
        if !notify(Event::Ready) {
            return;
        }

        for request in requests {
            let Request::Compare {
                left,
                right,
                options,
            } = request;
            let started = std::time::Instant::now();
            let result = run(&left, &right, &embedder, options, &|phase| {
                let _ = events.send(Event::Progress(phase));
                repaint();
            });
            let event = match result {
                Ok((left_text, right_text, comparison)) => Event::Finished(Box::new(Finished {
                    left: left_text,
                    right: right_text,
                    left_path: left,
                    right_path: right,
                    comparison,
                    elapsed: started.elapsed(),
                })),
                Err(error) => Event::Failed(format!("{error:#}")),
            };
            if !notify(event) {
                return;
            }
        }
    });
}

fn run(
    left_path: &PathBuf,
    right_path: &PathBuf,
    embedder: &Embedder,
    options: CompareOptions,
    progress: &dyn Fn(Phase),
) -> Result<(DecodedText, DecodedText, Comparison)> {
    let left = read(left_path)?;
    let right = read(right_path)?;
    let comparison = compare(&left, &right, embedder, options, progress)?;
    Ok((left, right, comparison))
}

fn read(path: &PathBuf) -> Result<DecodedText> {
    let bytes =
        std::fs::read(path).with_context(|| format!("ファイルを開けない: {}", path.display()))?;
    Ok(deepcompare_engine::decode_file(&bytes))
}
