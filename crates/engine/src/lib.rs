//! DeepCompare の比較エンジン。GUI には一切依存しない。
//!
//! ここを GUI から切り離しているのは、旧 Python 実装ではアライメントも差分整形も
//! PyQt のウィジェット生成と同じ関数に同居していて、GUI なしでは一行も検証できな
//! かったため。

pub mod align;
pub mod bert;
pub mod compare;
pub mod embed;
pub mod inline;
pub mod text;
pub mod weights;

pub use align::{Pair, Segment};
pub use compare::{compare, CompareOptions, Comparison, Phase, Row, Stats};
pub use embed::Embedder;
pub use inline::{inline_diff, Span, SpanKind};
pub use text::{decode_file, DecodedText, Encoding, LineEnding};
