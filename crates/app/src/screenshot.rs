//! 自分自身の描画結果を PNG として保存する。
//!
//! 画面の写真を撮るのではなく、アプリが描いた内容をそのまま書き出す。この違いが
//! 効く場面がある。Windows がロックされていると、表示中のデスクトップは Winlogon の
//! セキュアデスクトップに切り替わるので、画面を撮ってもログイン画面しか写らない。
//! アプリ自身のフレームバッファを読めば、デスクトップの状態と無関係に本物の描画が
//! 得られる。遠隔での確認や、CI での見た目の回帰検出にも使える。
//!
//! PNG は自前で組み立てる。画素はもう手元にあるので、そのために画像処理の依存を
//! 増やす必要がない。圧縮は行わず、zlib の非圧縮ブロックとして格納する。ファイルは
//! 大きくなるが、確認用の一時的な出力なので支障はない。

use anyhow::{Context, Result};
use std::path::Path;

/// RGBA の画素列を PNG として書き出す。
pub fn write_png(path: &Path, width: usize, height: usize, rgba: &[u8]) -> Result<()> {
    anyhow::ensure!(
        rgba.len() == width * height * 4,
        "画素数が寸法と合わない: {} != {}x{}x4",
        rgba.len(),
        width,
        height
    );

    let mut png = Vec::with_capacity(rgba.len() + 4096);
    png.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR: 8 ビット / RGBA / 非インタレース。
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &ihdr);

    // 各行の先頭にフィルタ種別（0 = なし）を置いたものが zlib の入力になる。
    let mut raw = Vec::with_capacity(height * (1 + width * 4));
    for y in 0..height {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * width * 4..(y + 1) * width * 4]);
    }
    chunk(&mut png, b"IDAT", &zlib_store(&raw));
    chunk(&mut png, b"IEND", &[]);

    std::fs::write(path, &png).with_context(|| format!("書き込めない: {}", path.display()))?;
    Ok(())
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// 圧縮せずに zlib ストリームとして包む（deflate の「格納」ブロック）。
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 16);
    out.extend_from_slice(&[0x78, 0x01]); // zlib ヘッダ（圧縮なし相当）

    // 格納ブロックは 1 つあたり 65535 バイトまで。
    let mut chunks = data.chunks(65535).peekable();
    if data.is_empty() {
        out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
    }
    while let Some(part) = chunks.next() {
        let last = chunks.peek().is_none();
        out.push(if last { 1 } else { 0 });
        out.extend_from_slice(&(part.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(part.len() as u16)).to_le_bytes());
        out.extend_from_slice(part);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PNG 仕様に載っている検査値。ここが合っていれば chunk の CRC も正しい。
    #[test]
    fn crc32_matches_the_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn adler32_matches_the_known_value() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn written_file_starts_with_the_png_signature_and_ends_with_iend() {
        let dir = std::env::temp_dir().join("dc-png-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.png");
        let rgba = vec![0xABu8; 4 * 4 * 4];
        write_png(&path, 4, 4, &rgba).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        assert_eq!(&bytes[bytes.len() - 8..bytes.len() - 4], b"IEND");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mismatched_pixel_count_is_rejected() {
        let path = std::env::temp_dir().join("dc-png-bad.png");
        assert!(write_png(&path, 4, 4, &[0u8; 3]).is_err());
    }
}
