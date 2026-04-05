// Minimal 4×7 bitmap font for headless video overlays.
//
// Each glyph is 4 pixels wide and 7 pixels tall. A single row is stored as
// a `u8` where bits [3:0] represent the 4 pixels (bit 3 = leftmost, bit 0 =
// rightmost). There is a 1-pixel gap between characters.
//
// Only the ASCII characters used in simulation overlays are defined; all
// others fall back to the '?' glyph.

use image::{Rgb, RgbImage};

/// Pixel width of one glyph cell.
pub const GLYPH_W: u32 = 4;
/// Pixel height of one glyph cell.
pub const GLYPH_H: u32 = 7;
/// Pixel gap between adjacent glyphs.
pub const GLYPH_GAP: u32 = 1;

/// Returns the 7 row bitmaps for `ch`.
///
/// Each byte's low 4 bits describe 4 pixels: bit 3 = leftmost, bit 0 =
/// rightmost. Falls back to '?' for unknown characters.
pub fn glyph_rows(ch: char) -> [u8; 7] {
    match ch {
        // Punctuation / symbols
        ' ' => [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0],
        '(' => [0x2, 0x4, 0x8, 0x8, 0x8, 0x4, 0x2],
        ')' => [0x4, 0x2, 0x1, 0x1, 0x1, 0x2, 0x4],
        '-' => [0x0, 0x0, 0x0, 0xF, 0x0, 0x0, 0x0],
        '.' => [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x6],
        ':' => [0x0, 0x6, 0x6, 0x0, 0x6, 0x6, 0x0],
        '=' => [0x0, 0xF, 0x0, 0xF, 0x0, 0x0, 0x0],
        '|' => [0x6, 0x6, 0x6, 0x6, 0x6, 0x6, 0x6],
        // Digits
        '0' => [0x6, 0x9, 0x9, 0x9, 0x9, 0x6, 0x0],
        '1' => [0x2, 0x6, 0x2, 0x2, 0x2, 0x7, 0x0],
        '2' => [0x6, 0x9, 0x1, 0x2, 0x4, 0xF, 0x0],
        '3' => [0x6, 0x9, 0x1, 0x3, 0x1, 0x9, 0x6],
        '4' => [0x0, 0x9, 0x9, 0xF, 0x1, 0x1, 0x0],
        '5' => [0xF, 0x8, 0xE, 0x1, 0x1, 0x9, 0x6],
        '6' => [0x7, 0x8, 0x8, 0xE, 0x9, 0x9, 0x6],
        '7' => [0xF, 0x1, 0x2, 0x2, 0x4, 0x4, 0x0],
        '8' => [0x6, 0x9, 0x9, 0x6, 0x9, 0x9, 0x6],
        '9' => [0x6, 0x9, 0x9, 0x7, 0x1, 0x1, 0x6],
        // Lowercase letters (only those used in overlay strings)
        'a' => [0x0, 0x6, 0x1, 0x7, 0x9, 0x7, 0x0],
        'd' => [0x1, 0x1, 0x7, 0x9, 0x9, 0x7, 0x0],
        'e' => [0x0, 0x6, 0x9, 0xF, 0x8, 0x6, 0x0],
        'f' => [0x3, 0x4, 0xF, 0x4, 0x4, 0x4, 0x0],
        'h' => [0x8, 0x8, 0xE, 0x9, 0x9, 0x9, 0x0],
        'i' => [0x2, 0x0, 0x2, 0x2, 0x2, 0x2, 0x0],
        'k' => [0x0, 0x9, 0xA, 0xC, 0xA, 0x9, 0x0],
        'l' => [0x4, 0x4, 0x4, 0x4, 0x4, 0x7, 0x0],
        'm' => [0x0, 0xF, 0xA, 0xA, 0xA, 0xA, 0x0],
        'n' => [0x0, 0xE, 0x9, 0x9, 0x9, 0x9, 0x0],
        'o' => [0x0, 0x6, 0x9, 0x9, 0x9, 0x6, 0x0],
        'p' => [0x0, 0xE, 0x9, 0x9, 0xE, 0x8, 0x8],
        'r' => [0x0, 0x7, 0x8, 0x8, 0x8, 0x8, 0x0],
        's' => [0x0, 0x6, 0x8, 0x6, 0x1, 0x6, 0x0],
        't' => [0x4, 0xF, 0x4, 0x4, 0x4, 0x3, 0x0],
        'w' => [0x0, 0x9, 0x9, 0xA, 0x6, 0x0, 0x0],
        'x' => [0x0, 0x9, 0x6, 0x6, 0x9, 0x0, 0x0],
        'y' => [0x0, 0x9, 0x9, 0x7, 0x1, 0x1, 0x6],
        // Fallback '?'
        _ => [0x6, 0x9, 0x1, 0x3, 0x2, 0x0, 0x2],
    }
}

/// Width in pixels of a rendered string at the given `scale`.
pub fn text_width(text: &str, scale: u32) -> u32 {
    let n = text.chars().count() as u32;
    if n == 0 {
        0
    } else {
        n * (GLYPH_W + GLYPH_GAP) * scale - GLYPH_GAP * scale
    }
}

/// Choose a text-scale factor based on the image pixel scale of the
/// simulation grid (each voxel is `img_scale` pixels wide).
pub fn text_scale_for(img_scale: u32) -> u32 {
    if img_scale >= 8 { 2 } else { 1 }
}

/// Height in pixels of the overlay strip (two text lines + padding).
pub fn overlay_strip_height(text_scale: u32) -> u32 {
    let line_h = GLYPH_H * text_scale;
    let pad = 3u32;
    2 * line_h + 3 * pad // top_pad + line1 + mid_pad + line2 + bot_pad
}

/// Draw `text` at pixel position `(px, py)` with foreground colour `fg`.
///
/// Each font pixel is rendered as a `scale × scale` block. Pixels outside
/// the image bounds are silently clipped.
pub fn draw_text(img: &mut RgbImage, text: &str, px: u32, py: u32, scale: u32, fg: [u8; 3]) {
    let color = Rgb(fg);
    let mut cx = px;
    for ch in text.chars() {
        let rows = glyph_rows(ch);
        for (row_idx, &row_bits) in rows.iter().enumerate() {
            for col in 0..GLYPH_W {
                // Bit 3 = leftmost pixel, bit 0 = rightmost
                if (row_bits >> (GLYPH_W - 1 - col)) & 1 != 0 {
                    let base_x = cx + col * scale;
                    let base_y = py + row_idx as u32 * scale;
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let fx = base_x + sx;
                            let fy = base_y + sy;
                            if fx < img.width() && fy < img.height() {
                                img.put_pixel(fx, fy, color);
                            }
                        }
                    }
                }
            }
        }
        cx += (GLYPH_W + GLYPH_GAP) * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_width_empty() {
        assert_eq!(text_width("", 2), 0);
    }

    #[test]
    fn text_width_single_char() {
        // 4px wide + 0 gap = 4 * scale
        assert_eq!(text_width("0", 2), 8);
    }

    #[test]
    fn text_width_multi_char() {
        // 3 chars: 3*(4+1)*scale - 1*scale = 14*scale
        assert_eq!(text_width("123", 1), 14);
    }

    #[test]
    fn draw_text_does_not_panic_oob() {
        let mut img = RgbImage::new(4, 8);
        // This should silently clip without panicking.
        draw_text(&mut img, "Hello world!", 0, 0, 1, [255, 255, 255]);
    }

    #[test]
    fn glyph_rows_fallback_for_unknown() {
        let unknown = glyph_rows('€');
        let fallback = glyph_rows('?');
        // Unknown chars map to the fallback '?' glyph.
        assert_ne!(unknown[0], 0, "fallback should not be blank");
        assert_eq!(unknown, fallback);
    }

    #[test]
    fn overlay_strip_height_nonzero() {
        assert!(overlay_strip_height(1) > 0);
        assert!(overlay_strip_height(2) > overlay_strip_height(1));
    }
}
