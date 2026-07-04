"""
overview_captions.py  -  code-rendered caption PNGs for the gig overview.

Renders transparent 1080x1920 PNG overlays (title card, outro card) with PIL,
so no ffmpeg drawtext/fontconfig (segfaults on this box). MINIMAL + mature per
the user's feedback (v2 thread was cringy, Impact font was childish): clean
Bahnschrift, wide letter-spacing, a soft drop-shadow (not a heavy stroke).
Text passed in by caller (gitignored workdir) - no band strings here.
"""
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
PURPLE = (170, 66, 240, 255)   # Rock-the-Straits title purple (~#AA42F0)
WHITE = (248, 248, 250, 255)
FONTS = r"C:\Windows\Fonts"
FACE = "bahnschrift.ttf"
ANTON = None   # set by caller to the Anton-Regular.ttf path (the matched Rock-the-Straits face)

def _font(size):
    return ImageFont.truetype(os.path.join(FONTS, FACE), size)

def _af(size):
    return ImageFont.truetype(ANTON, size) if ANTON and os.path.exists(ANTON) else _font(size)

def _new():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))

def _tracked(d, y, text, font, fill, tracking, stroke_w=1, shadow=(3, 4)):
    """Draw uppercase letter-spaced text, horizontally centred, top at y."""
    text = text.upper()
    widths = [d.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = (W - total) / 2
    for ch, w in zip(text, widths):
        if shadow:
            d.text((x + shadow[0], y + shadow[1]), ch, font=font, fill=(0, 0, 0, 140),
                   stroke_width=stroke_w, stroke_fill=(0, 0, 0, 140))
        d.text((x, y), ch, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=(10, 8, 16, 220))
        x += w + tracking
    return total

def render_title(lines, path):
    """RtS-style open title: INTRODUCING (white) over the BAND name in purple Anton,
    with VARIED sizing - connector words (<=3 ch, e.g. AND) small, main words HUGE -
    matching Rock the Straits. Block vertically centred. lines[0]=band, [1:]=venue/date."""
    img = _new(); d = ImageDraw.Draw(img)
    words = lines[0].split()
    sizes = [104 if len(w) <= 3 else 210 for w in words]
    gaps = [int(s * 0.84) for s in sizes]
    block_h = sum(gaps)
    _tracked(d, int(H * 0.30) - 150, "INTRODUCING", _af(70), WHITE, tracking=9, stroke_w=1)
    y = int(H * 0.50) - block_h // 2       # vertically centre the band block
    for w, s, g in zip(words, sizes, gaps):
        _tracked(d, y, w, _af(s), PURPLE, tracking=2, stroke_w=1)
        y += g
    y += 24
    for j, ln in enumerate(lines[1:]):
        _tracked(d, y + j * 58, ln, _font(44), WHITE, tracking=10, stroke_w=0)
    img.save(path)

def render_outro(band, venue, date, handle, path):
    img = _new(); d = ImageDraw.Draw(img)
    words = band.split(); cy = int(H * 0.36); fb = _af(178)
    for i, w in enumerate(words):
        _tracked(d, cy + i * 162, w, fb, PURPLE if i == len(words) - 1 else WHITE, tracking=2, stroke_w=1)
    y = cy + 162 * len(words) + 34
    _tracked(d, y, f"{venue}   {date}", _font(52), WHITE, tracking=12, stroke_w=0)
    _tracked(d, y + 74, handle, _font(44), PURPLE, tracking=8, stroke_w=0)
    img.save(path)

def render_line(text, path, pos="lower"):
    """Kept for compatibility; single mid line (unused when thread is off)."""
    img = _new(); d = ImageDraw.Draw(img)
    f = _font(120); cy = int(H * 0.80)
    _tracked(d, cy, text, f, WHITE, tracking=8, stroke_w=1)
    img.save(path)

def render_flash(path, alpha=170):
    Image.new("RGBA", (W, H), (255, 255, 255, alpha)).save(path)

if __name__ == "__main__":
    import sys, json
    meta = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    render_title(["BAND NAME", "VENUE", "DATE"], f"{out}/title.png")
    render_outro(meta["band_display"], meta["venue"], meta["date"], meta["handle"], f"{out}/outro.png")
    print("captions ->", out)
