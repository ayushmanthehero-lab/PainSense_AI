"""
utils/anatomy_map.py - PainSense AI
----------------------------------------------------------------------
Anatomy map overlays using per-muscle polygon highlighting.
Pure-Pillow approach — no native DLL dependencies (no Cairo required).
Each anatomical region paints individual muscle-shaped polygons/ellipses
onto the anatomy images for precise visual feedback.

Sources (all public domain / CC BY 4.0):
  anterior_unlabeled.png   - OpenStax, CC BY 4.0
  posterior_unlabeled.png  - OpenStax, CC BY 4.0
  gray_anterior_full.png   - Gray's Anatomy plate collage, public domain
  gray_back_lumbar.png     - Gray384 (back muscles), public domain
  gray_cervical.png        - Gray384 (neck cross-section), public domain
  gray_thigh_hip.png       - Gray430 (thigh/hip), public domain
  gray_forearm.png         - Gray418 (forearm extensors), public domain

Public API
----------
draw_region_highlight(region, mas_score) -> PIL.Image
draw_zoom_view(region, mas_score)        -> PIL.Image
get_muscle_list(region)                  -> List[str]
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter

_HERE  = Path(__file__).parent
_ASSET = _HERE.parent / "assets" / "anatomy"

# Region constants
REGION_SHOULDER = "shoulder"
REGION_ELBOW    = "elbow_wrist"
REGION_HIP_KNEE = "hip_knee"
REGION_LUMBAR   = "lumbar"
REGION_CERVICAL = "cervical"
REGION_FULL     = "full_body"
REGION_UNKNOWN  = "unknown"

# MAS colour scale
def _mas_colour(mas: float) -> Tuple[int, int, int]:
    if mas < 25:  return (76, 175, 80)    # green
    if mas < 50:  return (255, 193, 7)    # yellow
    if mas < 75:  return (255, 87, 34)    # orange
    return           (244, 67, 54)        # red

# Muscle groups per region
REGION_MUSCLES: Dict[str, List[str]] = {
    REGION_CERVICAL: ["Sternocleidomastoid","Trapezius (upper)","Splenius capitis",
                      "Semispinalis capitis","Scalenes","Levator scapulae"],
    REGION_SHOULDER: ["Deltoid","Rotator cuff","Supraspinatus","Infraspinatus",
                      "Teres minor","Subscapularis","Pectoralis major"],
    REGION_ELBOW:    ["Biceps brachii","Triceps brachii","Brachialis","Brachioradialis",
                      "Extensor carpi radialis","Flexor carpi ulnaris"],
    REGION_LUMBAR:   ["Erector spinae","Multifidus","Quadratus lumborum",
                      "Iliopsoas","Latissimus dorsi","Rhomboids"],
    REGION_HIP_KNEE: ["Gluteus maximus","Gluteus medius","Hamstrings","Quadriceps",
                      "Hip flexors","Iliotibial band"],
    REGION_FULL:     ["Full-body musculature"],
    REGION_UNKNOWN:  ["Full-body musculature"],
}

def get_muscle_list(region: str) -> List[str]:
    return REGION_MUSCLES.get(region, REGION_MUSCLES[REGION_UNKNOWN])

# Muscle shape definitions (normalized 0-1 coordinates per image)
# Shape format:
#   ('poly',  [(x0,y0), (x1,y1), ...])   - filled polygon
#   ('ellipse', cx, cy, rx, ry)          - filled ellipse
#
# Coordinates are normalised to the respective image:
#   'ant'  → anterior_unlabeled.png  (659×751)
#   'post' → posterior_unlabeled.png (427×759)
#
# Each region entry: { region_name: {'ant': [shapes], 'post': [shapes]} }

_SHAPES: Dict[str, Dict[str, list]] = {

    # ── CERVICAL ──────────────────────────────────────────────────────────
    REGION_CERVICAL: {
        "ant": [
            # Sternocleidomastoid – bilateral diagonal bands
            ("poly", [(0.46,0.16),(0.43,0.14),(0.37,0.21),(0.36,0.27),(0.41,0.26),(0.46,0.20)]),
            ("poly", [(0.54,0.16),(0.57,0.14),(0.63,0.21),(0.64,0.27),(0.59,0.26),(0.54,0.20)]),
            # Trapezius upper – fan from neck to shoulders
            ("poly", [(0.44,0.17),(0.37,0.19),(0.21,0.25),(0.16,0.29),(0.18,0.33),(0.29,0.28),(0.42,0.22)]),
            ("poly", [(0.56,0.17),(0.63,0.19),(0.79,0.25),(0.84,0.29),(0.82,0.33),(0.71,0.28),(0.58,0.22)]),
            # Anterior scalenes
            ("ellipse", 0.43, 0.21, 0.03, 0.05),
            ("ellipse", 0.57, 0.21, 0.03, 0.05),
        ],
        "post": [
            # Splenius capitis – bilateral diagonal
            ("poly", [(0.41,0.06),(0.48,0.13),(0.44,0.22),(0.37,0.19),(0.34,0.11)]),
            ("poly", [(0.59,0.06),(0.52,0.13),(0.56,0.22),(0.63,0.19),(0.66,0.11)]),
            # Semispinalis capitis – central band
            ("poly", [(0.41,0.04),(0.59,0.04),(0.61,0.18),(0.39,0.18)]),
            # Trapezius upper – posterior
            ("poly", [(0.42,0.15),(0.22,0.21),(0.11,0.27),(0.12,0.31),(0.24,0.26),(0.42,0.21)]),
            ("poly", [(0.58,0.15),(0.78,0.21),(0.89,0.27),(0.88,0.31),(0.76,0.26),(0.58,0.21)]),
            # Levator scapulae
            ("ellipse", 0.32, 0.22, 0.04, 0.07),
            ("ellipse", 0.68, 0.22, 0.04, 0.07),
        ],
    },

    # ── SHOULDER ─────────────────────────────────────────────────────────
    REGION_SHOULDER: {
        "ant": [
            # Anterior deltoid
            ("poly", [(0.14,0.23),(0.09,0.26),(0.08,0.39),(0.16,0.43),(0.23,0.37),(0.23,0.27)]),
            ("poly", [(0.86,0.23),(0.91,0.26),(0.92,0.39),(0.84,0.43),(0.77,0.37),(0.77,0.27)]),
            # Pectoralis major
            ("poly", [(0.29,0.22),(0.45,0.21),(0.45,0.39),(0.26,0.42),(0.22,0.35),(0.24,0.26)]),
            ("poly", [(0.71,0.22),(0.55,0.21),(0.55,0.39),(0.74,0.42),(0.78,0.35),(0.76,0.26)]),
            # Trapezius upper (carries into shoulder)
            ("poly", [(0.44,0.17),(0.37,0.19),(0.21,0.25),(0.16,0.29),(0.18,0.33),(0.29,0.28),(0.42,0.22)]),
            ("poly", [(0.56,0.17),(0.63,0.19),(0.79,0.25),(0.84,0.29),(0.82,0.33),(0.71,0.28),(0.58,0.22)]),
        ],
        "post": [
            # Posterior deltoid
            ("poly", [(0.08,0.24),(0.16,0.22),(0.18,0.36),(0.10,0.39),(0.07,0.32)]),
            ("poly", [(0.92,0.24),(0.84,0.22),(0.82,0.36),(0.90,0.39),(0.93,0.32)]),
            # Infraspinatus
            ("poly", [(0.12,0.27),(0.36,0.25),(0.37,0.38),(0.14,0.41),(0.09,0.34)]),
            ("poly", [(0.88,0.27),(0.64,0.25),(0.63,0.38),(0.86,0.41),(0.91,0.34)]),
            # Teres minor
            ("ellipse", 0.13, 0.38, 0.06, 0.025),
            ("ellipse", 0.87, 0.38, 0.06, 0.025),
            # Teres major
            ("ellipse", 0.14, 0.42, 0.06, 0.025),
            ("ellipse", 0.86, 0.42, 0.06, 0.025),
            # Trapezius mid
            ("poly", [(0.18,0.23),(0.44,0.20),(0.44,0.38),(0.10,0.38),(0.09,0.30)]),
            ("poly", [(0.82,0.23),(0.56,0.20),(0.56,0.38),(0.90,0.38),(0.91,0.30)]),
        ],
    },

    # ── ELBOW / WRIST ────────────────────────────────────────────────────
    REGION_ELBOW: {
        "ant": [
            # Biceps brachii
            ("ellipse", 0.17, 0.40, 0.05, 0.09),
            ("ellipse", 0.83, 0.40, 0.05, 0.09),
            # Brachialis (just below biceps)
            ("ellipse", 0.16, 0.46, 0.04, 0.04),
            ("ellipse", 0.84, 0.46, 0.04, 0.04),
            # Forearm flexors / pronator group
            ("poly", [(0.12,0.48),(0.22,0.47),(0.21,0.64),(0.14,0.66),(0.10,0.57)]),
            ("poly", [(0.88,0.48),(0.78,0.47),(0.79,0.64),(0.86,0.66),(0.90,0.57)]),
            # Brachioradialis (lateral forearm)
            ("ellipse", 0.15, 0.53, 0.03, 0.06),
            ("ellipse", 0.85, 0.53, 0.03, 0.06),
        ],
        "post": [
            # Triceps long head
            ("poly", [(0.07,0.27),(0.16,0.25),(0.17,0.48),(0.08,0.50),(0.04,0.38)]),
            ("poly", [(0.93,0.27),(0.84,0.25),(0.83,0.48),(0.92,0.50),(0.96,0.38)]),
            # Forearm extensors
            ("poly", [(0.06,0.48),(0.15,0.48),(0.16,0.64),(0.08,0.66),(0.04,0.56)]),
            ("poly", [(0.94,0.48),(0.85,0.48),(0.84,0.64),(0.92,0.66),(0.96,0.56)]),
        ],
    },

    # ── LUMBAR / BACK ────────────────────────────────────────────────────
    REGION_LUMBAR: {
        "ant": [
            # Rectus abdominis (central column)
            ("poly", [(0.43,0.38),(0.57,0.38),(0.56,0.56),(0.44,0.56)]),
            # External obliques
            ("poly", [(0.27,0.40),(0.44,0.39),(0.42,0.57),(0.26,0.59),(0.23,0.49)]),
            ("poly", [(0.73,0.40),(0.56,0.39),(0.58,0.57),(0.74,0.59),(0.77,0.49)]),
            # Iliopsoas (deep hip flexor, hinted at groin level)
            ("ellipse", 0.38, 0.58, 0.05, 0.035),
            ("ellipse", 0.62, 0.58, 0.05, 0.035),
        ],
        "post": [
            # Trapezius lower fibers
            ("poly", [(0.25,0.31),(0.75,0.31),(0.72,0.45),(0.28,0.45)]),
            # Rhomboids
            ("poly", [(0.32,0.25),(0.68,0.25),(0.65,0.37),(0.35,0.37)]),
            # Latissimus dorsi
            ("poly", [(0.11,0.35),(0.31,0.29),(0.36,0.52),(0.25,0.59),(0.12,0.53),(0.09,0.44)]),
            ("poly", [(0.89,0.35),(0.69,0.29),(0.64,0.52),(0.75,0.59),(0.88,0.53),(0.91,0.44)]),
            # Erector spinae
            ("poly", [(0.31,0.30),(0.42,0.30),(0.43,0.57),(0.30,0.58),(0.27,0.44)]),
            ("poly", [(0.69,0.30),(0.58,0.30),(0.57,0.57),(0.70,0.58),(0.73,0.44)]),
            # Multifidus
            ("ellipse", 0.37, 0.45, 0.04, 0.08),
            ("ellipse", 0.63, 0.45, 0.04, 0.08),
        ],
    },

    # ── HIP / KNEE ───────────────────────────────────────────────────────
    REGION_HIP_KNEE: {
        "ant": [
            # Rectus femoris (central thigh)
            ("poly", [(0.35,0.61),(0.42,0.60),(0.42,0.79),(0.34,0.80),(0.32,0.70)]),
            ("poly", [(0.65,0.61),(0.58,0.60),(0.58,0.79),(0.66,0.80),(0.68,0.70)]),
            # Vastus lateralis
            ("poly", [(0.27,0.62),(0.36,0.61),(0.34,0.80),(0.26,0.79),(0.24,0.70)]),
            ("poly", [(0.73,0.62),(0.64,0.61),(0.66,0.80),(0.74,0.79),(0.76,0.70)]),
            # Vastus medialis
            ("poly", [(0.42,0.63),(0.48,0.65),(0.46,0.80),(0.41,0.81),(0.38,0.73)]),
            ("poly", [(0.58,0.63),(0.52,0.65),(0.54,0.80),(0.59,0.81),(0.62,0.73)]),
            # Sartorius (diagonal)
            ("poly", [(0.37,0.60),(0.42,0.60),(0.33,0.79),(0.30,0.79)]),
            ("poly", [(0.63,0.60),(0.58,0.60),(0.67,0.79),(0.70,0.79)]),
            # Adductors
            ("poly", [(0.44,0.63),(0.50,0.65),(0.49,0.81),(0.44,0.82),(0.41,0.72)]),
            ("poly", [(0.56,0.63),(0.50,0.65),(0.51,0.81),(0.56,0.82),(0.59,0.72)]),
            # TFL
            ("ellipse", 0.29, 0.61, 0.04, 0.04),
            ("ellipse", 0.71, 0.61, 0.04, 0.04),
        ],
        "post": [
            # Gluteus maximus
            ("poly", [(0.21,0.52),(0.47,0.49),(0.48,0.63),(0.36,0.67),(0.21,0.63)]),
            ("poly", [(0.79,0.52),(0.53,0.49),(0.52,0.63),(0.64,0.67),(0.79,0.63)]),
            # Gluteus medius
            ("poly", [(0.19,0.45),(0.41,0.41),(0.43,0.54),(0.20,0.57)]),
            ("poly", [(0.81,0.45),(0.59,0.41),(0.57,0.54),(0.80,0.57)]),
            # Biceps femoris / hamstrings
            ("poly", [(0.27,0.63),(0.43,0.63),(0.43,0.81),(0.27,0.81)]),
            ("poly", [(0.73,0.63),(0.57,0.63),(0.57,0.81),(0.73,0.81)]),
            # Semimembranosus/semitendinosus (medial)
            ("poly", [(0.22,0.63),(0.29,0.63),(0.27,0.81),(0.21,0.81)]),
            ("poly", [(0.78,0.63),(0.71,0.63),(0.73,0.81),(0.79,0.81)]),
            # Iliotibial band (lateral)
            ("poly", [(0.21,0.63),(0.25,0.63),(0.28,0.81),(0.23,0.81)]),
            ("poly", [(0.79,0.63),(0.75,0.63),(0.72,0.81),(0.77,0.81)]),
        ],
    },

    # ── FULL BODY ────────────────────────────────────────────────────────
    REGION_FULL: {
        "ant": [
            # Neck
            ("poly", [(0.44,0.17),(0.37,0.19),(0.21,0.25),(0.16,0.29),(0.18,0.33),(0.29,0.28),(0.42,0.22)]),
            ("poly", [(0.56,0.17),(0.63,0.19),(0.79,0.25),(0.84,0.29),(0.82,0.33),(0.71,0.28),(0.58,0.22)]),
            # Chest
            ("poly", [(0.29,0.22),(0.45,0.21),(0.45,0.39),(0.26,0.42),(0.22,0.35),(0.24,0.26)]),
            ("poly", [(0.71,0.22),(0.55,0.21),(0.55,0.39),(0.74,0.42),(0.78,0.35),(0.76,0.26)]),
            # Deltoids
            ("poly", [(0.14,0.23),(0.09,0.26),(0.08,0.39),(0.16,0.43),(0.23,0.37),(0.23,0.27)]),
            ("poly", [(0.86,0.23),(0.91,0.26),(0.92,0.39),(0.84,0.43),(0.77,0.37),(0.77,0.27)]),
            # Arms
            ("ellipse", 0.17, 0.40, 0.05, 0.09),
            ("ellipse", 0.83, 0.40, 0.05, 0.09),
            ("poly", [(0.12,0.48),(0.22,0.47),(0.21,0.64),(0.14,0.66),(0.10,0.57)]),
            ("poly", [(0.88,0.48),(0.78,0.47),(0.79,0.64),(0.86,0.66),(0.90,0.57)]),
            # Abdomen
            ("poly", [(0.43,0.38),(0.57,0.38),(0.56,0.56),(0.44,0.56)]),
            ("poly", [(0.27,0.40),(0.44,0.39),(0.42,0.57),(0.26,0.59),(0.23,0.49)]),
            ("poly", [(0.73,0.40),(0.56,0.39),(0.58,0.57),(0.74,0.59),(0.77,0.49)]),
            # Thighs
            ("poly", [(0.27,0.62),(0.50,0.60),(0.49,0.81),(0.26,0.81),(0.24,0.70)]),
            ("poly", [(0.73,0.62),(0.50,0.60),(0.51,0.81),(0.74,0.81),(0.76,0.70)]),
        ],
        "post": [
            # Trapezius
            ("poly", [(0.18,0.23),(0.44,0.20),(0.44,0.38),(0.10,0.38),(0.09,0.30)]),
            ("poly", [(0.82,0.23),(0.56,0.20),(0.56,0.38),(0.90,0.38),(0.91,0.30)]),
            # Lats
            ("poly", [(0.11,0.35),(0.31,0.29),(0.36,0.52),(0.25,0.59),(0.12,0.53),(0.09,0.44)]),
            ("poly", [(0.89,0.35),(0.69,0.29),(0.64,0.52),(0.75,0.59),(0.88,0.53),(0.91,0.44)]),
            # Erector spinae
            ("poly", [(0.31,0.30),(0.42,0.30),(0.43,0.57),(0.30,0.58),(0.27,0.44)]),
            ("poly", [(0.69,0.30),(0.58,0.30),(0.57,0.57),(0.70,0.58),(0.73,0.44)]),
            # Glutes + hamstrings
            ("poly", [(0.21,0.52),(0.47,0.49),(0.48,0.63),(0.36,0.67),(0.21,0.63)]),
            ("poly", [(0.79,0.52),(0.53,0.49),(0.52,0.63),(0.64,0.67),(0.79,0.63)]),
            ("poly", [(0.27,0.63),(0.43,0.63),(0.43,0.81),(0.27,0.81)]),
            ("poly", [(0.73,0.63),(0.57,0.63),(0.57,0.81),(0.73,0.81)]),
            # Triceps
            ("poly", [(0.07,0.27),(0.16,0.25),(0.17,0.48),(0.08,0.50),(0.04,0.38)]),
            ("poly", [(0.93,0.27),(0.84,0.25),(0.83,0.48),(0.92,0.50),(0.96,0.38)]),
        ],
    },
}
# unknown falls back to full_body shapes
_SHAPES[REGION_UNKNOWN] = _SHAPES[REGION_FULL]

# Zoom image config: (filename, crop_frac or None for full image)
_ZOOM_CFG: Dict[str, Tuple[str, Optional[Tuple[float,float,float,float]]]] = {
    REGION_CERVICAL: ("gray_cervical.png",      None),
    REGION_SHOULDER: ("gray_anterior_full.png", (0.10, 0.00, 0.90, 0.35)),
    REGION_ELBOW:    ("gray_forearm.png",        None),
    REGION_LUMBAR:   ("gray_back_lumbar.png",    None),
    REGION_HIP_KNEE: ("gray_thigh_hip.png",      None),
    REGION_FULL:     ("anterior_unlabeled.png",  None),
    REGION_UNKNOWN:  ("anterior_unlabeled.png",  None),
}

# Helpers

def _try_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _paint_muscles(img: Image.Image,
                   shapes: list,
                   colour: Tuple[int,int,int],
                   fill_alpha: int = 90,
                   outline_alpha: int = 200) -> Image.Image:
    """
    Composite per-muscle polygon/ellipse overlays onto *img* (RGB).
    Returns a new RGB image.
    Shapes are normalised 0-1 to the image dimensions.
    """
    img = img.copy().convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    fill_c    = (*colour, fill_alpha)
    outline_c = (*colour, outline_alpha)

    for shape in shapes:
        kind = shape[0]
        if kind == "poly":
            pts_norm = shape[1]
            pts = [(int(x * W), int(y * H)) for x, y in pts_norm]
            if len(pts) >= 3:
                draw.polygon(pts, fill=fill_c, outline=outline_c)
        elif kind == "ellipse":
            cx_n, cy_n, rx_n, ry_n = shape[1], shape[2], shape[3], shape[4]
            cx, cy = int(cx_n * W), int(cy_n * H)
            rx, ry = int(rx_n * W), int(ry_n * H)
            bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
            draw.ellipse(bbox, fill=fill_c, outline=outline_c)

    # Soft-blur the overlay to smooth edges
    try:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    except Exception:
        pass

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def _load_zoom_img(region: str) -> Image.Image:
    fname, crop = _ZOOM_CFG.get(region, _ZOOM_CFG[REGION_UNKNOWN])
    path = _ASSET / fname
    if not path.exists():
        img = Image.new("RGB", (400, 540), (30, 30, 35))
        d = ImageDraw.Draw(img)
        d.text((60, 260), "Image not found", fill=(180, 180, 180))
        return img
    img = Image.open(path).convert("RGB")
    if crop:
        W, H = img.size
        box = (int(crop[0]*W), int(crop[1]*H), int(crop[2]*W), int(crop[3]*H))
        img = img.crop(box)
    return img


def _footer_bar(img: Image.Image, region: str, mas: float,
                colour: Tuple[int,int,int]) -> Image.Image:
    W, H  = img.size
    bar_h = 38
    out   = Image.new("RGB", (W, H + bar_h), (20, 20, 25))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([(0, H), (W, H + bar_h)], fill=(18, 18, 22))
    d.rectangle([(0, H), (W, H + 3)], fill=colour)
    label = region.replace("_", " ").title()
    text  = f"{label}  |  MAS {int(mas)}/100"
    font  = _try_font(13)
    d.text((12, H + 10), text, fill=colour, font=font)
    return out


def _header_bar(img: Image.Image, text: str,
                colour: Tuple[int,int,int]) -> Image.Image:
    W, H  = img.size
    bar_h = 32
    out   = Image.new("RGB", (W, H + bar_h), (20, 20, 25))
    d     = ImageDraw.Draw(out)
    d.rectangle([(0, 0), (W, bar_h)], fill=(18, 18, 22))
    d.rectangle([(0, bar_h - 3), (W, bar_h)], fill=colour)
    font = _try_font(12)
    d.text((12, 8), text, fill=colour, font=font)
    out.paste(img, (0, bar_h))
    return out

# Combined overview image — cached
_COMBINED_CACHE: Optional[Image.Image] = None
_COMBINED_GAP   = 20

def _build_combined() -> Image.Image:
    """Load anterior + posterior unlabeled images, composited side-by-side."""
    global _COMBINED_CACHE
    if _COMBINED_CACHE is not None:
        return _COMBINED_CACHE

    ant_path  = _ASSET / "anterior_unlabeled.png"
    post_path = _ASSET / "posterior_unlabeled.png"

    if not ant_path.exists() or not post_path.exists():
        img = Image.new("RGB", (800, 600), (30, 30, 35))
        d   = ImageDraw.Draw(img)
        d.text((300, 280), "Anatomy image not found", fill=(180, 180, 180))
        return img

    ant  = Image.open(ant_path).convert("RGB")
    post = Image.open(post_path).convert("RGB")

    target_h = max(ant.height, post.height)
    ant  = ant.resize((int(ant.width  * target_h / ant.height),  target_h), Image.LANCZOS)
    post = post.resize((int(post.width * target_h / post.height), target_h), Image.LANCZOS)

    W = ant.width + _COMBINED_GAP + post.width
    combined = Image.new("RGB", (W, target_h), (20, 20, 25))
    combined.paste(ant,  (0, 0))
    combined.paste(post, (ant.width + _COMBINED_GAP, 0))

    _COMBINED_CACHE = combined
    return combined

# Public API

# Pose-frame muscle overlay  (draws on the ACTUAL video frame using real
# MediaPipe landmark positions — much more accurate than generic anatomy art)

def _pose_overlays(
    draw: ImageDraw.ImageDraw,
    landmarks: dict,
    W: int,
    H: int,
    region: str,
    colour: tuple,
    fill_alpha: int = 160,
    outline_alpha: int = 240,
) -> None:
    """
    Draw per-muscle band/ellipse shapes onto *draw* (RGBA surface) using
    actual MediaPipe landmark positions scaled to the image (W × H).
    """
    import math

    fill_c    = (*colour, fill_alpha)
    outline_c = (*colour, outline_alpha)

    def px(name):
        """Return pixel coords for landmark regardless of visibility score.
        MediaPipe always provides valid x,y for all 33 landmarks when a person
        is detected — visibility only indicates confidence, not coordinate validity."""
        d = landmarks.get(name)
        if d is None:
            return None
        return (d["x"] * W, d["y"] * H)

    def d2(a, b):
        if not a or not b:
            return 40.0
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def lerp(a, b, t):
        if not a or not b:
            return None
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    def band(a, b, w):
        """Draw an elongated band (parallelogram) from a to b with half-width w."""
        if not a or not b or w < 1:
            return
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(dx, dy) + 1e-9
        ox, oy = -dy / ln * w, dx / ln * w
        pts = [
            (a[0] + ox, a[1] + oy),
            (b[0] + ox, b[1] + oy),
            (b[0] - ox, b[1] - oy),
            (a[0] - ox, a[1] - oy),
        ]
        draw.polygon(pts, fill=fill_c, outline=outline_c)

    def oval(cx, cy, rx, ry):
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                     fill=fill_c, outline=outline_c)

    # Key landmarks
    ls = px("left_shoulder");  rs = px("right_shoulder")
    le = px("left_elbow");     re = px("right_elbow")
    lw = px("left_wrist");     rw = px("right_wrist")
    lh = px("left_hip");       rh = px("right_hip")
    lk = px("left_knee");      rk = px("right_knee")
    la = px("left_ankle");     ra = px("right_ankle")
    lear = px("left_ear");     rear = px("right_ear")

    # Shoulder width = primary scale reference
    sw = d2(ls, rs) if (ls and rs) else 80.0

    # ── Cervical ──────────────────────────────────────────────────────────
    if region in (REGION_CERVICAL, REGION_FULL, REGION_UNKNOWN):
        if lear and ls:
            band(lear, lerp(lear, ls, 0.85), sw * 0.14)
        if rear and rs:
            band(rear, lerp(rear, rs, 0.85), sw * 0.14)
        neck = lerp(ls, rs, 0.5) if (ls and rs) else None
        if neck and ls:
            band(lerp(ls, rs, 0.28), ls, sw * 0.16)
        if neck and rs:
            band(lerp(ls, rs, 0.72), rs, sw * 0.16)

    # ── Shoulder ──────────────────────────────────────────────────────────
    if region in (REGION_SHOULDER, REGION_FULL, REGION_UNKNOWN):
        ua_l = d2(ls, le);  ua_r = d2(rs, re)
        # Deltoid wrap around upper arm
        if ls and le:
            band(ls, lerp(ls, le, 0.65), ua_l * 0.38)
        if rs and re:
            band(rs, lerp(rs, re, 0.65), ua_r * 0.38)
        # Pec major: shoulder → chest center → torso
        chest = lerp(ls, rs, 0.5) if (ls and rs) else None
        tl = lerp(ls, lh, 0.38) if (ls and lh) else None
        tr = lerp(rs, rh, 0.38) if (rs and rh) else None
        if ls and chest and tl:
            draw.polygon([ls, chest, tl], fill=fill_c, outline=outline_c)
        if rs and chest and tr:
            draw.polygon([rs, chest, tr], fill=fill_c, outline=outline_c)
        # Trapezius mid
        if ls and rs and lh and rh:
            sh_mid = lerp(ls, rs, 0.5)
            hi_mid = lerp(lh, rh, 0.5)
            band(lerp(ls, rs, 0.3), sh_mid, sw * 0.14)
            band(lerp(ls, rs, 0.7), sh_mid, sw * 0.14)

    # ── Elbow / Wrist ──────────────────────────────────────────────────────
    if region in (REGION_ELBOW, REGION_FULL, REGION_UNKNOWN):
        ua_l = d2(ls, le);  ua_r = d2(rs, re)
        fa_l = d2(le, lw);  fa_r = d2(re, rw)
        # Biceps: upper arm
        if ls and le:
            band(ls, le, ua_l * 0.26)
        if rs and re:
            band(rs, re, ua_r * 0.26)
        # Forearm flexors
        if le and lw:
            band(le, lw, fa_l * 0.26)
        if re and rw:
            band(re, rw, fa_r * 0.26)

    # ── Lumbar / Back ──────────────────────────────────────────────────────
    if region in (REGION_LUMBAR, REGION_FULL, REGION_UNKNOWN):
        if ls and rs and lh and rh:
            sh_mid = lerp(ls, rs, 0.5)
            hi_mid = lerp(lh, rh, 0.5)
            dx = hi_mid[0] - sh_mid[0]
            dy = hi_mid[1] - sh_mid[1]
            ln = math.hypot(dx, dy) + 1e-9
            ox, oy = -dy / ln * sw * 0.14, dx / ln * sw * 0.14
            # Bilateral erector spinae — wider and more opaque
            band(
                (sh_mid[0] + ox, sh_mid[1] + oy),
                (hi_mid[0] + ox, hi_mid[1] + oy),
                sw * 0.13,
            )
            band(
                (sh_mid[0] - ox, sh_mid[1] - oy),
                (hi_mid[0] - ox, hi_mid[1] - oy),
                sw * 0.13,
            )
        # Latissimus dorsi
        if ls and lh:
            band(lerp(ls, lh, 0.18), lh, sw * 0.30)
        if rs and rh:
            band(lerp(rs, rh, 0.18), rh, sw * 0.30)

    # ── Hip / Knee ──────────────────────────────────────────────────────────
    if region in (REGION_HIP_KNEE, REGION_FULL, REGION_UNKNOWN):
        tl = d2(lh, lk);  tr = d2(rh, rk)
        # Quadriceps: thigh band
        if lh and lk:
            band(lh, lk, tl * 0.32)
        if rh and rk:
            band(rh, rk, tr * 0.32)
        # Gluteal ellipse around hip
        hiw = sw * 0.40
        if lh:
            oval(lh[0], lh[1], hiw, hiw * 0.75)
        if rh:
            oval(rh[0], rh[1], hiw, hiw * 0.75)
        # Calf / lower leg
        sl = d2(lk, la);  sr = d2(rk, ra)
        if lk and la:
            band(lk, la, sl * 0.22)
        if rk and ra:
            band(rk, ra, sr * 0.22)


def draw_pose_muscle_overlay(
    frame_bgr: "np.ndarray",
    landmarks: dict,
    region: str,
    mas_score: float,
) -> Image.Image:
    """
    Draw per-muscle highlights on the actual video frame using MediaPipe
    landmark positions for accurate placement.

    Parameters
    ----------
    frame_bgr  : BGR numpy array (from OpenCV / PoseFrame.annotated_image)
    landmarks  : dict of {landmark_name: {x, y, z, visibility}}  (normalised 0-1)
    region     : body region string (REGION_* constant)
    mas_score  : 0-100 Movement Abnormality Score

    Returns
    -------
    PIL Image (RGB) — the frame with coloured muscle overlays and a header bar.
    """
    import cv2 as _cv2
    import numpy as _np

    colour    = _mas_colour(mas_score)
    frame_rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
    W, H      = frame_rgb.shape[1], frame_rgb.shape[0]

    base    = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    _pose_overlays(draw, landmarks, W, H, region, colour)

    try:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1))
    except Exception:
        pass

    result = Image.alpha_composite(base, overlay).convert("RGB")
    label  = region.replace("_", " ").title()
    result = _header_bar(
        result,
        f"Live Pose  |  {label}  |  MAS {int(mas_score)}/100",
        colour,
    )
    return result


def draw_pose_zoom_view(
    frame_bgr: "np.ndarray",
    landmarks: dict,
    region: str,
    mas_score: float,
) -> Image.Image:
    """
    Crop the pose frame to the affected body region and apply muscle overlays.

    Returns
    -------
    PIL Image (RGB) — cropped + highlighted + footer bar.
    """
    import cv2 as _cv2

    colour    = _mas_colour(mas_score)
    frame_rgb = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)
    W, H      = frame_rgb.shape[1], frame_rgb.shape[0]

    # First apply overlays on the full frame
    base    = Image.fromarray(frame_rgb).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    _pose_overlays(draw, landmarks, W, H, region, colour)
    try:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1))
    except Exception:
        pass
    full_rgb = Image.alpha_composite(base, overlay).convert("RGB")

    # Determine crop box from landmark positions
    _region_keys = {
        REGION_CERVICAL: ["nose", "left_ear", "right_ear",
                          "left_shoulder", "right_shoulder"],
        REGION_SHOULDER: ["left_shoulder", "right_shoulder",
                          "left_elbow", "right_elbow"],
        REGION_ELBOW:    ["left_shoulder", "right_shoulder",
                          "left_elbow", "right_elbow",
                          "left_wrist", "right_wrist"],
        REGION_LUMBAR:   ["left_shoulder", "right_shoulder",
                          "left_hip", "right_hip"],
        REGION_HIP_KNEE: ["left_hip", "right_hip",
                          "left_knee", "right_knee",
                          "left_ankle", "right_ankle"],
    }

    crop_img = full_rgb
    if region not in (REGION_FULL, REGION_UNKNOWN):
        keys = _region_keys.get(region, [])
        pts  = []
        for k in keys:
            d = landmarks.get(k)
            if d:   # accept all coordinates regardless of visibility
                pts.append((d["x"] * W, d["y"] * H))
        if pts:
            xs = [p[0] for p in pts];  ys = [p[1] for p in pts]
            span = max(max(xs) - min(xs), max(ys) - min(ys), 60)
            pad  = span * 0.45 + 30
            x0   = max(0, int(min(xs) - pad))
            y0   = max(0, int(min(ys) - pad))
            x1   = min(W, int(max(xs) + pad))
            y1   = min(H, int(max(ys) + pad))
            if x1 > x0 and y1 > y0:
                crop_img = full_rgb.crop((x0, y0, x1, y1))

    MAX_H = 580
    if crop_img.height > MAX_H:
        scale    = MAX_H / crop_img.height
        crop_img = crop_img.resize(
            (int(crop_img.width * scale), MAX_H), Image.LANCZOS
        )

    return _footer_bar(crop_img, region, mas_score, colour)


# Synthetic anatomy diagram (Pillow-drawn — no external image dependencies)

# Normalized (0–1) body silhouette shapes for a 420×560 canvas.
# Each shape: ('ellipse', cx, cy, rx, ry) or ('poly', [(x,y),...])
#   x increases right, y increases downward.
_BODY_SILHOUETTE = [
    # Head
    ("ellipse", 0.500, 0.075, 0.100, 0.070),
    # Neck
    ("poly", [(0.455, 0.140), (0.545, 0.140), (0.545, 0.195), (0.455, 0.195)]),
    # Torso (trapezoid: wide shoulders, narrower hips)
    ("poly", [(0.225, 0.195), (0.775, 0.195),
              (0.760, 0.565), (0.240, 0.565)]),
    # Left upper arm
    ("poly", [(0.145, 0.210), (0.235, 0.195),
              (0.250, 0.420), (0.160, 0.440)]),
    # Right upper arm
    ("poly", [(0.855, 0.210), (0.765, 0.195),
              (0.750, 0.420), (0.840, 0.440)]),
    # Left forearm
    ("poly", [(0.148, 0.440), (0.248, 0.420),
              (0.245, 0.610), (0.155, 0.625)]),
    # Right forearm
    ("poly", [(0.852, 0.440), (0.752, 0.420),
              (0.755, 0.610), (0.845, 0.625)]),
    # Left thigh
    ("poly", [(0.245, 0.565), (0.435, 0.565),
              (0.425, 0.790), (0.255, 0.800)]),
    # Right thigh
    ("poly", [(0.755, 0.565), (0.565, 0.565),
              (0.575, 0.790), (0.745, 0.800)]),
    # Left calf
    ("poly", [(0.260, 0.800), (0.420, 0.790),
              (0.415, 0.960), (0.268, 0.966)]),
    # Right calf
    ("poly", [(0.740, 0.800), (0.580, 0.790),
              (0.585, 0.960), (0.732, 0.966)]),
]

# Per-region muscle shapes + label (normalised to same 420×560 canvas)
# Format: ('ellipse'|'poly', coords..., label_str)
_DIAGRAM_MUSCLES: Dict[str, list] = {

    REGION_CERVICAL: [
        ("poly", [(0.400,0.145),(0.600,0.145),(0.600,0.190),(0.400,0.190)], "Sternocleidomastoid"),
        ("poly", [(0.225,0.195),(0.445,0.155),(0.435,0.210),(0.260,0.230)], "Trapezius (L)"),
        ("poly", [(0.775,0.195),(0.555,0.155),(0.565,0.210),(0.740,0.230)], "Trapezius (R)"),
        ("ellipse", 0.420,0.165,0.030,0.022,                              "Scalene (L)"),
        ("ellipse", 0.580,0.165,0.030,0.022,                              "Scalene (R)"),
        ("ellipse", 0.440,0.145,0.025,0.018,                              "Levator scapulae (L)"),
        ("ellipse", 0.560,0.145,0.025,0.018,                              "Levator scapulae (R)"),
    ],

    REGION_SHOULDER: [
        # Deltoid — wrap over shoulder joint
        ("ellipse", 0.195,0.235,0.065,0.060,   "Deltoid (L)"),
        ("ellipse", 0.805,0.235,0.065,0.060,   "Deltoid (R)"),
        # Trapezius mid
        ("poly", [(0.225,0.195),(0.430,0.185),(0.420,0.280),(0.240,0.295)], "Trapezius (L)"),
        ("poly", [(0.775,0.195),(0.570,0.185),(0.580,0.280),(0.760,0.295)], "Trapezius (R)"),
        # Pec major
        ("poly", [(0.260,0.200),(0.450,0.198),(0.445,0.330),(0.245,0.350)], "Pec Major (L)"),
        ("poly", [(0.740,0.200),(0.550,0.198),(0.555,0.330),(0.755,0.350)], "Pec Major (R)"),
        # Rotator cuff hint (posterior at shoulder)
        ("ellipse", 0.215,0.270,0.045,0.035,   "Rotator Cuff (L)"),
        ("ellipse", 0.785,0.270,0.045,0.035,   "Rotator Cuff (R)"),
    ],

    REGION_ELBOW: [
        # Biceps
        ("poly", [(0.158,0.230),(0.232,0.210),(0.242,0.390),(0.168,0.405)], "Biceps (L)"),
        ("poly", [(0.842,0.230),(0.768,0.210),(0.758,0.390),(0.832,0.405)], "Biceps (R)"),
        # Triceps (posterior — shown on same side for diagram)
        ("poly", [(0.145,0.225),(0.162,0.230),(0.170,0.415),(0.150,0.420)], "Triceps (L)"),
        ("poly", [(0.855,0.225),(0.838,0.230),(0.830,0.415),(0.850,0.420)], "Triceps (R)"),
        # Brachioradialis
        ("poly", [(0.150,0.440),(0.220,0.425),(0.230,0.540),(0.158,0.555)], "Brachioradialis (L)"),
        ("poly", [(0.850,0.440),(0.780,0.425),(0.770,0.540),(0.842,0.555)], "Brachioradialis (R)"),
        # Forearm flexors
        ("poly", [(0.158,0.555),(0.232,0.540),(0.238,0.620),(0.162,0.628)], "Forearm Flexors (L)"),
        ("poly", [(0.842,0.555),(0.768,0.540),(0.762,0.620),(0.838,0.628)], "Forearm Flexors (R)"),
    ],

    REGION_LUMBAR: [
        # Erector spinae — bilateral bands along spine
        ("poly", [(0.420,0.210),(0.470,0.210),(0.472,0.555),(0.418,0.555)], "Erector Spinae (L)"),
        ("poly", [(0.580,0.210),(0.530,0.210),(0.528,0.555),(0.582,0.555)], "Erector Spinae (R)"),
        # Latissimus dorsi — large fans from mid-back to hips
        ("poly", [(0.240,0.240),(0.420,0.215),(0.415,0.490),(0.275,0.555),(0.238,0.480)], "Latissimus Dorsi (L)"),
        ("poly", [(0.760,0.240),(0.580,0.215),(0.585,0.490),(0.725,0.555),(0.762,0.480)], "Latissimus Dorsi (R)"),
        # Multifidus — small deep muscles at lumbar
        ("ellipse", 0.445,0.440,0.030,0.065,   "Multifidus (L)"),
        ("ellipse", 0.555,0.440,0.030,0.065,   "Multifidus (R)"),
        # Quadratus lumborum — lateral low-back
        ("poly", [(0.250,0.430),(0.380,0.430),(0.375,0.555),(0.245,0.553)], "Quad. Lumborum (L)"),
        ("poly", [(0.750,0.430),(0.620,0.430),(0.625,0.555),(0.755,0.553)], "Quad. Lumborum (R)"),
    ],

    REGION_HIP_KNEE: [
        # Gluteus maximus
        ("poly", [(0.252,0.560),(0.430,0.556),(0.425,0.660),(0.258,0.668)], "Glut. Max (L)"),
        ("poly", [(0.748,0.560),(0.570,0.556),(0.575,0.660),(0.742,0.668)], "Glut. Max (R)"),
        # Gluteus medius
        ("poly", [(0.248,0.500),(0.390,0.490),(0.385,0.570),(0.250,0.574)], "Glut. Med (L)"),
        ("poly", [(0.752,0.500),(0.610,0.490),(0.615,0.570),(0.750,0.574)], "Glut. Med (R)"),
        # Quadriceps (anterior thigh)
        ("poly", [(0.268,0.572),(0.418,0.568),(0.412,0.780),(0.268,0.788)], "Quadriceps (L)"),
        ("poly", [(0.732,0.572),(0.582,0.568),(0.588,0.780),(0.732,0.788)], "Quadriceps (R)"),
        # Hamstrings (posterior — slightly offset for visibility)
        ("poly", [(0.270,0.580),(0.290,0.580),(0.282,0.788),(0.262,0.790)], "Hamstrings (L)"),
        ("poly", [(0.730,0.580),(0.710,0.580),(0.718,0.788),(0.738,0.790)], "Hamstrings (R)"),
        # Calf / Gastrocnemius
        ("poly", [(0.268,0.795),(0.408,0.788),(0.402,0.920),(0.272,0.926)], "Gastrocnemius (L)"),
        ("poly", [(0.732,0.795),(0.592,0.788),(0.598,0.920),(0.728,0.926)], "Gastrocnemius (R)"),
    ],

    REGION_FULL: [
        # Body-wide highlight — lighter fill just for overview
        ("poly", [(0.225,0.195),(0.775,0.195),(0.760,0.565),(0.240,0.565)], "Torso"),
        ("poly", [(0.245,0.565),(0.435,0.565),(0.425,0.900),(0.255,0.906)], "Left Leg"),
        ("poly", [(0.755,0.565),(0.565,0.565),(0.575,0.900),(0.745,0.906)], "Right Leg"),
        ("poly", [(0.145,0.210),(0.235,0.195),(0.248,0.610),(0.155,0.622)], "Left Arm"),
        ("poly", [(0.855,0.210),(0.765,0.195),(0.752,0.610),(0.845,0.622)], "Right Arm"),
    ],
}
_DIAGRAM_MUSCLES[REGION_UNKNOWN] = _DIAGRAM_MUSCLES[REGION_FULL]


def _S(val: float, dim: int) -> int:
    """Scale normalised 0-1 value to pixel dimension."""
    return int(val * dim)


def draw_anatomy_diagram(region: str, mas_score: float) -> Image.Image:
    """
    Render a stylised Pillow-drawn muscle anatomical diagram for the given
    body region.  Does NOT depend on external images or native Cairo DLLs.

    Returns
    -------
    PIL Image (RGB), 420×580 with header bar.
    """
    import math as _math

    W, H = 420, 560
    colour = _mas_colour(mas_score)

    # ── Background ────────────────────────────────────────────────────────
    img = Image.new("RGBA", (W, H), (10, 15, 35, 255))   # dark navy

    # subtle radial vignette: slightly lighter in centre
    vignette = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    for r in range(min(W, H) // 2, 0, -8):
        alpha = max(0, 40 - int(40 * r / (min(W, H) // 2)))
        vd.ellipse(
            [W // 2 - r, H // 2 - r, W // 2 + r, H // 2 + r],
            fill=(255, 255, 255, alpha),
        )
    img = Image.alpha_composite(img, vignette)

    # ── Body silhouette ───────────────────────────────────────────────────
    body_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body_layer)
    BODY_FILL    = (42, 50, 75, 220)
    BODY_OUTLINE = (130, 150, 200, 200)

    for shape in _BODY_SILHOUETTE:
        if shape[0] == "ellipse":
            _, cx, cy, rx, ry = shape
            bd.ellipse(
                [_S(cx - rx, W), _S(cy - ry, H), _S(cx + rx, W), _S(cy + ry, H)],
                fill=BODY_FILL, outline=BODY_OUTLINE, width=2,
            )
        elif shape[0] == "poly":
            pts = [(_S(x, W), _S(y, H)) for x, y in shape[1]]
            bd.polygon(pts, fill=BODY_FILL, outline=BODY_OUTLINE)

    img = Image.alpha_composite(img, body_layer)

    # ── Grid lines (subtle) ───────────────────────────────────────────────
    grid_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid_layer)
    for x in range(0, W, 30):
        gd.line([(x, 0), (x, H)], fill=(30, 45, 80, 60), width=1)
    for y in range(0, H, 30):
        gd.line([(0, y), (W, y)], fill=(30, 45, 80, 60), width=1)
    img = Image.alpha_composite(img, grid_layer)

    # ── Glow pass (soft halo in muscle colour) ────────────────────────────
    muscles = _DIAGRAM_MUSCLES.get(region, _DIAGRAM_MUSCLES[REGION_FULL])
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gl = ImageDraw.Draw(glow_layer)
    glow_fill = (*colour, 35)
    for shape in muscles:
        if shape[0] == "ellipse":
            _, cx, cy, rx, ry, _lbl = shape
            for expand in [20, 12, 6]:
                gl.ellipse(
                    [_S(cx - rx, W) - expand, _S(cy - ry, H) - expand,
                     _S(cx + rx, W) + expand, _S(cy + ry, H) + expand],
                    fill=glow_fill,
                )
        elif shape[0] == "poly":
            pts = [(_S(x, W), _S(y, H)) for x, y in shape[1]]
            cx_ = sum(p[0] for p in pts) // len(pts)
            cy_ = sum(p[1] for p in pts) // len(pts)
            expanded = [
                (cx_ + int((px_ - cx_) * 1.15),
                 cy_ + int((py_ - cy_) * 1.15))
                for px_, py_ in pts
            ]
            gl.polygon(expanded, fill=glow_fill)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=14))
    img = Image.alpha_composite(img, glow_layer)

    # ── Muscle fills ──────────────────────────────────────────────────────
    muscle_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ml = ImageDraw.Draw(muscle_layer)
    muscle_fill    = (*colour, 170)
    muscle_outline = (*colour, 255)

    for shape in muscles:
        if shape[0] == "ellipse":
            _, cx, cy, rx, ry, _lbl = shape
            ml.ellipse(
                [_S(cx - rx, W), _S(cy - ry, H), _S(cx + rx, W), _S(cy + ry, H)],
                fill=muscle_fill, outline=muscle_outline, width=2,
            )
        elif shape[0] == "poly":
            pts = [(_S(x, W), _S(y, H)) for x, y in shape[1]]
            ml.polygon(pts, fill=muscle_fill, outline=muscle_outline)

    img = Image.alpha_composite(img, muscle_layer)

    # ── Labels ────────────────────────────────────────────────────────────
    label_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(label_layer)

    try:
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font_sm = ImageFont.load_default()

    seen_labels: set = set()
    for shape in muscles:
        lbl = shape[-1]
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        if shape[0] == "ellipse":
            _, cx, cy, rx, ry, _ = shape
            tx, ty = _S(cx, W), _S(cy, H)
        else:
            pts = [(_S(x, W), _S(y, H)) for x, y in shape[1]]
            tx = sum(p[0] for p in pts) // len(pts)
            ty = sum(p[1] for p in pts) // len(pts)

        # background pill behind text
        try:
            bbox = ld.textbbox((tx, ty), lbl, font=font_sm)
            pad = 3
            ld.rounded_rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                radius=3,
                fill=(10, 15, 35, 200),
            )
        except Exception:
            pass
        ld.text((tx, ty), lbl, fill=(255, 255, 255, 230), font=font_sm, anchor="mm")

    img = Image.alpha_composite(img, label_layer)

    # ── Legend dot & title ────────────────────────────────────────────────
    legend_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    legendd = ImageDraw.Draw(legend_layer)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 13)
    except Exception:
        font_title = font_sm

    region_label = region.replace("_", " ").title()
    severity = (
        "Normal"   if mas_score < 25 else
        "Mild"     if mas_score < 50 else
        "Moderate" if mas_score < 75 else "Severe"
    )
    title_text = f"{region_label}  ·  {severity} ({int(mas_score)}/100)"

    # title bar background
    legendd.rectangle([0, H - 36, W, H], fill=(10, 15, 35, 230))
    legendd.text((W // 2, H - 18), title_text,
                 fill=colour + (255,), font=font_title, anchor="mm")

    img = Image.alpha_composite(img, legend_layer)

    # convert to RGB
    final = img.convert("RGB")
    return final


# Original anatomy-image based functions (kept as fallback)

def draw_region_highlight(region: str, mas_score: float) -> Image.Image:
    """
    Draw per-muscle polygon overlays on the combined anterior+posterior
    overview image.  Returns a PIL Image.
    """
    colour    = _mas_colour(mas_score)
    combined  = _build_combined().copy()

    ant_path  = _ASSET / "anterior_unlabeled.png"
    post_path = _ASSET / "posterior_unlabeled.png"

    shapes_cfg = _SHAPES.get(region, _SHAPES[REGION_FULL])

    # ── Paint anterior half ───────────────────────────────────────────────
    if ant_path.exists() and shapes_cfg.get("ant"):
        ant = Image.open(ant_path).convert("RGB")
        H_target = combined.height
        ant = ant.resize((int(ant.width * H_target / ant.height), H_target), Image.LANCZOS)
        ant = _paint_muscles(ant, shapes_cfg["ant"], colour)
        combined.paste(ant, (0, 0))

    # ── Paint posterior half ──────────────────────────────────────────────
    ant_w = Image.open(ant_path).width if ant_path.exists() else 659
    ant_w_scaled = int(ant_w * combined.height / (Image.open(ant_path).height if ant_path.exists() else 751))
    post_x = ant_w_scaled + _COMBINED_GAP

    if post_path.exists() and shapes_cfg.get("post"):
        post = Image.open(post_path).convert("RGB")
        H_target = combined.height
        post = post.resize((int(post.width * H_target / post.height), H_target), Image.LANCZOS)
        post = _paint_muscles(post, shapes_cfg["post"], colour)
        combined.paste(post, (post_x, 0))

    label   = region.replace("_", " ").title()
    combined = _header_bar(combined, f"Affected Region: {label}  |  MAS {int(mas_score)}/100", colour)
    return combined


# Regions whose zoom image is a posterior/back view — use 'post' shapes, not 'ant'
_POSTERIOR_REGIONS = {REGION_LUMBAR, REGION_CERVICAL}


def draw_zoom_view(region: str, mas_score: float) -> Image.Image:
    """
    Zoom view — real Gray's anatomy image with per-muscle polygon
    highlights and a labelled muscle legend.
    """
    colour = _mas_colour(mas_score)
    img    = _load_zoom_img(region).convert("RGB")

    # Pick correct shape set (posterior images need 'post' landmarks)
    shapes_cfg = _SHAPES.get(region, _SHAPES[REGION_FULL])
    if region in _POSTERIOR_REGIONS:
        zoom_shapes = shapes_cfg.get("post") or shapes_cfg.get("ant") or []
    else:
        zoom_shapes = shapes_cfg.get("ant") or shapes_cfg.get("post") or []

    if zoom_shapes:
        img = _paint_muscles(img, zoom_shapes, colour,
                             fill_alpha=155, outline_alpha=255)

    # ── Resize to max 540 tall ────────────────────────────────────────────
    MAX_H = 540
    if img.height > MAX_H:
        scale = MAX_H / img.height
        img   = img.resize((int(img.width * scale), MAX_H), Image.LANCZOS)

    # ── Muscle legend panel on the right ─────────────────────────────────
    muscles      = REGION_MUSCLES.get(region, [])
    legend_w     = 190
    font_lbl     = _try_font(12)
    font_title   = _try_font(13)
    line_h       = 22
    pad          = 10
    legend_h     = img.height
    legend_img   = Image.new("RGB", (legend_w, legend_h), (14, 18, 32))
    ld           = ImageDraw.Draw(legend_img)

    # title bar
    title_text = region.replace("_", " ").title() + " Muscles"
    ld.rectangle([(0, 0), (legend_w, 30)], fill=colour)
    ld.text((legend_w // 2, 15), title_text,
            fill=(255, 255, 255), font=font_title, anchor="mm")

    severity = (
        "Normal"   if mas_score < 25 else
        "Mild"     if mas_score < 50 else
        "Moderate" if mas_score < 75 else "Severe"
    )
    ld.text((legend_w // 2, 43), f"MAS {int(mas_score)}/100  \u00b7  {severity}",
            fill=colour, font=font_lbl, anchor="mm")

    y = 62
    for i, muscle in enumerate(muscles):
        # alternating row tint
        row_bg = (20, 26, 46) if i % 2 == 0 else (14, 18, 32)
        ld.rectangle([(0, y - 2), (legend_w, y + line_h - 2)], fill=row_bg)
        # colour dot
        ld.ellipse([(pad, y + 4), (pad + 9, y + 13)], fill=colour)
        # muscle name — wrap long names
        name = muscle
        ld.text((pad + 14, y + 3), name, fill=(220, 230, 255), font=font_lbl)
        y += line_h
        if y + line_h > legend_h - 10:
            break  # avoid overflow

    # separator line
    ld.line([(0, 0), (0, legend_h)], fill=colour, width=3)

    # ── Compose image + legend side-by-side ──────────────────────────────
    combined = Image.new("RGB", (img.width + legend_w, img.height), (14, 18, 32))
    combined.paste(img,        (0, 0))
    combined.paste(legend_img, (img.width, 0))

    combined = _footer_bar(combined, region, mas_score, colour)
    return combined
