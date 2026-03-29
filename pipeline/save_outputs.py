"""Save figures and animations to the images/ directory."""

import os

from utils.logger import get_logger

_log = get_logger("tora.save")

_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")


def ensure_image_dir():
    """Create the images directory if it doesn't exist."""
    os.makedirs(_IMG_DIR, exist_ok=True)
    return _IMG_DIR


def save_figure(fig, name, dpi=150):
    """Save a matplotlib figure to images/<name>.png."""
    img_dir = ensure_image_dir()
    path = os.path.join(img_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _log.info("Saved: %s", path)
    return path


def save_animation(anim, name, fps=50):
    """Save a matplotlib animation as GIF to images/<name>.gif."""
    img_dir = ensure_image_dir()
    path = os.path.join(img_dir, f"{name}.gif")
    anim.save(path, writer="pillow", fps=fps)
    _log.info("Saved animation: %s", path)
    return path
