"""Consistent axis styling for all TORA plots."""


def apply_style(ax, grid=True, zero_line=True):
    """Apply standard style to a matplotlib Axes."""
    if grid:
        ax.grid(True, alpha=0.3, linewidth=0.5)
    if zero_line:
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.tick_params(labelsize=8)
