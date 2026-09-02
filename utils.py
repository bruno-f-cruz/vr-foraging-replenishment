"""Colors and site-band styling shared by the visualization builders.

Ported from AllenNeuralDynamics/Aind.Behavior.VrForaging.Dashboard's
``viz/theme.py`` (patch_index_colormap / get_color_from_site).
"""

PATCH_COLORMAP = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
]

_INTERPATCH_COLOR = "#A9A9A9"
_INTERSITE_COLOR = "#4C4C4C"
_UNKNOWN_SITE_COLOR = "#CCCCCC"

CHOICE_COLOR = "#d62728"
REWARD_COLOR = "#1f77b4"
FORCE_REWARD_COLOR = "#00bfff"  # cyan — distinguishes forced rewards from normal rewards
LICK_COLOR = "#2ca02c"
VELOCITY_COLOR = "#222222"
ANNOTATION_COLOR = "#9467bd"  # purple — software-event annotations
BLOCK_SEPARATOR_COLOR = "#444444"  # vertical separators between blocks
SNIFF_COLOR = "#ff7f0e"  # orange — sniffing frequency overlay


def patch_color(index: int) -> str:
    """Stable color for a patch by its (type) index."""
    return PATCH_COLORMAP[int(index) % len(PATCH_COLORMAP)]


def get_color_from_site(site_label: str, patch_color_index: int) -> str:
    """Color a track segment: reward sites take a per-patch color, others grey.

    ``patch_color_index`` should be derived from the patch *label* so that every
    reward site belonging to the same patch shares a single color.
    """
    label = str(site_label or "").lower()
    if "reward" in label:
        return patch_color(patch_color_index)
    if "interpatch" in label:
        return _INTERPATCH_COLOR
    if "intersite" in label:
        return _INTERSITE_COLOR
    return _UNKNOWN_SITE_COLOR
