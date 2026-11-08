"""Event-only motion compensation experiments for DSEC."""

from .algorithm import (
    contrast_objective,
    estimate_translation,
    refocus_event_slices,
    slice_local_events,
)
from .data import (
    EventWindow,
    decode_flow_png,
    load_event_window,
    load_rectification_map,
    rectify_xy,
)
from .blockwise import (
    BlockwiseTranslation,
    estimate_blockwise_translation,
    phase_correlation_shift,
    slice_event_images,
    warp_event_volume,
    warp_events_with_field,
)
from .affine import (
    AffineField,
    estimate_affine_from_event_groups,
    estimate_local_affine_field,
    fit_local_affine_from_blocks,
)

__all__ = [
    "EventWindow",
    "BlockwiseTranslation",
    "AffineField",
    "contrast_objective",
    "decode_flow_png",
    "estimate_blockwise_translation",
    "estimate_affine_from_event_groups",
    "estimate_local_affine_field",
    "estimate_translation",
    "load_event_window",
    "load_rectification_map",
    "fit_local_affine_from_blocks",
    "phase_correlation_shift",
    "rectify_xy",
    "refocus_event_slices",
    "slice_event_images",
    "warp_event_volume",
    "warp_events_with_field",
    "slice_local_events",
]
