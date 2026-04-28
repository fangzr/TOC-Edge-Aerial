"""Edge-side localization modules for multi-view UAV features."""

from .attention_fusion import EnhancedMultiViewFusion, MultiViewAttentionFusion, ViewWeightPredictor

__all__ = [
    "EnhancedMultiViewFusion",
    "MultiViewAttentionFusion",
    "ViewWeightPredictor",
]
