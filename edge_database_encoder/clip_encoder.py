"""CLIP-based feature extraction for edge-side processing."""

from __future__ import annotations

from typing import Sequence, Union

import clip  # type: ignore
import numpy as np
import torch
from PIL import Image


class CLIPFeatureExtractor:
    """Wrapper around the CLIP vision encoder."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(
        self,
        images: Union[Image.Image, Sequence[Image.Image]],
    ) -> np.ndarray:
        """Return L2-normalized CLIP features."""

        if isinstance(images, Image.Image):
            images = [images]

        processed = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.model.encode_image(processed)
        features = features / features.norm(dim=-1, keepdim=True)
        features_np = features.cpu().numpy()

        if len(features_np) == 1:
            return features_np[0]
        return features_np
