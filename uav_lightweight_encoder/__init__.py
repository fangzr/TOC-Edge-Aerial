"""Lightweight VIB-based encoder for UAV-side compression."""

from .model import VariationalInformationBottleneck, MultiViewVIB

__all__ = ["VariationalInformationBottleneck", "MultiViewVIB"]
