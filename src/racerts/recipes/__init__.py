"""Recipes: pipelines put together from the stages for one purpose."""

from .saddles import saddles
from .staged import Level, staged

__all__ = ["Level", "saddles", "staged"]
