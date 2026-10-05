"""Recipes: pipelines put together from the stages for one purpose."""

from .saddles import saddles
from .staged import staged

__all__ = ["saddles", "staged"]
