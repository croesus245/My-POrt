"""
Model serving.

Production inference is where the rubber meets the road.
"""

from .app import app, create_app

__all__ = ["app", "create_app"]
