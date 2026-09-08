"""
DETRPose-style pose transformer components vendored into this repo.

We keep this in a separate package so we can:
- integrate a DETRPose/GroupPose-style decoder without breaking existing DFINETransformer
- switch via YAML config / training script flags
"""

from .transformer import DETRPoseTransformer  # noqa: F401

