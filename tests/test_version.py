# ------------------------------------------------------------------------
# RF-DETR+
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
# Licensed under the Platform Model License 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Test basic package functionality."""


def test_version() -> None:
    """Test that the package version is accessible."""
    from rfdetr_plus import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
