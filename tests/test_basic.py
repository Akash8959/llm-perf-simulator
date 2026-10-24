"""Basic smoke tests for llmsim package."""

import llmsim


def test_version() -> None:
    """Verify package version is set."""
    assert llmsim.__version__ == "0.1.0"
