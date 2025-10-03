"""Tests for the main module."""

import pytest
from project_tailwind.main import main


def test_main_runs():
    """Test that main function runs without error."""
    # This is a basic smoke test
    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"main() raised {e} unexpectedly!")