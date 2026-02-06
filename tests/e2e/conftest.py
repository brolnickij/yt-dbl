"""E2E conftest — force sequential execution for GPU-heavy tests."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Put all E2E tests into a single xdist group so they run sequentially.

    With ``-n auto --dist loadgroup`` the xdist scheduler keeps all tests
    that share the same ``xdist_group`` marker on a single worker, which
    prevents multiple GPU models from loading in parallel.
    """
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.xdist_group("e2e_sequential"))
