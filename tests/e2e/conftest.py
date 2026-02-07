"""E2E conftest — force sequential execution for GPU-heavy tests.

E2E tests are always invoked with ``-n0`` (see ``just test-e2e``), so xdist
parallelism is disabled.  The ``xdist_group`` marker below is kept as a
safety net: if someone ever runs the full suite with ``--dist loadgroup``
instead of the default ``--dist worksteal``, the marker ensures all E2E
tests land on a single worker to avoid concurrent GPU model loading.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every E2E test with ``xdist_group`` for sequential scheduling."""
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.xdist_group("e2e_sequential"))
