"""E2E tests — real I/O, real network, real GPU models, marked @pytest.mark.slow.

All E2E tests run sequentially (single xdist group) because GPU models
are too heavy to run in parallel on a single machine.
"""
