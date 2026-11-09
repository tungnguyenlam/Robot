import pytest

from egograph.data.index import index_path


def require_index(dataset: str):
    """Skip data-dependent tests on machines without the dataset (anywhere except amdserver)."""
    return pytest.mark.skipif(not index_path(dataset).exists(), reason=f"{dataset} index not built here")
