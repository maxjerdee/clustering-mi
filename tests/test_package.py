from __future__ import annotations

import importlib.metadata

import clustering_mi as m


def test_version():
    assert importlib.metadata.version("clustering_mi") == m.__version__
