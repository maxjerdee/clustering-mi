"""
Copyright (c) 2025 Max Jerdee. All rights reserved.

clustering-mi: Compute the mutual information between two clusterings of the same objects
"""

from __future__ import annotations

# first party imports
from clustering_mi._input_output import _get_contingency_table
from clustering_mi.mutual_information import (
    mutual_information,
    normalized_mutual_information,
)

from ._version import version as __version__

__all__ = [
    "__version__",
    "_get_contingency_table",
    "mutual_information",
    "normalized_mutual_information",
]
