from __future__ import annotations

import os

import numpy as np
import pytest

from clustering_mi._input_output import _get_contingency_table

abs_path = os.path.dirname(os.path.realpath(__file__))


def test__get_contingency_table():
    # This is a bit finicky, since the permutation of the group labels is arbitrary
    contingency_table_true = [[1, 0, 3], [2, 2, 0]]

    # Passing lists of labels
    labels1 = ["red", "red", "red", "blue", "blue", "blue", "green", "green"]
    labels2 = [1, 1, 1, 1, 2, 2, 2, 2]
    contingency_table_found = _get_contingency_table(labels1, labels2)
    assert np.array_equal(contingency_table_found, contingency_table_true)

    # Passing a file
    contingency_table_found = _get_contingency_table(abs_path + "/data/example.txt")
    assert np.array_equal(contingency_table_found, contingency_table_true)
    contingency_table_found = _get_contingency_table(
        abs_path + "/data/example_commas.txt"
    )
    assert np.array_equal(contingency_table_found, contingency_table_true)
    contingency_table_found = _get_contingency_table(
        abs_path + "/data/example_tabs.txt"
    )
    assert np.array_equal(contingency_table_found, contingency_table_true)

    with pytest.raises(AssertionError):
        _get_contingency_table(abs_path + "/data/example_missing_values.txt")

    # Passing a contingency table
    contingency_table_found = _get_contingency_table(contingency_table_true)
    assert np.array_equal(contingency_table_found, contingency_table_true)

    contingency_table_non_integer = [[3.2, 1, 0], [0, 2, 2]]
    with pytest.raises(AssertionError):
        _get_contingency_table(contingency_table_non_integer)

    contingency_table_negative = [[3, -1, 0], [0, 2, 2]]
    with pytest.raises(AssertionError):
        _get_contingency_table(contingency_table_negative)
