# Checking input validity and converting each of the input types into a contingency table
# to be passed to the functions that calculate the mutual information.
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_contingency_table(
    input_data_1: ArrayLike | str, input_data_2: ArrayLike | None = None
) -> ArrayLike:
    """
    Get the contingency table T between two labelings from a pair of lists or the name of a space separated file of labels.
    T[r][s] counts the number of objects with label s in the first labeling and label r in the second labeling.

    Raises AssertionError for invalid inputs.

    Parameters
    ----------
    input_data_1 : ArrayLike or str
        First argument. This will either be a 2D array-like which specifies the contingency table,
        or a string which is the path to a file containing a list of pairs of labels,
        or a 1-D array-like of labels.
    input_data_2 : ArrayLike, optional
        Second argument. This can only be a 1-D array-like of labels in the case where the first argument is also such a list.

    Returns
    -------
    ArrayLike
        Contingency table as a 2D NumPy array.
    """

    # Print the types of the inputs
    if isinstance(input_data_1, str):
        # If the first argument is a string, it must be a file path
        # Check if the file exists
        if not Path(input_data_1).is_file():
            raise FileNotFoundError(f"File {input_data_1} does not exist.")
        # Read the file and convert it to a contingency table
        with Path(input_data_1).open(encoding="utf-8") as f:
            lines = f.readlines()
        labels = [line.strip().split() for line in lines]
        # Make sure there are two labels per line, report the offending line
        if not all(len(label) == 2 for label in labels):
            raise AssertionError(
                "Each line in the file must contain exactly two labels."
            )
        labels1, labels2 = zip(*labels)
        return _get_contingency_table(labels1, labels2)

    if isinstance(input_data_1, (list, np.ndarray, tuple)):
        # If the array is 2D, it is a contingency table
        input_data_1 = np.array(input_data_1)
        if input_data_1.ndim == 2:
            # Remove any empty rows or columns
            input_data_1 = input_data_1[~np.all(input_data_1 == 0, axis=1)]
            input_data_1 = input_data_1[:, ~np.all(input_data_1 == 0, axis=0)]
            if input_data_1.size == 0:
                raise AssertionError(
                    "The contingency table is empty after removing empty rows and columns."
                )
            if not np.issubdtype(input_data_1.dtype, np.integer):
                raise AssertionError(
                    "The contingency table must contain integer values."
                )
            if np.any(input_data_1 < 0):
                raise AssertionError(
                    "The contingency table must not contain negative values."
                )
            return input_data_1
        if input_data_1.ndim == 1:
            # If the array is 1D, it should be a list of labels, and we need a second argument
            if isinstance(input_data_2, (list, np.ndarray, tuple)):
                input_data_2 = np.array(input_data_2)
                if input_data_2.ndim != 1:
                    raise AssertionError(
                        "The second argument must be a 1D array-like of labels."
                    )
                # Create a contingency table from the two labelings.
                # Rows index labels2 and columns index labels1; we flip this
                # around in accordance with how the table is defined in
                # https://arxiv.org/abs/2307.01282.
                # Map each label to a contiguous index and accumulate counts in a
                # single O(n) pass (bincount over flattened indices) rather than
                # an O(qg * qc * n) scan.
                labels1, labels2 = input_data_1, input_data_2
                unique_labels1, inv1 = np.unique(labels1, return_inverse=True)
                unique_labels2, inv2 = np.unique(labels2, return_inverse=True)
                inv1 = np.ravel(inv1)  # numpy 2.x may return a 2-D inverse
                inv2 = np.ravel(inv2)
                q1 = len(unique_labels1)
                q2 = len(unique_labels2)
                flat = inv2 * q1 + inv1
                return np.bincount(flat, minlength=q1 * q2).reshape(q2, q1)
            raise AssertionError(
                "If the first argument is a 1D array-like of labels, a second list of labels must be provided."
            )
        raise AssertionError("First argument has too many dimensions.")

    # If we reach this point, the input is invalid
    raise TypeError(
        "First argument must be a 2D array-like (contingency table), a 1D array-like of labels, or a string file path."
    )
