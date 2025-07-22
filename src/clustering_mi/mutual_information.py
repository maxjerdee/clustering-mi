# Compute the mutual information and associated entropies
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from clustering_mi._input_output import _get_contingency_table

# from scipy.optimize import minimize_scalar # Used to optimize the alpha parameter in the Dirichlet-multinomial reduction
from clustering_mi._util import (
    _log_binom,
    _log_factorial,
    _log_Omega_EC,
    _minimize_golden_section_log,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _stirling_mutual_information(contingency_table: ArrayLike) -> float:
    """
    Compute the Stirling approximated mutual information for the given contingency table.
    This is the mutual information of the corresponding probability distributions (times the number of objects).

    Parameters
    ----------
    contingency_table : ArrayLike
        Contingency table T as a 2D NumPy array where T[r][s] counts the number of objects with label r in the ground truth g and label s in the candidate c.

    Returns
    -------
    float
        Mutual information (base 2)
    """

    # Compute summary information
    n: float = np.sum(contingency_table)
    nc = np.sum(contingency_table, axis=0)
    ng = np.sum(contingency_table, axis=1)

    MI = 0
    for r, ng_r in enumerate(ng):
        for s, nc_s in enumerate(nc):
            if contingency_table[r, s] > 0:
                MI += contingency_table[r, s] * np.log(
                    n * contingency_table[r, s] / (ng_r * nc_s)
                )

    # Convert to bits (log base 2)
    MI /= np.log(2)

    return float(MI)


def _traditional_mutual_information(contingency_table: ArrayLike) -> float:
    """
    Compute the unreduced microcanonical mutual information and entropies of the ground truth and candidate labelings.

    Parameters
    ----------
    contingency_table : ArrayLike
        Contingency table T as a 2D NumPy array where T[r][s] counts the number of objects with label r in the ground truth g and label s in the candidate c.

    Returns
    -------
    float
        Mutual information (base 2).
    """

    # Compute summary information
    n: float = np.sum(contingency_table)
    nc = np.sum(contingency_table, axis=0)
    ng = np.sum(contingency_table, axis=1)

    MI = (
        _log_factorial(n)
        - np.sum(_log_factorial(ng))
        - np.sum(_log_factorial(nc))
        + np.sum(_log_factorial(contingency_table.flatten()))
    )

    # Convert to bits (log base 2)
    MI /= np.log(2)

    return float(MI)


def _adjusted_mutual_information(contingency_table: ArrayLike) -> float:
    """
    Compute the adjusted mutual information, which corrects the mutual information for chance (random permutations of the labels).

    Parameters
    ----------
    contingency_table : ArrayLike
        Contingency table T as a 2D NumPy array where T[r][s] counts the number of objects with label r in the ground truth g and label s in the candidate c.

    Returns
    -------
    float
        Adjusted mutual information (base 2).
    """

    n: float = np.sum(contingency_table)
    nc = np.sum(contingency_table, axis=0)
    ng = np.sum(contingency_table, axis=1)
    qg = len(ng)
    qc = len(nc)

    EMI = 0
    for r in range(qg):
        for s in range(qc):
            for ngc in range(max(1, ng[r] + nc[s] - n), min(ng[r], nc[s]) + 1):
                EMI += (
                    ngc
                    * (np.log(n) + np.log(ngc) - np.log(ng[r]) - np.log(nc[s]))
                    * np.exp(
                        _log_binom(nc[s], ngc)
                        + _log_binom(n - nc[s], ng[r] - ngc)
                        - _log_binom(n, ng[r])
                    )
                )

    # Convert to bits (log base 2)
    EMI /= np.log(2)

    return float(_stirling_mutual_information(contingency_table) - EMI)


def _reduced_flat_mutual_information(contingency_table: ArrayLike) -> float:
    """
    Compute the reduced mutual information, using the flat reduction method of https://arxiv.org/pdf/1907.12581.

    Parameters
    ----------
    contingency_table : ArrayLike
        Contingency table T as a 2D NumPy array where T[r][s] counts the number of objects with label r in the ground truth g and label s in the candidate c.

    Returns
    -------
    float
        Reduced mutual information (base 2).
    """

    nc = np.sum(contingency_table, axis=0)
    ng = np.sum(contingency_table, axis=1)

    logOmega = _log_Omega_EC(nc, ng)

    return float(_traditional_mutual_information(contingency_table) - logOmega)


def _H_ng_G_alpha(ng: ArrayLike, alpha: float) -> float:
    """
    Compute the entropy of a vector of group sizes given the concentration parameter alpha.

    Parameters
    ----------
    ng : list or np.ndarray
        Vector of group sizes.
    alpha : float
        Concentration parameter.

    Returns
    -------
    float
        Entropy of the group sizes. (base e).

    """
    n: float = np.sum(ng)
    q = len(ng)

    H_ng = _log_binom(
        n + q * alpha - 1, q * alpha - 1
    )  # Dirichlet-multinomial distribution
    for r in range(q):
        H_ng -= _log_binom(ng[r] + alpha - 1, alpha - 1)

    return H_ng


def _H_ngc_G_nc_alpha(ngc: ArrayLike, alpha: float) -> float:
    """
    Compute the entropy of a contingency table given knowledge of the column sums and the concentration parameter alpha.

    Parameters
    ----------
    ngc : list or np.ndarray
        Vector of column sums.
    alpha : float
        Concentration parameter.

    Returns
    -------
    float
        Entropy of the contingency table. (base e).
    """

    qg = int(ngc.shape[0])
    qc = int(ngc.shape[1])
    nc = np.sum(ngc, axis=0)  # Column sums

    H_ngc = 0.0
    for s in range(qc):
        H_ngc += _log_binom(
            nc[s] + float(qg) * alpha - 1, float(qg) * alpha - 1
        )  # Independent Dirichlet-multinomial distributions of the columns
        for r in range(qg):
            H_ngc -= _log_binom(ngc[r, s] + alpha - 1, alpha - 1)

    return H_ngc


def _reduced_mutual_information(contingency_table: ArrayLike) -> float:
    """
    Compute the reduced mutual information, using the Dirichlet-multinomial reduction of https://arxiv.org/pdf/2405.05393.

    Parameters
    ----------
    contingency_table : ArrayLike
        Contingency table T as a 2D NumPy array where T[r][s] counts the number of objects with label r in the ground truth g and label s in the candidate c.

    Returns
    -------
    float
        Reduced mutual information (base 2).
    """
    n: float = np.sum(contingency_table)
    nc = np.sum(contingency_table, axis=0)
    ng = np.sum(contingency_table, axis=1)

    # Range of values of the concentration parameter alpha to consider in the Dirichlet-multinomial transmissions
    min_alpha = 0.0001
    max_alpha = 10000

    # H_g
    H_qg: float = np.log(n)

    _, H_ng_G_alpha = _minimize_golden_section_log(
        lambda alpha: _H_ng_G_alpha(ng, alpha), min_alpha, max_alpha
    )  # Note that we neglect the cost to transmit the alpha parameter here, although a fixed cost would cancel in the mutual information calculation.
    H_ng_G_alpha = float(H_ng_G_alpha)
    H_g_G_ng: float = float(_log_factorial(n)) - float(np.sum(_log_factorial(ng)))
    H_g: float = H_qg + H_ng_G_alpha + H_g_G_ng

    # H_g_G_c
    _, H_ngc_G_nc_alpha = _minimize_golden_section_log(
        lambda alpha: _H_ngc_G_nc_alpha(contingency_table, alpha), min_alpha, max_alpha
    )
    H_ngc_G_nc_alpha = float(H_ngc_G_nc_alpha)
    H_g_G_c_ngc: float = float(np.sum(_log_factorial(nc))) - float(
        np.sum(_log_factorial(contingency_table.flatten()))
    )
    H_g_G_c: float = H_qg + H_ngc_G_nc_alpha + H_g_G_c_ngc

    MI: float = H_g - H_g_G_c

    return float(MI / np.log(2))  # Convert to bits (log base 2)


def normalized_mutual_information(
    input_data_1: ArrayLike | str,
    input_data_2: ArrayLike | None = None,
    *,
    variation: str = "reduced",
    normalization: str = "second",
) -> ArrayLike:
    """
    Compute the normalized mutual information between two labelings from a pair of lists, the name of a space separated file of labels,
    or a contingency table. Can specify the variation of mutual information and type of normalization.
    For the asymmetric (default) normalization, the result is reported as a fraction of the entropy of the second labeling, which is considered the ground truth.

    Raises AssertionError for invalid inputs.

    Parameters
    ----------
    input_data_1 : ArrayLike or str
        First argument. This will either be a 2D array-like which specifies the contingency table whose columns are the first labeling and rows are the second labeling,
        or a string which is the path to a file containing a list of pairs of labels,
        or a 1-D array-like of labels.
    input_data_2 : ArrayLike, optional
        Second argument. This can only be a 1-D array-like of labels in the case where the first argument is also such a list.
    variation : str, optional
        Variation of mutual information to compute. Options are:
            - "stirling": Stirling's approximation of the traditional mutual information, equal to the mutual information of the corresponding probability distributions (times the number of objects).
            - "reduced" (default): Reduced mutual information (RMI), Dirichlet-multinomial reduction of https://arxiv.org/pdf/2405.05393
            - "reduced_flat": Reduced mutual information (RMI), flat reduction of https://arxiv.org/pdf/1907.12581
            - "adjusted": Adjusted mutual information (AMI), correcting for chance: https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf
            - "traditional": Traditional mutual information (MI), microcanonical
            - "stirling": Stirling's approximation of the traditional mutual information, equal to the mutual information of the corresponding probability distributions (times the number of objects).
    normalization : str, optional
        Type of normalization to apply. Options are:
            - "second" (default): Asymmetric normalization, measures how much the first labeling tells us about the second, as a fraction of all there is to know about the second labeling.
            - "first": Asymmetric normalization, measures how much the second labeling tells us about the first, as a fraction of all there is to know about the first labeling.
            - "mean": Symmetric normalization by the arithmetic mean of the two entropies.
            - "min": Normalize by the minimum of the two entropies.
            - "max": Normalize by the maximum of the two entropies.
            - "geometric": Normalize by the geometric mean of the two entropies.
            - "none": No normalization, returns the mutual information in bits.

    Returns
    -------
    float
        Normalized mutual information
    """

    # Get the contingency table
    contingency_table = _get_contingency_table(input_data_1, input_data_2)

    # Make the candidate-candidate (labeling 1) and truth-truth (label 2) contingency tables
    nc = np.sum(contingency_table, axis=0)
    ng = np.sum(contingency_table, axis=1)
    ncc = np.diag(nc)  # Candidate-candidate
    ngg = np.diag(ng)  # Truth-truth

    # Compute the mutual information between each pair of labelings
    MI_c_g, MI_g_c, MI_c_c, MI_g_g = None, None, None, None  # Values to be computed
    if variation == "stirling":
        MI_c_g = _stirling_mutual_information(
            contingency_table
        )  # Mutual information between candidate and ground truth
        MI_g_c = MI_c_g  # Symmetric measure
        MI_c_c = _stirling_mutual_information(
            ncc
        )  # Mutual information between candidate and candidate
        MI_g_g = _stirling_mutual_information(
            ngg
        )  # Mutual information between ground truth and ground truth
    elif variation == "traditional":
        MI_c_g = _traditional_mutual_information(contingency_table)
        MI_g_c = MI_c_g  # Symmetric measure
        MI_c_c = _traditional_mutual_information(ncc)
        MI_g_g = _traditional_mutual_information(ngg)
    elif variation == "adjusted":
        MI_c_g = _adjusted_mutual_information(contingency_table)
        MI_g_c = MI_c_g  # Symmetric measure
        MI_c_c = _adjusted_mutual_information(ncc)
        MI_g_g = _adjusted_mutual_information(ngg)
    elif variation == "reduced":
        MI_c_g = _reduced_mutual_information(contingency_table)
        MI_g_c = _reduced_mutual_information(
            contingency_table.T
        )  # Transpose to get the ground truth as the first labeling
        MI_c_c = _reduced_mutual_information(ncc)
        MI_g_g = _reduced_mutual_information(ngg)
    elif variation == "reduced_flat":
        MI_c_g = _reduced_flat_mutual_information(contingency_table)
        MI_g_c = _reduced_flat_mutual_information(contingency_table.T)
        MI_c_c = _reduced_flat_mutual_information(ncc)
        MI_g_g = _reduced_flat_mutual_information(ngg)
    else:
        raise ValueError(f"Unknown variation type: {variation}")

    # Compute the normalized mutual information
    if normalization == "second":
        # Asymmetric normalization, measures how much the first labeling tells us about the second, as a fraction of all there is to know about the second labeling
        return MI_c_g / MI_g_g if MI_g_g > 0 else 0
    if normalization == "first":
        # Asymmetric normalization, measures how much the second labeling tells us about the first, as a fraction of all there is to know about the first labeling
        return MI_g_c / MI_c_c if MI_c_c > 0 else 0
    if (
        normalization == "mean"
    ):  # Note that the numerators of these symmetric measures are non-standard in order to account for asymmetries in the calculated MI_c_g vs MI_g_c
        return (MI_c_g + MI_g_c) / (MI_c_c + MI_g_g) if (MI_c_c + MI_g_g) > 0 else 0
    if normalization == "min":
        return (
            min(MI_c_g, MI_g_c) / min(MI_c_c, MI_g_g) if min(MI_c_c, MI_g_g) > 0 else 0
        )
    if normalization == "max":
        return (
            max(MI_c_g, MI_g_c) / max(MI_c_c, MI_g_g) if max(MI_c_c, MI_g_g) > 0 else 0
        )
    if normalization == "geometric":
        return (
            np.sqrt(MI_c_g * MI_g_c) / np.sqrt(MI_c_c * MI_g_g)
            if (MI_c_c * MI_g_g) > 0
            else 0
        )
    if normalization == "none":
        return MI_c_g  # Return the mutual information in bits without normalization (note that this may not be symmetric for the reduced measures)
    raise ValueError(f"Unknown normalization type: {normalization}")


def mutual_information(
    input_data_1: ArrayLike | str,
    input_data_2: ArrayLike | None = None,
    *,
    variation: str = "reduced",
) -> ArrayLike:
    """
    Compute the mutual information between two labelings from a pair of lists, the name of a space separated file of labels,
    or a contingency table. Can specify the variation of mutual information to compute.

    Raises AssertionError for invalid inputs.

    Parameters
    ----------
    input_data_1 : ArrayLike or str
        First argument. This will either be a 2D array-like which specifies the contingency table whose columns are the first labeling and rows are the second labeling,
        or a string which is the path to a file containing a list of pairs of labels,
        or a 1-D array-like of labels.
    input_data_2 : ArrayLike, optional
        Second argument. This can only be a 1-D array-like of labels in the case where the first argument is also such a list.
    variation : str, optional
        Variation of mutual information to compute. Options are:
            - "reduced" (default): Reduced mutual information (RMI), reduction of https://arxiv.org/pdf/2405.05393, note that this can be (slightly) asymmetric.
            - "reduced_flat": Reduced mutual information (RMI), flat reduction of https://arxiv.org/pdf/1907.12581
            - "adjusted": Adjusted mutual information (AMI), correcting for chance: https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf
            - "traditional": Traditional mutual information (MI), microcanonical
            - "stirling": Stirling's approximation of the traditional mutual information, equal to the mutual information of the corresponding probability distributions (times the number of objects).

    Returns
    -------
    float
        Mutual information value in bits (base 2).
    """

    return normalized_mutual_information(
        input_data_1,
        input_data_2,
        variation=variation,
        normalization="none",
    )
