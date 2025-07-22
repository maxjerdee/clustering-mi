from __future__ import annotations

import numpy as np
import pytest

from clustering_mi._input_output import _get_contingency_table
from clustering_mi.mutual_information import _H_ng_G_alpha, _H_ngc_G_nc_alpha, _stirling_mutual_information, _traditional_mutual_information, _adjusted_mutual_information, _reduced_flat_mutual_information, _reduced_mutual_information, mutual_information, normalized_mutual_information


def test__stirling_mutual_information():

    g = [1,1,2,2,2,3,3,3]
    c = [1,1,1,2,2,2,2,2]
    I_true = 4.88058 # Computed using Mathematica in Tests.nb    
    contingency_table = _get_contingency_table(c, g)
    I = _stirling_mutual_information(contingency_table)
    assert I == pytest.approx(I_true, rel=1e-5)


def test__traditional_mutual_information():

    c = [1,1,1,2,2,2,2,2]
    g = [1,1,2,2,2,3,3,3]
    I_true = 4.22239 # Computed using Mathematica in Tests.nb

    contingency_table = _get_contingency_table(c, g)
    I = _traditional_mutual_information(contingency_table)
    assert I == pytest.approx(I_true, rel=1e-5)


def test__adjusted_mutual_information():
    c = [1,1,1,2,2,2,2,2]
    g = [1,1,2,2,2,3,3,3]
    I_true = 2.74404 # Computed using Mathematica in Tests.nb

    contingency_table = _get_contingency_table(c, g)
    I = _adjusted_mutual_information(contingency_table)
    assert I == pytest.approx(I_true, rel=1e-5)


def test__reduced_flat_mutual_information():
    c = [1,1,1,2,2,2,2,2]
    g = [1,1,2,2,2,3,3,3]
    I_true = 1.07763 # Computed using Mathematica in Tests.nb
    contingency_table = _get_contingency_table(c, g)
    I = _reduced_flat_mutual_information(contingency_table)
    assert I == pytest.approx(I_true, rel=1e-5)    


def test__reduced_mutual_information():
    c = [1,1,1,2,2,2,2,2]
    g = [1,1,2,2,2,3,3,3]
    I_true = 0.19313 # Computed using Mathematica in Tests.nb
    contingency_table = _get_contingency_table(c, g)
    I = _reduced_mutual_information(contingency_table)
    assert I == pytest.approx(I_true, rel=1e-5)


def test__H_ng_G_alpha():
    ng = np.array([30, 20])
    alpha = 0.5
    H_ng_G_alpha_true = 4.35361  # Computed using Mathematica in Tests.nb
    H_ng_G_alpha = _H_ng_G_alpha(ng, alpha)
    assert H_ng_G_alpha == pytest.approx(H_ng_G_alpha_true, rel=1e-5)


def test__H_ngc_G_nc_alpha():
    ngc = np.array([
        [1, 0, 3],
        [2, 2, 0]
    ])
    alpha = 0.5
    H_ngc_G_nc_alpha_true = 3.81796  # Computed using Mathematica in Tests.nb
    H_ngc_G_nc_alpha = _H_ngc_G_nc_alpha(ngc, alpha)
    assert H_ngc_G_nc_alpha == pytest.approx(H_ngc_G_nc_alpha_true, rel=1e-5)


def test__reduced_mutual_information():
    c = [1,1,1,2,2,2,2,2]
    g = [1,1,2,2,2,3,3,3]
    I_true = 0.160127 # Computed using Mathematica in Tests.nb
    contingency_table = _get_contingency_table(c, g)
    I = _reduced_mutual_information(contingency_table)
    assert I == pytest.approx(I_true, rel=1e-5)


def test_mutual_information():
    c = [1,1,1,2,2,2,2,2]
    g = [1,1,2,2,2,3,3,3]
    
    # - "reduced" (default): Reduced mutual information (RMI), reduction of https://arxiv.org/pdf/2405.05393, note that this can be (slightly) asymmetric.
    I_true = 0.160127 # Computed using Mathematica in Tests.nb
    I = mutual_information(c, g, variation="reduced")
    assert I == pytest.approx(I_true, rel=1e-5)

    # - "reduced_flat": Reduced mutual information (RMI), flat reduction of https://arxiv.org/pdf/1907.12581
    I_true = 1.07763 # Computed using Mathematica in Tests.nb
    I = mutual_information(c, g, variation="reduced_flat")
    assert I == pytest.approx(I_true, rel=1e-5)
    
    # - "adjusted": Adjusted mutual information (AMI), correcting for chance: https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf
    I_true = 2.74404 # Computed using Mathematica in Tests.nb
    I = mutual_information(c, g, variation="adjusted")
    assert I == pytest.approx(I_true, rel=1e-5)
    
    # - "traditional": Traditional mutual information (MI), microcanonical
    I_true = 4.22239 # Computed using Mathematica in Tests.nb
    I = mutual_information(c, g, variation="traditional")
    assert I == pytest.approx(I_true, rel=1e-5)

    # - "stirling": Stirling's approximation of the traditional mutual information, equal to the mutual information of the corresponding probability distributions (times the number of objects).
    I_true = 4.88058 # Computed using Mathematica in Tests.nb
    I = mutual_information(c, g, variation="stirling")
    assert I == pytest.approx(I_true, rel=1e-5)

    # Bad variation types
    with pytest.raises(ValueError):
        mutual_information(c, g, variation="unknown")


def test_normalized_mutual_information():
    #  - "second" (default): Asymmetric normalization, measures how much the first labeling tells us about the second, as a fraction of all there is to know about the second labeling.
    c = [1,1,1,2,2,2,2,2]
    g = [1,1,2,2,2,3,3,3]
    NMI_true = 0.0202079
    NMI = normalized_mutual_information(c, g, normalization="second")
    assert NMI == pytest.approx(NMI_true, rel=1e-5)

    # - "first": Asymmetric normalization, measures how much the second labeling tells us about the first, as a fraction of all there is to know about the first labeling.
    NMI_true = 0.412743 # Computed using Mathematica in Tests.nb
    NMI = normalized_mutual_information(c, g, normalization="first", variation="adjusted")
    assert NMI == pytest.approx(NMI_true, rel=1e-5)

    # - "mean": Symmetric normalization by the arithmetic mean of the two entropies.
    NMI_true = 0.48501 # Computed using Mathematica in Tests.nb
    NMI = normalized_mutual_information(c, g, normalization="mean", variation="stirling")
    assert NMI == pytest.approx(NMI_true, rel=1e-5)

    # - "min": Normalize by the minimum of the two entropies.
    NMI_true = 0.727077 # Computed using Mathematica in Tests.nb
    NMI = normalized_mutual_information(c, g, normalization="min", variation="traditional")
    assert NMI == pytest.approx(NMI_true, rel=1e-5)

    # - "max": Normalize by the maximum of the two entropies.
    NMI_true = 0.267954 # Computed using Mathematica in Tests.nb
    NMI = normalized_mutual_information(c, g, normalization="max", variation="reduced_flat")
    assert NMI == pytest.approx(NMI_true, rel=1e-5)

    # - "geometric": Normalize by the geometric mean of the two entropies.
    NMI_true = 0.05534 # Computed using Mathematica in Tests.nb
    NMI = normalized_mutual_information(c, g, normalization="geometric", variation="reduced")
    assert NMI == pytest.approx(NMI_true, rel=1e-5)

    # - "none": No normalization, returns the mutual information in bits.
    NMI_true = 0.160127 # Computed using Mathematica in Tests.nb
    NMI = normalized_mutual_information(c, g, normalization="none", variation="reduced")
    assert NMI == pytest.approx(NMI_true, rel=1e-5)

    # Bad normalization types
    with pytest.raises(ValueError):
        normalized_mutual_information(c, g, normalization="unknown", variation="reduced")

