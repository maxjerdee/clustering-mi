# clustering-mi

[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![DOI][doi-badge]][doi-link]

<!-- SPHINX-START -->

**Mutual information between clusterings.**
_Maximilian Jerdee, Alec Kirkley, and Mark Newman._

A Python package for computing the mutual information between two clusterings of
the same set of objects. This implementation includes multiple variations and
normalizations of the mutual information.

The package implements the reduced mutual information (RMI) as described in
[Jerdee, Kirkley, and Newman (2024)](https://arxiv.org/pdf/2405.05393), which
corrects the standard measure's bias towards labelings with too many groups. The
asymmetric normalization of
[Jerdee, Kirkley, and Newman (2023)](https://arxiv.org/abs/2307.01282) is also
included to remove the biases of symmetric normalizations. Data used to generate
the figures in those papers is available in the `examples/data` directory of the
repository.

The mutual information `variation` (passed to both `mutual_information` and
`normalized_mutual_information`) selects the measure:

| `variation`      | Description                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `"reduced"`      | **(default)** Reduced MI, Dirichlet-multinomial reduction ([Jerdee et al. 2024](https://arxiv.org/pdf/2405.05393)). |
| `"reduced_flat"` | Reduced MI, flat reduction ([Newman et al. 2019](https://arxiv.org/pdf/1907.12581)).                     |
| `"adjusted"`     | Adjusted MI (AMI), correcting for chance under random permutations ([Vinh et al. 2010](https://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf)). |
| `"traditional"`  | Traditional (microcanonical) mutual information.                                           |
| `"stirling"`     | Stirling approximation of the traditional MI. |

For `normalized_mutual_information`, the `normalization` selects the denominator:

| `normalization`     | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| `"second"`          | **(default)** Asymmetric: fraction of the second labeling's information recovered. |
| `"first"`           | Asymmetric: fraction of the first labeling's information recovered.               |
| `"mean"`            | Symmetric: normalize by the arithmetic mean of the two entropies.                 |
| `"min"` / `"max"`   | Symmetric: normalize by the minimum / maximum of the two entropies.               |
| `"geometric"`       | Symmetric: normalize by the geometric mean of the two entropies.                  |
| `"none"`            | No normalization; return the mutual information in bits.                          |

## Installation

`clustering-mi` can be installed through pip:

```bash
pip install clustering-mi
```

or built locally with a c++ compiler by cloning [this repository](https://github.com/maxjerdee/clustering-mi) and running

```bash
pip install .
```

in the base directory.

## Typical usage

Once installed, the package can be imported as

```python
import clustering_mi as cmi
```

Note that this is not `import clustering-mi`.

Two clusterings (or "labelings") can be loaded in several ways; the names of the groups are
irrelevant:

```python
# As arrays:
labels1 = ["red", "red", "red", "blue", "blue", "blue", "green", "green"]
labels2 = [1, 1, 1, 1, 2, 2, 2, 2]

# As a contingency table, i.e., a matrix that counts label co-occurrences.
# Columns are the first labeling, rows are the second labeling:
contingency_table = [[3, 1, 0], [0, 2, 2]]

# Or as the path to a space-separated file, one object (label1 label2) per line:
filename = "data/example.txt"
```

where `data/example.txt` contains one labeled object per line:

```text
red 1
red 1
red 1
blue 1
blue 2
blue 2
green 2
green 2
```

The package can then compute the mutual information (in bits) between the
two labelings from any format:

```python
# Defaults to the reduced mutual information (RMI)
mutual_information = cmi.mutual_information(labels1, labels2)  # From lists
mutual_information = cmi.mutual_information(contingency_table)  # From contingency table
mutual_information = cmi.mutual_information(filename)  # Reads filename

print(f"Mutual Information: {mutual_information:.3f} (bits)")

# Compute other variants using the "variation" parameter.
# Correcting for chance (random permutations)
adjusted_mutual_information = cmi.mutual_information(labels1, labels2, variation="adjusted")
# Traditional mutual information
traditional_mutual_information = cmi.mutual_information(labels1, labels2, variation="traditional")
```

The package can also compute the normalized mutual information (NMI) between the two
labelings, a measure bounded above by 1 when the two labelings are
identical. Depending on the application, a symmetric or asymmetric normalization
may be appropriate.

```python
# Symmetric normalization
normalized_mutual_information = cmi.normalized_mutual_information(labels1, labels2, normalization="mean")
# "Normalized Mutual Information" most commonly refers to the Stirling-approximated mutual information
# divided by the mean of the entropies of the two labelings, although this is not our preferred measure.
normalized_stirling_mutual_information = cmi.normalized_mutual_information(labels1, labels2, variation="stirling", normalization="mean")

print(f"(symmetric) Normalized Mutual Information (labels1 <-> labels2): {normalized_mutual_information:.3f}")

# Asymmetric normalization measures how much the first labeling tells us about the second,
# as a fraction of all there is to know about the second labeling.
# This form is appropriate when the second labeling is a "ground truth" and the first is a prediction.
asymmetric_normalized_mutual_information_1_2 = cmi.normalized_mutual_information(labels1, labels2, normalization="second")
# Or when the first labeling is the ground truth and the second is a prediction.
asymmetric_normalized_mutual_information_2_1 = cmi.normalized_mutual_information(labels1, labels2, normalization="first")

print(f"(asymmetric) Normalized Mutual Information (labels1 -> labels2): {asymmetric_normalized_mutual_information_1_2:.3f}")
print(f"(asymmetric) Normalized Mutual Information (labels2 -> labels1): {asymmetric_normalized_mutual_information_2_1:.3f}")
```

Further usage examples can be found in the `examples` directory of the
repository and the [package documentation][rtd-link].

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/maxjerdee/clustering-mi/workflows/CI/badge.svg
[actions-link]:             https://github.com/maxjerdee/clustering-mi/actions
[codecov-badge]:            https://codecov.io/github/maxjerdee/clustering-mi/graph/badge.svg?token=In4SI7LJjQ
[codecov-link]:             https://codecov.io/github/maxjerdee/clustering-mi
[pypi-link]:                https://pypi.org/project/clustering-mi/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/clustering-mi
[pypi-version]:             https://img.shields.io/pypi/v/clustering-mi
[rtd-badge]:                https://readthedocs.org/projects/clustering-mi/badge/?version=stable
[rtd-link]:                 https://clustering-mi.readthedocs.io/en/stable
[doi-badge]:                https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17211478-blue.svg
[doi-link]:                 https://doi.org/10.5281/zenodo.17211478
<!-- prettier-ignore-end -->
