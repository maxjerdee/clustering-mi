# clustering-mi

[![Documentation Status][rtd-badge]][rtd-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/maxjerdee/clustering-mi/workflows/CI/badge.svg
[actions-link]:             https://github.com/maxjerdee/clustering-mi/actions
[codecov-badge]:            https://codecov.io/github/maxjerdee/clustering-mi/graph/badge.svg?token=In4SI7LJjQ
[codecov-link]:             https://codecov.io/github/maxjerdee/clustering-mi
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/maxjerdee/clustering-mi/discussions
[pypi-link]:                https://pypi.org/project/clustering-mi/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/clustering-mi
[pypi-version]:             https://img.shields.io/pypi/v/clustering-mi
[rtd-badge]:                https://readthedocs.org/projects/clustering-mi/badge/?version=stable
[rtd-link]:                 https://clustering-mi.readthedocs.io/en/stable

<!-- prettier-ignore-end -->

### Computing mutual information between clusterings

##### Maximilian Jerdee, Alec Kirkley, and Mark Newman

A Python package for computing the mutual information between two clusterings of
the same set of objects. This implementation includes multiple variations and
normalizations of the mutual information.

The package implements the reduced mutual information (RMI) as described in
[Jerdee, Kirkley, and Newman (2024)](https://arxiv.org/pdf/2405.05393), which
corrects the standard measure's bias towards labelings with too many groups. The
asymmetric normalization of
[Jerdee, Kirkley, and Newman (2023)](https://arxiv.org/abs/2307.01282) is also
included to remove the biases of symmetric normalizations.

## Installation

`clustering-mi` can be installed through pip:

```bash
pip install clustering-mi
```

or built locally by cloning this repository and running

```bash
pip install .
```

in the base directory.

## Typical usage

Once installed, the package can be imported as

```python
import clustering_mi
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

# Or as a space-separated file:
"""
red 1
red 1
red 1
blue 1
blue 2
blue 2
green 2
green 2
"""
filename = "data/example.txt"
```

The package can then compute the mutual information (in bits) between the
two labelings from any format:

```python
# Defaults to the reduced mutual information (RMI)
mutual_information = clustering_mi.mutual_information(labels1, labels2)  # From lists
mutual_information = clustering_mi.mutual_information(contingency_table)  # From contingency table
mutual_information = clustering_mi.mutual_information(filename)  # Reads filename

print(f"Mutual Information: {mutual_information:.3f} (bits)")

# Compute other variants using the "variation" parameter.
# Correcting for chance (random permutations)
adjusted_mutual_information = clustering_mi.mutual_information(labels1, labels2, variation="adjusted")  
# Traditional mutual information
traditional_mutual_information = clustering_mi.mutual_information(labels1, labels2, variation="traditional")
```

The package can also compute the normalized mutual information (NMI) between the two
labelings, a measure bounded above by 1 when the two labelings are
identical. Depending on the application, a symmetric or asymmetric normalization
may be appropriate.

```python
# Symmetric normalization
normalized_mutual_information = clustering_mi.normalized_mutual_information(labels1, labels2, normalization="mean")
# "Normalized Mutual Information" most commonly refers to the Stirling-approximated mutual information
# divided by the mean of the entropies of the two labelings, although this is not our preferred measure.
normalized_stirling_mutual_information = clustering_mi.normalized_mutual_information(labels1, labels2, variation="stirling", normalization="mean")

print(f"(symmetric) Normalized Mutual Information (labels1 <-> labels2): {normalized_mutual_information:.3f}")

# Asymmetric normalization measures how much the first labeling tells us about the second,
# as a fraction of all there is to know about the second labeling.
# This form is appropriate when the second labeling is a "ground truth" and the first is a prediction.
asymmetric_normalized_mutual_information_1_2 = clustering_mi.normalized_mutual_information(labels1, labels2, normalization="second")
# Or when the first labeling is the ground truth and the second is a prediction.
asymmetric_normalized_mutual_information_2_1 = clustering_mi.normalized_mutual_information(labels1, labels2, normalization="first")

print(f"(asymmetric) Normalized Mutual Information (labels1 -> labels2): {asymmetric_normalized_mutual_information_1_2:.3f}")
print(f"(asymmetric) Normalized Mutual Information (labels2 -> labels1): {asymmetric_normalized_mutual_information_2_1:.3f}")
```

Further usage examples can be found in the `examples` directory of the
repository and the [package documentation][rtd-link].