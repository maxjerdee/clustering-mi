# Reproducing figures from the paper
# "Normalized mutual information is a biased measure for classification"
# (https://arxiv.org/abs/2307.01282)
from __future__ import annotations

import clustering_mi

# Figure 1

print("Figure 1:")

normalization_names = {"none": "I_0", "second": "NMI_0^A", "mean": "NMI_0^S"}

for normalization, name in normalization_names.items():
    NMI = clustering_mi.normalized_mutual_information(
        "data/2307.01282/fig_1_1.txt",
        normalization=normalization,
        variation="traditional",
    )  # Traditional mutual information
    print(f"{name}(c_1;g) = {NMI:.3f}")
    NMI = clustering_mi.normalized_mutual_information(
        "data/2307.01282/fig_1_2.txt",
        normalization=normalization,
        variation="traditional",
    )  # Traditional mutual information
    print(f"{name}(c_2;g) = {NMI:.3f}")

print()

# Figure 2

print("Figure 2:")

normalization_names = {"none": "I_0", "second": "NMI_0^A", "mean": "NMI_0^S"}

for normalization, name in normalization_names.items():
    NMI = clustering_mi.normalized_mutual_information(
        "data/2307.01282/fig_2_1.txt",
        normalization=normalization,
        variation="traditional",
    )  # Traditional mutual information
    print(f"{name}(c_1;g) = {NMI:.3f}")
    NMI = clustering_mi.normalized_mutual_information(
        "data/2307.01282/fig_2_2.txt",
        normalization=normalization,
        variation="traditional",
    )  # Traditional mutual information
    print(f"{name}(c_2;g) = {NMI:.3f}")

print()

# Figure 3

print("Figure 3:")

print("Contingency Table:")
print(clustering_mi._get_contingency_table("data/2307.01282/fig_3.txt"))

normalizations = ["second", "mean", "first"]
variations = ["traditional", "adjusted", "reduced"]

for normalization in normalizations:
    for variation in variations:
        NMI = clustering_mi.normalized_mutual_information(
            "data/2307.01282/fig_3.txt",
            normalization=normalization,
            variation=variation,
        )
        print(
            f"Normalization: {normalization}, Variation: {variation}, NMI = {NMI:.3f}"
        )
