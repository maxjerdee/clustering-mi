# Reproducing figures from the paper
# "Normalized mutual information is a biased measure for classification"
# (https://arxiv.org/abs/2307.01282)
import clustering_mi as cmi

# Figure 1

print("Figure 1:")

normalization_names = {"none": "I_0", "second": "NMI_0^A", "mean": "NMI_0^S"}

for normalization, name in normalization_names.items():
    NMI = cmi.normalized_mutual_information(
        "data/2307.01282/fig_1_1.txt",
        normalization=normalization,
        variation="traditional",
    )  # Traditional mutual information
    print(f"{name}(c_1;g) = {NMI:.3f}")
    NMI = cmi.normalized_mutual_information(
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
    NMI = cmi.normalized_mutual_information(
        "data/2307.01282/fig_2_1.txt",
        normalization=normalization,
        variation="traditional",
    )  # Traditional mutual information
    print(f"{name}(c_1;g) = {NMI:.3f}")
    NMI = cmi.normalized_mutual_information(
        "data/2307.01282/fig_2_2.txt",
        normalization=normalization,
        variation="traditional",
    )  # Traditional mutual information
    print(f"{name}(c_2;g) = {NMI:.3f}")

print()

# Figure 3

print("Figure 3:")

print("Contingency Table:")
print(cmi._get_contingency_table("data/2307.01282/fig_3.txt"))

normalizations = ["second", "mean", "first"]
variations = ["traditional", "adjusted", "reduced"]

for normalization in normalizations:
    for variation in variations:
        NMI = cmi.normalized_mutual_information(
            "data/2307.01282/fig_3.txt",
            normalization=normalization,
            variation=variation,
        )
        print(
            f"Normalization: {normalization}, Variation: {variation}, NMI = {NMI:.3f}"
        )

print()

# Figure 4

print("Figure 4:")

dataset_names_dict = {"InfoMap": "fig_4_1.txt", "Modularity (γ = 2)": "fig_4_2.txt"}

variations = ["traditional", "adjusted", "reduced"]
normalizations = ["second", "mean"]

for dataset_name, dataset_file in dataset_names_dict.items():
    print(f"Dataset: {dataset_name}")
    for variation in variations:
        for normalization in normalizations:
            NMI = cmi.normalized_mutual_information(
                f"data/2307.01282/{dataset_file}",
                normalization=normalization,
                variation=variation,
            )
            print(
                f"Normalization: {normalization}, Variation: {variation}, NMI = {NMI:.3f}"
            )