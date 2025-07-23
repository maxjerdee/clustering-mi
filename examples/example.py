# Basic example usage of the clustering_mi package as described in the README.md file.
import clustering_mi

# Load two labelings of the same set of objects

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