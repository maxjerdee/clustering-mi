# Speed to calculate variations of the mutual information

import clustering_mi as cmi

import numpy as np
import time

# Fairly large examples of 10000 objects randomly split into two partitions of 100 groups each.
labels1 = np.random.randint(0, 100, size=10000)
labels2 = np.random.randint(0, 100, size=10000)

variations = ["reduced", "reduced_flat", "adjusted", "traditional", "stirling"]

for variation in variations:
    # Measure time for mutual information computation
    start_time = time.time()
    NMI = cmi.normalized_mutual_information(labels1, labels2, variation=variation)
    end_time = time.time()

    print(f"{variation} NMI = {NMI:.3f} (computed in {end_time - start_time:.4f} seconds)")


# Results (on a laptop circa 2025):
# reduced NMI = 0.000 (computed in 2.1774 seconds)
# reduced_flat NMI = -0.023 (computed in 0.1295 seconds)
# adjusted NMI = 0.002 (computed in 17.0538 seconds)
# traditional NMI = 0.271 (computed in 0.1388 seconds)
# stirling NMI = 0.124 (computed in 0.1425 seconds)