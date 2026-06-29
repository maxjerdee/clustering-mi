/* Core mutual-information routines for clustering-mi.
 *
 * Language-agnostic numeric core (no pybind11 / Python dependency). Each
 * variation takes a contingency table T (T[r][s] counts objects with row label
 * r and column label s) and returns the mutual information in BITS (base 2).
 *
 * These mirror the reference Python implementation in
 * src/clustering_mi/mutual_information.py exactly.
 */

#ifndef CMI_CORE_H
#define CMI_CORE_H

#include <vector>

namespace cmi {

using Table = std::vector<std::vector<long>>;

// Stirling approximation of the traditional mutual information (bits).
double stirling_mutual_information(const Table& T);

// Traditional microcanonical mutual information (bits).
double traditional_mutual_information(const Table& T);

// Adjusted mutual information, corrected for chance (bits).
double adjusted_mutual_information(const Table& T);

// Reduced mutual information via the flat reduction (arXiv:1907.12581) (bits).
double reduced_flat_mutual_information(const Table& T);

// Reduced mutual information via the Dirichlet-multinomial reduction
// (arXiv:2405.05393) (bits).
double reduced_mutual_information(const Table& T);

// Entropy of a vector of group sizes given concentration alpha (base e).
double H_ng_G_alpha(const std::vector<long>& ng, double alpha);

// Entropy of a contingency table given column sums and concentration alpha
// (base e).
double H_ngc_G_nc_alpha(const Table& ngc, double alpha);

}  // namespace cmi

#endif  // CMI_CORE_H
