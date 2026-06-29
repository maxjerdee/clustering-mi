/* Helper math routines for clustering-mi.
 *
 * Language-agnostic: this header and helpers.cpp contain no pybind11 (or any
 * Python) dependency, so the same code compiles into the Python extension, the
 * native Catch2 tests, and (in future) other language bindings.
 *
 * All routines mirror the reference Python implementation in
 * src/clustering_mi/_util.py so that results are numerically identical.
 */

#ifndef CMI_HELPERS_H
#define CMI_HELPERS_H

#include <functional>
#include <utility>
#include <vector>

namespace cmi {

// log(n!) = lgamma(n + 1). Argument is a double so non-integer values work too.
double log_factorial(double n);

// log(binomial(n, m)) = log(n!) - log(m!) - log((n - m)!), all base e.
double log_binom(double n, double m);

// EC estimate of log(number of contingency tables) with the given row sums `rs`
// and column sums `cs` (Jerdee, Kirkley, Newman 2022, arXiv:2209.14869).
// Returns the estimate in BASE 2 (matching the Python _log_Omega_EC). Zero
// entries are dropped first. Returns -inf when either list is empty.
double log_Omega_EC(const std::vector<long>& rs, const std::vector<long>& cs,
                    bool useShortDimension = false, bool symmetrize = false);

// Golden-section minimisation in the logarithmic domain (mirrors
// _minimize_golden_section_log). Returns {argmin, f(argmin)}.
std::pair<double, double> minimize_golden_section_log(
    const std::function<double(double)>& f, double min_val, double max_val,
    double tol = 1e-5);

}  // namespace cmi

#endif  // CMI_HELPERS_H
