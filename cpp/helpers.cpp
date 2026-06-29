/* Helper math routines for clustering-mi. See helpers.h. */

#include "helpers.h"

#include <cmath>
#include <limits>

namespace cmi {

double log_factorial(double n) { return std::lgamma(n + 1.0); }

double log_binom(double n, double m) {
    return log_factorial(n) - log_factorial(m) - log_factorial(n - m);
}

double log_Omega_EC(const std::vector<long>& rs_in, const std::vector<long>& cs_in,
                    bool useShortDimension, bool symmetrize) {
    // Drop zero entries (matches rs = rs[rs > 0] in the Python).
    std::vector<long> rs;
    std::vector<long> cs;
    rs.reserve(rs_in.size());
    cs.reserve(cs_in.size());
    for (long r : rs_in) {
        if (r > 0) rs.push_back(r);
    }
    for (long c : cs_in) {
        if (c > 0) cs.push_back(c);
    }

    if (rs.empty() || cs.empty()) {
        return -std::numeric_limits<double>::infinity();
    }

    // Performance of the EC estimate improves with more rows than columns;
    // optionally swap so that the longer dimension is the rows.
    if (useShortDimension) {
        if (rs.size() >= cs.size()) {
            return log_Omega_EC(rs, cs, /*useShortDimension=*/false, symmetrize);
        }
        return log_Omega_EC(cs, rs, /*useShortDimension=*/false, symmetrize);
    }

    if (symmetrize) {
        return (log_Omega_EC(rs, cs, false, false) +
                log_Omega_EC(cs, rs, false, false)) /
               2.0;
    }

    const double m = static_cast<double>(rs.size());
    double N = 0.0;
    for (long r : rs) N += static_cast<double>(r);

    // Exact result (equivalent to alpha = inf). Note: the reference Python
    // returns this branch in base e (no division by log 2); replicate exactly.
    if (static_cast<double>(cs.size()) == N) {
        double result = log_factorial(N + 1.0);
        for (long r : rs) result -= log_factorial(static_cast<double>(r) + 1.0);
        return result;
    }

    double sum_cs_sq = 0.0;
    for (long c : cs) sum_cs_sq += static_cast<double>(c) * static_cast<double>(c);

    const double alphaC =
        (N * N - N + (N * N - sum_cs_sq) / m) / (sum_cs_sq - N);

    double result = -log_binom(N + m * alphaC - 1.0, m * alphaC - 1.0);
    for (long r : rs) {
        result += log_binom(static_cast<double>(r) + alphaC - 1.0, alphaC - 1.0);
    }
    for (long c : cs) {
        result += log_binom(static_cast<double>(c) + m - 1.0, m - 1.0);
    }
    return result / std::log(2.0);  // Convert to base 2.
}

std::pair<double, double> minimize_golden_section_log(
    const std::function<double(double)>& f, double min_val, double max_val,
    double tol) {
    double a = std::log(min_val);
    double b = std::log(max_val);

    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double resphi = 2.0 - phi;

    double c = a + resphi * (b - a);
    double d = b - resphi * (b - a);

    // Retain the two interior evaluations across iterations: each step only
    // needs one fresh evaluation instead of two. The (a, b, c, d) trajectory is
    // identical to the naive two-eval version, so the result is unchanged.
    double fc = f(std::exp(c));
    double fd = f(std::exp(d));

    while (std::fabs(c - d) > tol) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = a + resphi * (b - a);
            fc = f(std::exp(c));
        } else {
            a = c;
            c = d;
            fc = fd;
            d = b - resphi * (b - a);
            fd = f(std::exp(d));
        }
    }

    const double x = std::exp((a + b) / 2.0);
    return {x, f(x)};
}

}  // namespace cmi
