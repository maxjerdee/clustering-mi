/* Core mutual-information routines for clustering-mi. See cmi_core.h. */

#include "cmi_core.h"

#include <algorithm>
#include <cmath>

#include "helpers.h"

namespace cmi {

namespace {

const double LOG2 = std::log(2.0);

std::vector<long> row_sums(const Table& T) {
    std::vector<long> ng(T.size(), 0);
    for (std::size_t r = 0; r < T.size(); ++r) {
        for (long v : T[r]) ng[r] += v;
    }
    return ng;
}

std::vector<long> col_sums(const Table& T) {
    const std::size_t ncols = T.empty() ? 0 : T[0].size();
    std::vector<long> nc(ncols, 0);
    for (const auto& row : T) {
        for (std::size_t s = 0; s < ncols; ++s) nc[s] += row[s];
    }
    return nc;
}

long total(const std::vector<long>& v) {
    long n = 0;
    for (long x : v) n += x;
    return n;
}

}  // namespace

double stirling_mutual_information(const Table& T) {
    const std::vector<long> ng = row_sums(T);
    const std::vector<long> nc = col_sums(T);
    const double n = static_cast<double>(total(ng));

    double MI = 0.0;
    for (std::size_t r = 0; r < ng.size(); ++r) {
        for (std::size_t s = 0; s < nc.size(); ++s) {
            if (T[r][s] > 0) {
                MI += static_cast<double>(T[r][s]) *
                      std::log(n * static_cast<double>(T[r][s]) /
                               (static_cast<double>(ng[r]) * static_cast<double>(nc[s])));
            }
        }
    }
    return MI / LOG2;
}

double traditional_mutual_information(const Table& T) {
    const std::vector<long> ng = row_sums(T);
    const std::vector<long> nc = col_sums(T);
    const double n = static_cast<double>(total(ng));

    double MI = log_factorial(n);
    for (long v : ng) MI -= log_factorial(static_cast<double>(v));
    for (long v : nc) MI -= log_factorial(static_cast<double>(v));
    for (const auto& row : T) {
        for (long v : row) MI += log_factorial(static_cast<double>(v));
    }
    return MI / LOG2;
}

double adjusted_mutual_information(const Table& T) {
    const std::vector<long> ng = row_sums(T);
    const std::vector<long> nc = col_sums(T);
    const long n = total(ng);
    const double nd = static_cast<double>(n);

    // Precompute log(k!) for every integer up to n. All binomials in the inner
    // sum take integer arguments, so each becomes three array lookups instead
    // of nine std::lgamma calls. logfact[k] == lgamma(k+1), so the result is
    // bit-identical to calling log_binom() in the loop.
    std::vector<double> logfact(static_cast<std::size_t>(n) + 1);
    for (long k = 0; k <= n; ++k) {
        logfact[static_cast<std::size_t>(k)] = std::lgamma(static_cast<double>(k) + 1.0);
    }
    auto lbinom = [&logfact](long a, long b) {
        return logfact[static_cast<std::size_t>(a)] - logfact[static_cast<std::size_t>(b)] -
               logfact[static_cast<std::size_t>(a - b)];
    };

    const double log_n = std::log(nd);

    double EMI = 0.0;
    for (std::size_t r = 0; r < ng.size(); ++r) {
        // Constant across the s and ngc loops for this row.
        const double log_ng_r = std::log(static_cast<double>(ng[r]));
        const double lb_n_ngr = lbinom(n, ng[r]);
        for (std::size_t s = 0; s < nc.size(); ++s) {
            // Constant across the ngc loop for this column.
            const double log_nc_s = std::log(static_cast<double>(nc[s]));
            const double base = log_n - log_ng_r - log_nc_s;
            const long lo = std::max(1L, ng[r] + nc[s] - n);
            const long hi = std::min(ng[r], nc[s]);
            for (long ngc = lo; ngc <= hi; ++ngc) {
                EMI += static_cast<double>(ngc) *
                       (base + std::log(static_cast<double>(ngc))) *
                       std::exp(lbinom(nc[s], ngc) +
                                lbinom(n - nc[s], ng[r] - ngc) - lb_n_ngr);
            }
        }
    }
    EMI /= LOG2;
    return stirling_mutual_information(T) - EMI;
}

double reduced_flat_mutual_information(const Table& T) {
    const std::vector<long> ng = row_sums(T);
    const std::vector<long> nc = col_sums(T);
    // Matches Python: _log_Omega_EC(nc, ng) with default flags.
    const double logOmega = log_Omega_EC(nc, ng);
    return traditional_mutual_information(T) - logOmega;
}

double H_ng_G_alpha(const std::vector<long>& ng, double alpha) {
    const double n = static_cast<double>(total(ng));
    const double q = static_cast<double>(ng.size());

    double H = log_binom(n + q * alpha - 1.0, q * alpha - 1.0);
    for (long v : ng) {
        H -= log_binom(static_cast<double>(v) + alpha - 1.0, alpha - 1.0);
    }
    return H;
}

double H_ngc_G_nc_alpha(const Table& ngc, double alpha) {
    const double qg = static_cast<double>(ngc.size());
    const std::vector<long> nc = col_sums(ngc);
    const std::size_t qc = nc.size();

    double H = 0.0;
    for (std::size_t s = 0; s < qc; ++s) {
        H += log_binom(static_cast<double>(nc[s]) + qg * alpha - 1.0, qg * alpha - 1.0);
        for (std::size_t r = 0; r < ngc.size(); ++r) {
            // A zero cell contributes log_binom(alpha-1, alpha-1) == 0 exactly,
            // so skip it. This is a no-op for dense tables but a large win for
            // sparse/diagonal ones (e.g. the self-tables used in normalisation).
            if (ngc[r][s] != 0) {
                H -= log_binom(static_cast<double>(ngc[r][s]) + alpha - 1.0, alpha - 1.0);
            }
        }
    }
    return H;
}

double reduced_mutual_information(const Table& T) {
    const std::vector<long> ng = row_sums(T);
    const std::vector<long> nc = col_sums(T);
    const double n = static_cast<double>(total(ng));

    const double min_alpha = 0.0001;
    const double max_alpha = 10000.0;

    const double H_qg = std::log(n);

    // H_g
    const double H_ng_alpha =
        minimize_golden_section_log(
            [&ng](double alpha) { return H_ng_G_alpha(ng, alpha); }, min_alpha,
            max_alpha)
            .second;
    double H_g_G_ng = log_factorial(n);
    for (long v : ng) H_g_G_ng -= log_factorial(static_cast<double>(v));
    const double H_g = H_qg + H_ng_alpha + H_g_G_ng;

    // H_g_G_c
    const double H_ngc_alpha =
        minimize_golden_section_log(
            [&T](double alpha) { return H_ngc_G_nc_alpha(T, alpha); }, min_alpha,
            max_alpha)
            .second;
    double H_g_G_c_ngc = 0.0;
    for (long v : nc) H_g_G_c_ngc += log_factorial(static_cast<double>(v));
    for (const auto& row : T) {
        for (long v : row) H_g_G_c_ngc -= log_factorial(static_cast<double>(v));
    }
    const double H_g_G_c = H_qg + H_ngc_alpha + H_g_G_c_ngc;

    return (H_g - H_g_G_c) / LOG2;
}

}  // namespace cmi
