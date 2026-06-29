/* pybind11 bindings for clustering-mi.
 *
 * Thin glue layer: converts numpy contingency tables into the C++ core's
 * std::vector<std::vector<long>> representation and dispatches to the
 * language-agnostic routines in cpp/cmi_core.{h,cpp}. The heavy loops run in
 * C++ with the GIL released.
 */

#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cmi_core.h"

namespace py = pybind11;

namespace {

// Force-cast an arbitrary array-like into a contiguous int64 array. This copies
// when needed (e.g. transposed views or int32 tables) so strides are trivial.
using IntArray =
    py::array_t<long long, py::array::c_style | py::array::forcecast>;

cmi::Table to_table(const IntArray& arr) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument("contingency table must be a 2-D array");
    }
    const auto rows = static_cast<std::size_t>(arr.shape(0));
    const auto cols = static_cast<std::size_t>(arr.shape(1));
    auto r = arr.unchecked<2>();

    cmi::Table T(rows, std::vector<long>(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            T[i][j] = static_cast<long>(r(i, j));
        }
    }
    return T;
}

std::vector<long> to_vector(const IntArray& arr) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument("expected a 1-D array");
    }
    const auto n = static_cast<std::size_t>(arr.shape(0));
    auto r = arr.unchecked<1>();
    std::vector<long> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<long>(r(i));
    return v;
}

// Wrap a Table-consuming core function: convert input, release the GIL, run.
template <typename Fn>
double run(Fn&& fn, const IntArray& arr) {
    cmi::Table T = to_table(arr);
    py::gil_scoped_release release;
    return fn(T);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "clustering-mi native core (thin wrapper over C++)";

    m.def(
        "stirling_mutual_information",
        [](const IntArray& T) { return run(&cmi::stirling_mutual_information, T); },
        py::arg("contingency_table"));
    m.def(
        "traditional_mutual_information",
        [](const IntArray& T) { return run(&cmi::traditional_mutual_information, T); },
        py::arg("contingency_table"));
    m.def(
        "adjusted_mutual_information",
        [](const IntArray& T) { return run(&cmi::adjusted_mutual_information, T); },
        py::arg("contingency_table"));
    m.def(
        "reduced_flat_mutual_information",
        [](const IntArray& T) { return run(&cmi::reduced_flat_mutual_information, T); },
        py::arg("contingency_table"));
    m.def(
        "reduced_mutual_information",
        [](const IntArray& T) { return run(&cmi::reduced_mutual_information, T); },
        py::arg("contingency_table"));

    m.def(
        "H_ng_G_alpha",
        [](const IntArray& ng, double alpha) {
            std::vector<long> v = to_vector(ng);
            py::gil_scoped_release release;
            return cmi::H_ng_G_alpha(v, alpha);
        },
        py::arg("ng"), py::arg("alpha"));
    m.def(
        "H_ngc_G_nc_alpha",
        [](const IntArray& ngc, double alpha) {
            cmi::Table T = to_table(ngc);
            py::gil_scoped_release release;
            return cmi::H_ngc_G_nc_alpha(T, alpha);
        },
        py::arg("ngc"), py::arg("alpha"));
}
