/* Native Catch2 tests for the clustering-mi C++ core.
 *
 * Build with -DBUILD_TESTING=ON and run via ctest. Expected values are the
 * Mathematica reference figures used by the Python tests (tests/Tests.nb).
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cmi_core.h"
#include "helpers.h"

using Catch::Approx;

namespace {
// Contingency table for c = [1,1,1,2,2,2,2,2], g = [1,1,2,2,2,3,3,3].
const cmi::Table kT = {{2, 0}, {1, 2}, {0, 3}};
}  // namespace

TEST_CASE("stirling matches Mathematica", "[mi]") {
    REQUIRE(cmi::stirling_mutual_information(kT) == Approx(4.88058).epsilon(1e-5));
}

TEST_CASE("traditional matches Mathematica", "[mi]") {
    REQUIRE(cmi::traditional_mutual_information(kT) == Approx(4.22239).epsilon(1e-5));
}

TEST_CASE("adjusted matches Mathematica", "[mi]") {
    REQUIRE(cmi::adjusted_mutual_information(kT) == Approx(2.74404).epsilon(1e-5));
}

TEST_CASE("reduced_flat matches Mathematica", "[mi]") {
    REQUIRE(cmi::reduced_flat_mutual_information(kT) == Approx(1.07763).epsilon(1e-5));
}

TEST_CASE("reduced matches Mathematica", "[mi]") {
    REQUIRE(cmi::reduced_mutual_information(kT) == Approx(0.160127).epsilon(1e-5));
}

TEST_CASE("entropy helpers match Mathematica", "[entropy]") {
    REQUIRE(cmi::H_ng_G_alpha({30, 20}, 0.5) == Approx(4.35361).epsilon(1e-5));
    const cmi::Table ngc = {{1, 0, 3}, {2, 2, 0}};
    REQUIRE(cmi::H_ngc_G_nc_alpha(ngc, 0.5) == Approx(3.81796).epsilon(1e-5));
}

TEST_CASE("log_Omega_EC matches Mathematica", "[helpers]") {
    REQUIRE(cmi::log_Omega_EC({1, 3, 4, 4}, {5, 2, 2, 3}) == Approx(9.4314).epsilon(1e-4));
}
