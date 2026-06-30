# Maintaining clustering-mi

`clustering-mi` is a small Python package with a compiled C++ core. The Python
layer handles input parsing, dispatch, and normalization; the numerically heavy
mutual-information routines live in C++ and are exposed through a thin pybind11
module. This document covers the layout and the build / test / release flow.

## Layout

```
cpp/                         language-agnostic C++ core (no pybind / Python deps)
  helpers.{h,cpp}            log-factorial / log-binom / log_Omega_EC / golden-section
  cmi_core.{h,cpp}           the 5 MI variations + entropy helpers (operate on a table)
bindings/python/bindings.cpp pybind11 glue -> PYBIND11_MODULE(_core, ...)
src/clustering_mi/           Python package
  _input_output.py           parse inputs into an integer contingency table
  mutual_information.py       public API; dispatch + normalization, calls _core
  _util.py                    pure-Python helpers (also unit-tested directly)
  _core.pyi                   type stub for the compiled extension
CMakeLists.txt               builds cmi_core (static) + _core (pybind module)
tests/                       Python tests (parity oracle) + tests/cpp (Catch2)
```

The C++ core is deliberately free of any pybind11/Python include so the same
code can compile into native tests today and other language bindings later. The
binding layer force-casts the numpy table to a contiguous int64 array and
releases the GIL around the computation.

## Build system

Backend: **scikit-build-core** + **CMake** + **pybind11**; versioning via
**setuptools_scm** (from git tags). The build is configured in `pyproject.toml`
(`[tool.scikit-build]`, `[tool.setuptools_scm]`) and `CMakeLists.txt`.

A wheel is **not** numpy-version-locked: pybind11 does not compile against
numpy's C-API, so one wheel works on both numpy 1.x and 2.x. The declared floor
is `numpy >= 1.21`.

## Development loop

```bash
pip install -e . --no-build-isolation      # builds the _core extension
pytest                                       # run the Python suite
```

- Editing **Python** files (`src/clustering_mi/*.py`) is live under the editable
  install — no rebuild needed.
- Editing **C++** files (`cpp/*`, `bindings/*`) requires re-running
  `pip install -e .` for the change to take effect.
- If an editable install ever appears to load stale code, fully remove the
  installed package from site-packages and reinstall (a leftover non-redirect
  copy can shadow the `src/` redirect).

The Python test suite is the numerical-parity oracle: the C++ results must match
the reference values (computed in Mathematica, see `tests/Tests.nb`) to within
the existing `pytest.approx` tolerances.

## Native C++ tests (optional)

```bash
cmake -S . -B build-test -DBUILD_TESTING=ON
cmake --build build-test
ctest --test-dir build-test
```

This builds `tests/cpp/cmi_core_test.cpp` (Catch2, fetched automatically) and
checks the core directly, independent of Python.

## Release

Wheels are built by **cibuildwheel** across linux / macOS / Windows in
`.github/workflows/release.yml`, triggered on a published GitHub Release. The
build set is `cp39`–`cp313` (matching `requires-python >= 3.9`); macOS wheels are
arm64 only (add `macos-13` to the matrix for Intel-mac wheels).

To cut a release:

```bash
git tag vX.Y.Z
git push origin master --tags
```

then publish the GitHub Release for that tag. Because the C++ rewrite is a major
change, bump the minor/major version accordingly rather than a patch.
