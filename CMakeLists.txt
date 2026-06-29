cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

option(BUILD_TESTING "Build Catch2-based native tests" OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Language-agnostic numeric core, built as a reusable static library so other
# language bindings can link against it without pulling in pybind11.
add_library(cmi_core STATIC
    cpp/cmi_core.cpp
    cpp/helpers.cpp)
target_include_directories(cmi_core PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
set_target_properties(cmi_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(cmi_core PRIVATE $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_compile_features(cmi_core PUBLIC cxx_std_17)

# Python bindings: a small pybind11 module linked against the core.
pybind11_add_module(_core MODULE bindings/python/bindings.cpp)
target_link_libraries(_core PRIVATE cmi_core)
target_compile_features(_core PRIVATE cxx_std_17)
# Drop the built extension straight into the package source tree so editable
# installs and in-tree test runs can import it.
set_target_properties(
  _core PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                   ${CMAKE_SOURCE_DIR}/src/clustering_mi)
install(TARGETS _core LIBRARY DESTINATION clustering_mi)

if(BUILD_TESTING)
  enable_testing()

  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.7.1
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  endif()

  add_executable(cmi_core_test
    tests/cpp/cmi_core_test.cpp
    cpp/cmi_core.cpp
    cpp/helpers.cpp
  )
  target_include_directories(cmi_core_test PRIVATE cpp)
  target_compile_features(cmi_core_test PRIVATE cxx_std_17)
  target_compile_options(cmi_core_test PRIVATE -Wall -Wextra)
  target_link_libraries(cmi_core_test PRIVATE Catch2::Catch2WithMain)

  include(CTest)
  add_test(NAME cmi_core_test COMMAND cmi_core_test)
endif()
