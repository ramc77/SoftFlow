cmake_minimum_required(VERSION 3.18)
project(SoftFlow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(USE_OPENMP "Enable OpenMP parallelization" ON)

# OpenMP
if(USE_OPENMP)
    # On macOS with Apple Clang, libomp from Homebrew needs hints
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(LIBOMP_PREFIX)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib" CACHE STRING "" FORCE)
        endif()
    endif()
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP not found, building without parallelization")
        set(USE_OPENMP OFF)
    endif()
endif()

# Compiler flags
include(cmake/CompilerFlags.cmake)

# ── Build provenance for run_manifest.json ─────────────────────
# Capture git SHA, branch, dirty flag, compiler ID/version, and the
# resolved CXX flags into a generated header. Included by the engine
# at runtime so every output directory carries a complete fingerprint.
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SOFTFLOW_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SOFTFLOW_GIT_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SOFTFLOW_GIT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(SOFTFLOW_GIT_STATUS STREQUAL "")
        set(SOFTFLOW_GIT_DIRTY false)
    else()
        set(SOFTFLOW_GIT_DIRTY true)
    endif()
else()
    set(SOFTFLOW_GIT_SHA    "unknown")
    set(SOFTFLOW_GIT_BRANCH "unknown")
    set(SOFTFLOW_GIT_DIRTY  false)
endif()

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    set(SOFTFLOW_OPENMP_BOOL true)
else()
    set(SOFTFLOW_OPENMP_BOOL false)
endif()

# SOFTFLOW_CXX_FLAGS is populated by softflow_add_flag() in
# cmake/CompilerFlags.cmake. The base CXX flags from CMake itself are
# prepended so that anything the user passed via CXXFLAGS env / -D is
# also captured.
set(SOFTFLOW_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}${SOFTFLOW_CXX_FLAGS}")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/build_info.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/softflow/build_info.h
    @ONLY)

# Core library
add_subdirectory(src)

# Python bindings
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Example executables (C++ examples, if present)
option(BUILD_EXAMPLES "Build example simulations" OFF)

# Tests
if(BUILD_TESTS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
        enable_testing()
        add_subdirectory(tests)
    else()
        message(WARNING "BUILD_TESTS=ON but tests/CMakeLists.txt is missing; skipping.")
    endif()
endif()

message(STATUS "SoftFlow v${PROJECT_VERSION}")
message(STATUS "  OpenMP:  ${USE_OPENMP}")
message(STATUS "  Python:  ${BUILD_PYTHON}")
message(STATUS "  Tests:   ${BUILD_TESTS}")
