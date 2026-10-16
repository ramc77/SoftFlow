# SoftFlow test suite
#
# Layout:
#   tests/unit/         — focused tests for a single module (no full sim)
#   tests/verification/ — analytical-benchmark regression tests
#   tests/validation/   — comparison vs published numerical/experimental data

include(FetchContent)

FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
)
FetchContent_MakeAvailable(doctest)

# All tests link to the SoftFlow core library (target name: `softflow`,
# defined in src/CMakeLists.txt) and to doctest. Each test is a standalone
# executable with its own `main`, provided by doctest's
# DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN macro.
function(softflow_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE softflow doctest::doctest)
    target_compile_features(${name} PRIVATE cxx_std_20)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# A single sanity test today; the V&V cases of the project specification land in
# tests/verification/ in a follow-up commit. Each subdirectory is added
# only if it carries its own CMakeLists.txt, so empty placeholders don't
# break configure.
foreach(sub IN ITEMS unit verification validation)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${sub}/CMakeLists.txt)
        add_subdirectory(${sub})
    endif()
endforeach()
