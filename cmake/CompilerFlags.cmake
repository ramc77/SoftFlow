# Compiler flags for SoftFlow
# Auto-detects Intel vs Apple Silicon and optimizes accordingly.
#
# Every flag is added via softflow_add_flag(), which both calls
# add_compile_options() AND appends to SOFTFLOW_CXX_FLAGS. The latter is
# read at configure time by build_info.h.in so the run_manifest.json
# carries the exact set of flags used to compile libsoftflow.
#
# Note on -ffast-math: it allows the compiler to assume IEEE behavior
# is irrelevant, which can defeat std::isnan checks in checkStability().
# Phase 1 keeps it for performance; a follow-up should provide a
# -DSOFTFLOW_DETERMINISTIC=ON build that drops -ffast-math.

set(SOFTFLOW_CXX_FLAGS "" CACHE INTERNAL "Resolved compile flags for run_manifest.json")

function(softflow_add_flag)
    add_compile_options(${ARGN})
    foreach(flag IN LISTS ARGN)
        set(SOFTFLOW_CXX_FLAGS "${SOFTFLOW_CXX_FLAGS} ${flag}" CACHE INTERNAL "")
    endforeach()
endfunction()

# Detect Apple Silicon vs Intel
if(APPLE)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(ARCH STREQUAL "arm64")
        set(SOFTFLOW_APPLE_SILICON TRUE)
        message(STATUS "  Platform: Apple Silicon (${ARCH}) — using NEON SIMD")
    else()
        set(SOFTFLOW_APPLE_SILICON FALSE)
        message(STATUS "  Platform: Intel Mac (${ARCH}) — using AVX2/SSE SIMD")
    endif()
else()
    set(SOFTFLOW_APPLE_SILICON FALSE)
    message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME} (${CMAKE_SYSTEM_PROCESSOR})")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    softflow_add_flag(-Wall -Wextra -Wpedantic)

    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "")
        softflow_add_flag(-O3 -DNDEBUG)

        if(SOFTFLOW_APPLE_SILICON)
            # Apple Silicon: -mcpu=apple-m1 covers M1/M2/M3/M4 NEON+AMX
            softflow_add_flag(-mcpu=apple-m1 -ffast-math)
        else()
            # Intel: use native arch for best AVX2/SSE
            softflow_add_flag(-march=native -ffast-math)
        endif()
    else()
        softflow_add_flag(-O2)
    endif()

elseif(MSVC)
    softflow_add_flag(/W3 /O2 /fp:fast)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        softflow_add_flag(/Ox /GL /arch:AVX2)
    endif()
endif()
