cmake_minimum_required(VERSION 3.14)
project(NeonVcvtqTests LANGUAGES CXX)

# User options
option(ENABLE_OPENMP "Enable OpenMP for full-range tests" ON)

# Require aarch64 with NEON; you can relax this if you want cross-compiles.
if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    message(FATAL_ERROR "This project is intended for AArch64/ARM64 with NEON.")
endif()

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Release, Debug, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable -march for NEON where applicable (GCC/Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-march=armv8-a+simd)
endif()

# ----------------- GoogleTest -----------------
#
# Prefer a system-installed GTest. If not found, you can:
#   - add_subdirectory of a vendored GTest,
#   - or use FetchContent.
#
# Here we first try system GTest. If not found, we optionally download it.

include(CMakeFindDependencyMacro)

find_package(GTest QUIET)

if (NOT GTest_FOUND)
    message(STATUS "GTest not found via find_package; trying FetchContent...")

    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )

    # For MSVC: avoid overriding runtime library settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    # After this, GTest::gtest and GTest::gtest_main targets are available
    set(GTEST_BOTH_LIBRARIES GTest::gtest GTest::gtest_main)
    set(GTEST_INCLUDE_DIRS "")
else()
    message(STATUS "Found system GTest")
    set(GTEST_BOTH_LIBRARIES GTest::GTest GTest::Main)
endif()

# ----------------- OpenMP -----------------

if (ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if (OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP")
        set(HAVE_OPENMP TRUE)
    else()
        message(WARNING "OpenMP requested but not found; building without OpenMP")
        set(HAVE_OPENMP FALSE)
    endif()
else()
    set(HAVE_OPENMP FALSE)
endif()

# ----------------- Test executable -----------------

# Add your test source file here
add_executable(simdc
    main.cpp
)

target_include_directories(simdc PRIVATE
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(simdc PRIVATE
    ${GTEST_BOTH_LIBRARIES}
    pthread
)

if (HAVE_OPENMP)
    target_compile_definitions(simdc PRIVATE HAVE_OPENMP=1)
    target_link_libraries(simdc PRIVATE OpenMP::OpenMP_CXX)
endif()

# Some compilers need explicit NEON header path; usually not necessary.
# You can add include dirs here if needed:
# target_include_directories(simdc PRIVATE /path/to/arm/neon/includes)

# ----------------- CTest integration -----------------

include(CTest)
if (BUILD_TESTING)
    add_test(NAME vcvtq_full_range COMMAND simdc)
endif()
