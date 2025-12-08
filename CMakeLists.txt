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
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable -march for NEON where applicable (GCC/Clang)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-march=armv8-a+simd)
endif()

# ----------------- GoogleTest (required, no auto-fetch) -----------------

find_package(GTest REQUIRED)

message(STATUS "Found GoogleTest: ${GTest_DIR}")
# Common configurations use imported targets GTest::gtest and GTest::gtest_main.
# Some distros expose GTest::GTest / GTest::Main instead. Handle both.

if (TARGET GTest::gtest AND TARGET GTest::gtest_main)
    set(GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
elseif (TARGET GTest::GTest AND TARGET GTest::Main)
    set(GTEST_LIBRARIES GTest::GTest GTest::Main)
else()
    message(FATAL_ERROR
        "GTest was found but expected targets are missing. "
        "Available targets: ${GLOBAL_PROPERTY:TARGETS}")
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

add_executable(simdc
    main.cpp
)

target_link_libraries(simdc PRIVATE
    ${GTEST_LIBRARIES}
    pthread
)

if (HAVE_OPENMP)
    target_compile_definitions(simdc PRIVATE HAVE_OPENMP=1)
    target_link_libraries(simdc PRIVATE OpenMP::OpenMP_CXX)
endif()

# ----------------- CTest integration -----------------

include(CTest)
if (BUILD_TESTING)
    add_test(NAME vcvtq_full_range COMMAND simdc)
endif()
