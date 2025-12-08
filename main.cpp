// File: test_vcvtq_omp_full_range.cpp
//
// Build on AArch64 with NEON, OpenMP, and GoogleTest:
//   g++ -std=c++17 -O2 -march=armv8-a+simd -fopenmp \
//       test_vcvtq_omp_full_range.cpp \
//       -lgtest -lgtest_main -pthread -o test_vcvtq
//
// This test walks *all* 2^32 float bit patterns in parallel using OpenMP,
// converts them with vcvtq_s32_f32 / vcvtq_u32_f32, and compares against
// scalar reference functions that model “round toward zero + clamping”.
//
// NOTE: This is extremely expensive (4+ billion iterations). It is meant
// as a stress/validation tool, not a quick unit test. Consider restricting
// the range in practice (e.g. by stepping by >1 or sharding across machines).

#include <arm_neon.h>
#include <gtest/gtest.h>
#include <omp.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <atomic>

// ----------------- Scalar reference implementations -----------------

// Deterministic model for FCVTZS (f32 -> s32).
static int32_t ref_vcvtq_s32_f32(float x) {
    if (std::isinf(x)) {
        return (std::signbit(x)) ? std::numeric_limits<int32_t>::min()
                                 : std::numeric_limits<int32_t>::max();
	return 0;
    } else if (std::isnan(x)) {
	return 0;
    }

    float t = std::trunc(x);  // round toward zero

    if (t > static_cast<float>(std::numeric_limits<int32_t>::max()))
        return std::numeric_limits<int32_t>::max();
    if (t < static_cast<float>(std::numeric_limits<int32_t>::min()))
        return std::numeric_limits<int32_t>::min();

    return static_cast<int32_t>(t);
}

// Deterministic model for FCVTZU (f32 -> u32).
static uint32_t ref_vcvtq_u32_f32(float x) {
    if (std::isinf(x)) {
        return (std::signbit(x)) ? 0u : std::numeric_limits<uint32_t>::max();
    } else if (std::isnan(x)) {
	return 0;
    }

    if (x <= 0.0f)
        return 0u;

    float t = std::trunc(x);  // round toward zero

    if (t > static_cast<float>(std::numeric_limits<uint32_t>::max()))
        return std::numeric_limits<uint32_t>::max();

    return static_cast<uint32_t>(t);
}

// ----------------- Tests -----------------

// Full-range test for vcvtq_s32_f32.
TEST(neon, vcvtq_s32_f32) {
    // Use atomics to record mismatches (to keep the test thread-safe).
    std::atomic<uint64_t> mismatch_count{0};
    std::atomic<uint32_t> first_mismatch_pattern{0};
    std::atomic<uint32_t> expected_result{0};
    std::atomic<uint32_t> actual_result{0};
    std::atomic<bool>     have_first_mismatch{false};

    // Iterate over all 2^32 bit patterns.
    // Parallelized with OpenMP.
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (1LL << 32); ++i) {
        uint32_t bits = static_cast<uint32_t>(i);
        float x = std::bit_cast<float>(bits);

        // Build a vector with x in lane 0.
        float32x4_t vin = vdupq_n_f32(0.0f);
        vin = vsetq_lane_f32(x, vin, 0);

        int32x4_t vout = vcvtq_s32_f32(vin);
        int32_t neon_result = vgetq_lane_s32(vout, 0);
        int32_t ref_result  = ref_vcvtq_s32_f32(x);

        if (neon_result != ref_result) {
            mismatch_count.fetch_add(1, std::memory_order_relaxed);
            bool expected_false = false;
            if (have_first_mismatch.compare_exchange_strong(expected_false, true,
                                                            std::memory_order_relaxed)) {
                first_mismatch_pattern.store(bits, std::memory_order_relaxed);
		expected_result.store(static_cast<uint32_t>(ref_result), std::memory_order_relaxed);
		actual_result.store(static_cast<uint32_t>(neon_result), std::memory_order_relaxed);
            }
        }
    }

    uint64_t mismatches = mismatch_count.load(std::memory_order_relaxed);
    if (mismatches != 0) {
        uint32_t pattern = first_mismatch_pattern.load(std::memory_order_relaxed);
        float x = std::bit_cast<float>(pattern);
        FAIL() << "Found " << mismatches
               << " mismatches for vcvtq_s32_f32.\n"
               << "First mismatch at bit pattern 0x"
               << std::hex << pattern << std::dec
               << " (float value " << x << ")\n"
	       << "Expected result: " << std::hex << expected_result.load(std::memory_order_relaxed) << "\n"
	       << "Actual result: " << std::hex << actual_result.load(std::memory_order_relaxed) << "\n";
    }
}

// Full-range test for vcvtq_u32_f32.
TEST(neon, vcvtq_u32_f32) {
    std::atomic<uint64_t> mismatch_count{0};
    std::atomic<uint32_t> first_mismatch_pattern{0};
    std::atomic<bool>     have_first_mismatch{false};

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (1LL << 32); ++i) {
        uint32_t bits = static_cast<uint32_t>(i);
        float x = std::bit_cast<float>(bits);

        float32x4_t vin = vdupq_n_f32(0.0f);
        vin = vsetq_lane_f32(x, vin, 0);

        uint32x4_t vout = vcvtq_u32_f32(vin);
        uint32_t neon_result = vgetq_lane_u32(vout, 0);
        uint32_t ref_result  = ref_vcvtq_u32_f32(x);

        if (neon_result != ref_result) {
            mismatch_count.fetch_add(1, std::memory_order_relaxed);
            bool expected_false = false;
            if (have_first_mismatch.compare_exchange_strong(expected_false, true,
                                                            std::memory_order_relaxed)) {
                first_mismatch_pattern.store(bits, std::memory_order_relaxed);
            }
        }
    }

    uint64_t mismatches = mismatch_count.load(std::memory_order_relaxed);
    if (mismatches != 0) {
        uint32_t pattern = first_mismatch_pattern.load(std::memory_order_relaxed);
        float x = std::bit_cast<float>(pattern);
        FAIL() << "Found " << mismatches
               << " mismatches for vcvtq_u32_f32.\n"
               << "First mismatch at bit pattern 0x"
               << std::hex << pattern << std::dec
               << " (float value " << x << ")";
    }
}

// main can be omitted if you link with gtest_main.
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
