#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_sve.h>
#endif
#include <gtest/gtest.h>
#include <omp.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

// ----------------- Template traits for signed/unsigned cases -----------------
template <typename FloatT, typename IntT> struct FloatToIntTraits;
template <> struct FloatToIntTraits<float, int32_t> {
  using FloatType = float;
  using IntType = int32_t;
  static const int bit_width = 32;
  static_assert(sizeof(FloatType) == sizeof(IntType));

#ifdef __aarch64__
  static IntType ref(float x) {
    if (std::isinf(x)) {
      return (std::signbit(x)) ? std::numeric_limits<IntType>::min()
                               : std::numeric_limits<IntType>::max();
    } else if (std::isnan(x)) {
      return 0;
    }

    float t = std::trunc(x); // toward zero
    if (t > static_cast<float>(std::numeric_limits<IntType>::max())) {
      return std::numeric_limits<IntType>::max();
    }
    if (t < static_cast<float>(std::numeric_limits<IntType>::min())) {
      return std::numeric_limits<IntType>::min();
    }
    return static_cast<IntType>(t);
  }

  static IntType neon(FloatType x) {
    float32x4_t vin = vdupq_n_f32(x);
    int32x4_t vout = vcvtq_s32_f32(vin);
    return vgetq_lane_s32(vout, 0);
  }

#ifdef __ARM_FEATURE_SVE
  static IntType sve(FloatType x) {
    svfloat32_t vin = svdup_f32(x);
    svbool_t pg = svptrue_b32();
    svint32_t vout = svcvt_s32_f32_x(pg, vin);
    return svlastb_s32(pg, vout);
  }
#endif
#endif

  static const char *name() { return "fcvtzs"; }
};

// Specialization for unsigned 32-bit int (models FCVTZU: f32 -> u32).
template <> struct FloatToIntTraits<float, uint32_t> {
  using FloatType = float;
  using IntType = uint32_t;
  static const int bit_width = 32;
  static_assert(sizeof(FloatType) == sizeof(IntType));

#ifdef __aarch64__
  static IntType ref(float x) {
    // Deterministic model for FCVTZU: round toward zero + clamp, with
    // negative / -inf inputs mapped to 0.
    if (std::isinf(x)) {
      return (std::signbit(x)) ? 0u : std::numeric_limits<IntType>::max();
    } else if (std::isnan(x)) {
      return 0;
    }
    if (x <= 0.0f) {
      return 0u;
    }
    float t = std::trunc(x); // toward zero
    if (t > static_cast<float>(std::numeric_limits<IntType>::max())) {
      return std::numeric_limits<IntType>::max();
    }
    return static_cast<IntType>(t);
  }

  static IntType neon(FloatType x) {
    float32x4_t vin = vdupq_n_f32(x);
    uint32x4_t vout = vcvtq_u32_f32(vin);
    return vgetq_lane_u32(vout, 0);
  }

#ifdef __ARM_FEATURE_SVE
  static IntType sve(FloatType x) {
    svfloat32_t vin = svdup_f32(x);
    svbool_t pg = svptrue_b32();
    svuint32_t vout = svcvt_u32_f32_x(pg, vin);
    return svlastb_u32(pg, vout);
  }
#endif
#endif

  static const char *name() { return "fcvtzu"; }
};

// ----------------- Shared full-range test template -----------------
template <typename Traits, typename Convert> void RunFullRangeTest(Convert Cv) {
  using IntType = typename Traits::IntType;
  using FloatType = typename Traits::FloatType;
  static_assert(Traits::bit_width <= 32);
  std::atomic<uint64_t> mismatch_count{0};
  std::atomic<IntType> first_mismatch_pattern{0};
  std::atomic<bool> have_first_mismatch{false};
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < (1LL << Traits::bit_width); ++i) {
    IntType bits = static_cast<IntType>(i);
    FloatType x = std::bit_cast<FloatType>(bits);
    IntType neon_result = Cv(x);
    IntType ref_result = Traits::ref(x);
    if (neon_result != ref_result) {
      mismatch_count.fetch_add(1, std::memory_order_relaxed);
      bool expected_false = false;
      if (have_first_mismatch.compare_exchange_strong(
              expected_false, true, std::memory_order_relaxed)) {
        first_mismatch_pattern.store(bits, std::memory_order_relaxed);
      }
    }
  }
  uint64_t mismatches = mismatch_count.load(std::memory_order_relaxed);
  if (mismatches != 0) {
    IntType pattern = first_mismatch_pattern.load(std::memory_order_relaxed);
    FloatType x = std::bit_cast<FloatType>(pattern);
    FAIL() << "Found " << mismatches << " mismatches for " << Traits::name()
           << ".\n"
           << "First mismatch at bit pattern 0x" << std::hex << pattern
           << std::dec << " (float value " << x << ")";
  }
}

// ----------------- Tests -----------------
TEST(neon, vcvtq_s32_f32) {
  RunFullRangeTest<FloatToIntTraits<float, int32_t>>(
      FloatToIntTraits<float, int32_t>::neon);
}
TEST(neon, vcvtq_u32_f32) {
  RunFullRangeTest<FloatToIntTraits<float, uint32_t>>(
      FloatToIntTraits<float, uint32_t>::neon);
}

#ifdef __ARM_FEATURE_SVE
TEST(sve, vcvtq_s32_f32) {
  RunFullRangeTest<FloatToIntTraits<float, int32_t>>(
      FloatToIntTraits<float, int32_t>::sve);
}
TEST(sve, vcvtq_u32_f32) {
  RunFullRangeTest<FloatToIntTraits<float, uint32_t>>(
      FloatToIntTraits<float, uint32_t>::sve);
}
#endif

// main can be omitted if you link with gtest_main.
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
