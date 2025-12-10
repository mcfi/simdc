#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_sve.h>
#elif defined(__x86_64__)
#include <immintrin.h>
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

  static const char *name() { return "fcvtzs"; }

#elif defined(__x86_64__)

  static IntType ref(float x) {
    if (!std::isfinite(x)) {
      return std::numeric_limits<IntType>::min();
    }

    float t = std::trunc(x); // toward zero
    if (t > static_cast<float>(std::numeric_limits<IntType>::max())) {
      return std::numeric_limits<IntType>::min();
    }
    if (t < static_cast<float>(std::numeric_limits<IntType>::min())) {
      return std::numeric_limits<IntType>::min();
    }
    return static_cast<IntType>(t);
  }

  static IntType avx512(FloatType x) {
    __m512 v = _mm512_set1_ps(x);
    __m512i vi =
        _mm512_cvt_roundps_epi32(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(vi));
  }

  static const char *name() { return "vcvtps2dq"; }

#endif
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
    float t = std::trunc(x); // toward zero
    if (t > static_cast<float>(std::numeric_limits<IntType>::max())) {
      return std::numeric_limits<IntType>::max();
    }
    if (t < -0.0f) {
      return 0u;
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

  static const char *name() { return "fcvtzu"; }

#elif defined(__x86_64__)

  static IntType ref(FloatType x) {
    if (!std::isfinite(x)) {
      return std::numeric_limits<IntType>::max();
    }

    FloatType t = std::trunc(x); // toward zero
    if (t > static_cast<FloatType>(std::numeric_limits<IntType>::max())) {
      return std::numeric_limits<IntType>::max();
    }
    if (t < -0.0f) {
      return std::numeric_limits<IntType>::max();
    }
    return static_cast<IntType>(t);
  }

#ifdef __AVX512F__
  static IntType avx512(FloatType x) {
    __m512 v = _mm512_set1_ps(x);
    __m512i vi =
        _mm512_cvt_roundps_epu32(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(vi));
  }
#endif

  static const char *name() { return "vcvtps2udq"; }

#endif
};

// ----------------- Shared full-range test template -----------------
template <typename Traits, typename Convert> void RunFullRangeTest(Convert Cv) {
  using IntType = typename Traits::IntType;
  using FloatType = typename Traits::FloatType;
  static_assert(Traits::bit_width <= 32);
  std::atomic<IntType> first_mismatch_pattern{0};
  std::atomic<bool> have_first_mismatch{false};
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < (1LL << Traits::bit_width); ++i) {
    IntType bits = static_cast<IntType>(i);
    FloatType x = std::bit_cast<FloatType>(bits);
    IntType neon_result = Cv(x);
    IntType ref_result = Traits::ref(x);
    if (neon_result != ref_result) {
      bool expected_false = false;
      if (!have_first_mismatch.load(std::memory_order_relaxed)) {
        if (have_first_mismatch.compare_exchange_strong(
                expected_false, true, std::memory_order_relaxed)) {
          first_mismatch_pattern.store(bits, std::memory_order_relaxed);
        }
      }
    }
  }
  if (have_first_mismatch.load(std::memory_order_relaxed)) {
    IntType pattern = first_mismatch_pattern.load(std::memory_order_relaxed);
    FloatType x = std::bit_cast<FloatType>(pattern);
    FAIL() << "Found mismatches for " << Traits::name() << ".\n"
           << "First mismatch at bit pattern 0x" << std::hex << pattern
           << std::dec << " (float value " << x << ")";
  }
}

// ----------------- Tests -----------------
#ifdef __aarch64__
TEST(neon, vcvtq_s32_f32) {
  RunFullRangeTest<FloatToIntTraits<float, int32_t>>(
      FloatToIntTraits<float, int32_t>::neon);
}
TEST(neon, vcvtq_u32_f32) {
  RunFullRangeTest<FloatToIntTraits<float, uint32_t>>(
      FloatToIntTraits<float, uint32_t>::neon);
}

// This test always fails because fp16 fma only does rounding once while fp32
// fma + fp16 conversion does rounding twice so the results could be different.
TEST(neon, fp16_fp32_fma) {
  std::atomic<uint16_t> first_a{0}, first_b{0}, first_c{0}, first_f16rs{0},
      first_f16_32rs{0};
  std::atomic<bool> have_first_mismatch{false};
#pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < (1LL << 34); ++i) {
    uint16_t a = (i & 0xffff);
    uint16_t b = ((i >> 16) & 0xffff);
    uint8_t c = ((i >> 32) & 0xff);
    float16x4_t f16a = vreinterpret_f16_u16(vdup_n_u16(a));
    float16x4_t f16b = vreinterpret_f16_u16(vdup_n_u16(b));
    float16x4_t f16c = vcvt_f16_u16(vdup_n_u16(c));
    float16x4_t f16r = vfma_f16(f16b, f16a, f16c);
    float16_t f16rs = vget_lane_f16(f16r, 0);

    float32x4_t f32a = vcvt_f32_f16(f16a);
    float32x4_t f32b = vcvt_f32_f16(f16b);
    float32x4_t f32c = vcvt_f32_f16(f16c);
    float32x4_t f32r = vfmaq_f32(f32b, f32a, f32c);
    float16x4_t f16_32r = vcvt_f16_f32(f32r);
    float16_t f16_32rs = vget_lane_f16(f16_32r, 0);

    if (std::bit_cast<uint16_t>(f16rs) != std::bit_cast<uint16_t>(f16_32rs)) {
      bool expected_false = false;
      if (!have_first_mismatch.load(std::memory_order_relaxed)) {
        if (have_first_mismatch.compare_exchange_strong(
                expected_false, true, std::memory_order_relaxed)) {
          first_a.store(a, std::memory_order_relaxed);
          first_b.store(b, std::memory_order_relaxed);
          first_c.store(c, std::memory_order_relaxed);
          first_f16rs.store(std::bit_cast<uint16_t>(f16rs),
                            std::memory_order_relaxed);
          first_f16_32rs.store(std::bit_cast<uint16_t>(f16_32rs),
                               std::memory_order_relaxed);
        }
      }
    }
  }
  if (have_first_mismatch.load(std::memory_order_relaxed)) {
    FAIL() << "Found mismatches for (" << std::hex << first_a.load() << ", "
           << first_b.load() << ", " << first_c.load() << ")\n"
           << first_f16rs.load() << ", " << first_f16_32rs.load() << "\n";
  }
}

#ifdef __ARM_FEATURE_SVE
TEST(sve, svcvt_s32_f32_x) {
  RunFullRangeTest<FloatToIntTraits<float, int32_t>>(
      FloatToIntTraits<float, int32_t>::sve);
}
TEST(sve, svcvt_u32_f32_x) {
  RunFullRangeTest<FloatToIntTraits<float, uint32_t>>(
      FloatToIntTraits<float, uint32_t>::sve);
}

#endif

#elif defined(__x86_64__)

#ifdef __AVX512F__
TEST(avx512, _mm512_cvt_roundps_epi32) {
  RunFullRangeTest<FloatToIntTraits<float, int32_t>>(
      FloatToIntTraits<float, int32_t>::avx512);
}
TEST(avx512, _mm512_cvt_roundps_epu32) {
  RunFullRangeTest<FloatToIntTraits<float, uint32_t>>(
      FloatToIntTraits<float, uint32_t>::avx512);
}
#endif

#endif

// main can be omitted if you link with gtest_main.
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
