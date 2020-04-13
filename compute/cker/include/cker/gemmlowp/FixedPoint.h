/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNFW_CKER_GEMMLOWP_FIXED_POINT_H__
#define __NNFW_CKER_GEMMLOWP_FIXED_POINT_H__

#include <algorithm>
#include <cassert>

namespace nnfw
{
namespace cker
{
namespace gemmlowp
{

inline int32_t RoundingHalfSum(int32_t a, int32_t b)
{
  int64_t a64 = a;
  int64_t b64 = b;
  int64_t sum = a64 + b64;
  int64_t sign = sum >= 0 ? 1 : -1;
  return static_cast<int32_t>((sum + sign) / 2);
}

inline int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b)
{
  bool overflow = a == b && a == std::numeric_limits<int32_t>::min();
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;
  int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<int32_t>::max() : ab_x2_high32;
}

// Correctly-rounded-to-nearest division by a power-of-two.
// Also known as a rounding arithmetic right shift.
inline int32_t RoundingDivideByPOT(int32_t x, int exponent)
{
  assert(exponent >= 0);
  assert(exponent <= 31);
  const int32_t mask = ((1ll << exponent) - 1);
  const int32_t zero = 0;
  const int32_t one = 1;
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + ((x < zero) ? one : zero);
  return ((x >> exponent) + ((remainder > threshold) ? one : zero));
}

// Returns the product of a run-time integer value by a compile-time power
// of two, with either a positive exponent (equivalent to an arithmetic
// left shift, saturating) or a negative exponent (equivalent to an arithmetic
// right shift, rounding to nearest).
template <int Exponent, int ExponentSign = (Exponent > 0 ? 1 : Exponent < 0 ? -1 : 0)>
struct ImplSaturatingRoundingMultiplyByPOT
{
};

template <int Exponent> struct ImplSaturatingRoundingMultiplyByPOT<Exponent, 0>
{
  static int32_t eval(int32_t x) { return x; }
};

template <int Exponent> struct ImplSaturatingRoundingMultiplyByPOT<Exponent, 1>
{
  static int32_t eval(int32_t x)
  {
    const int32_t min = (std::numeric_limits<int32_t>::min());
    const int32_t max = (std::numeric_limits<int32_t>::max());
    const int32_t threshold = ((1 << (31 - Exponent)) - 1);
    const int32_t zero = 0;
    const int32_t one = 1;

    const int32_t positive_mask = ((x > threshold) ? ~zero : zero);
    const int32_t negative_mask = ((x < -threshold) ? ~zero : zero);

    int32_t result = (x * (one << Exponent));
    result = (positive_mask ? max : result);
    result = (negative_mask ? min : result);
    return result;
  }
};

template <int Exponent> struct ImplSaturatingRoundingMultiplyByPOT<Exponent, -1>
{
  static int32_t eval(int32_t x) { return RoundingDivideByPOT(x, -Exponent); }
};

template <int Exponent> int32_t SaturatingRoundingMultiplyByPOT(int32_t x)
{
  return ImplSaturatingRoundingMultiplyByPOT<Exponent>::eval(x);
}

template <int tIntegerBits> class FixedPoint
{
public:
  static constexpr int kTotalBits = 8 * sizeof(int32_t);
  static constexpr int kIntegerBits = tIntegerBits;
  static constexpr int kFractionalBits = kTotalBits - 1 - kIntegerBits;
  static_assert(kIntegerBits >= 0 && kIntegerBits < kTotalBits, "bad IntegerBits");

  static int32_t ScalarRawMax() { return std::numeric_limits<int32_t>::max(); }

  static FixedPoint FromRaw(int32_t x)
  {
    FixedPoint retval;
    retval.raw() = x;
    return retval;
  }

  static FixedPoint FromScalarRaw(int32_t x) { return FromRaw(x); }

  template <int Exponent> static FixedPoint ConstantPOT()
  {
    static constexpr int kOffset = kFractionalBits + Exponent;
    static_assert(kOffset < 31, "Constant not exactly representable in this fixed-point format");
    return FromScalarRaw((int32_t)1 << kOffset);
  }

  static FixedPoint Zero() { return FromScalarRaw(0); }

  static FixedPoint One()
  {
    return FromScalarRaw(kIntegerBits == 0 ? ScalarRawMax() : ((int32_t)1 << kFractionalBits));
  }

  int32_t raw() const { return i_; }
  int32_t &raw() { return i_; }

private:
  int32_t i_;
};

// A FixedPoint multiplication is just a
// SaturatingRoundingDoublingHighMul operation on the underlying
// raw integer values. The IntegerBits simply add up, as is obvious
// from the fact that the range is [-2^IntegerBits, 2^IntegerBits).
template <int tIntegerBits_a, int tIntegerBits_b>
FixedPoint<tIntegerBits_a + tIntegerBits_b> operator*(FixedPoint<tIntegerBits_a> a,
                                                      FixedPoint<tIntegerBits_b> b)
{
  FixedPoint<tIntegerBits_a + tIntegerBits_b> c;
  c.raw() = SaturatingRoundingDoublingHighMul(a.raw(), b.raw());
  return c;
}

// Tweaking IntegerBits gives exact multiplication by a power of two.
template <int tExponent, int tIntegerBits>
FixedPoint<tExponent + tIntegerBits> ExactMulByPot(FixedPoint<tIntegerBits> a)
{
  FixedPoint<tExponent + tIntegerBits> c;
  c.raw() = a.raw();
  return c;
}

template <int tIntegerBits>
FixedPoint<tIntegerBits> operator+(FixedPoint<tIntegerBits> a, FixedPoint<tIntegerBits> b)
{
  return FixedPoint<tIntegerBits>::FromRaw((a.raw() + b.raw()));
}
template <int tIntegerBits>
FixedPoint<tIntegerBits> operator-(FixedPoint<tIntegerBits> a, FixedPoint<tIntegerBits> b)
{
  return FixedPoint<tIntegerBits>::FromRaw((a.raw() - b.raw()));
}
template <int tIntegerBits>
FixedPoint<tIntegerBits> operator&(FixedPoint<tIntegerBits> a, FixedPoint<tIntegerBits> b)
{
  return FixedPoint<tIntegerBits>::FromRaw((a.raw() & b.raw()));
}

// Rescale changes the number of IntegerBits and updates the underlying
// raw integer value accordingly.
template <int tIntegerBitsDst, int tIntegerBitsSrc>
FixedPoint<tIntegerBitsDst> Rescale(FixedPoint<tIntegerBitsSrc> x)
{
  static constexpr int kExponent = tIntegerBitsSrc - tIntegerBitsDst;
  FixedPoint<tIntegerBitsDst> result;
  result.raw() = SaturatingRoundingMultiplyByPOT<kExponent>(x.raw());
  return result;
}

// Implementation of exponential function.

// Returns exp(x) for x in [-1/4, 0).
inline FixedPoint<0> exp_on_interval_between_negative_one_quarter_and_0_excl(FixedPoint<0> a)
{
  typedef FixedPoint<0> F;
  const F constant_term = F::FromScalarRaw(RoundingDivideByPOT(1895147668, 0));
  const F constant_1_over_3 = F::FromScalarRaw(RoundingDivideByPOT(715827883, 0));
  // We're evaluating a Taylor expansion around -1/8, so we do the change of
  // variable: x = a + 1/8.
  // In fixed-point with 0 integer bits, 1/8 is represented by 1 << 28.
  F x = a + F::template ConstantPOT<-3>();
  F x2 = x * x;
  F x3 = x2 * x;
  F x4 = x2 * x2;
  F x4_over_4 = F::FromScalarRaw(SaturatingRoundingMultiplyByPOT<-2>(x4.raw()));
  F x4_over_24_plus_x3_over_6_plus_x2_over_2 = F::FromScalarRaw(
      SaturatingRoundingMultiplyByPOT<-1>((((x4_over_4 + x3) * constant_1_over_3) + x2).raw()));
  return (constant_term + constant_term * (x + x4_over_24_plus_x3_over_6_plus_x2_over_2));
}

// Returns exp(x) for x < 0.
template <int tIntegerBits> FixedPoint<0> exp_on_negative_values(FixedPoint<tIntegerBits> a)
{
  typedef FixedPoint<tIntegerBits> InputF;
  typedef FixedPoint<0> ResultF;
  static constexpr int kFractionalBits = InputF::kFractionalBits;
  static constexpr int kIntegerBits = InputF::kIntegerBits;
  const InputF kOneQuarter = InputF::template ConstantPOT<-2>();
  InputF mask = kOneQuarter - InputF::FromScalarRaw(1);
  InputF a_mod_quarter_minus_one_quarter = (a & mask) - kOneQuarter;
  ResultF result = exp_on_interval_between_negative_one_quarter_and_0_excl(
      Rescale<0>(a_mod_quarter_minus_one_quarter));
  int32_t remainder = (a_mod_quarter_minus_one_quarter - a).raw();

#define GEMMLOWP_EXP_BARREL_SHIFTER(Exponent, FixedPointMultiplier)                 \
  if (kIntegerBits > Exponent)                                                      \
  {                                                                                 \
    const ResultF kMultiplier =                                                     \
        ResultF::FromScalarRaw(RoundingDivideByPOT(FixedPointMultiplier, 0));       \
    static constexpr int kShiftAmount =                                             \
        ((kIntegerBits > Exponent) ? (kFractionalBits + Exponent) : 0);             \
    result = ((remainder & (1 << kShiftAmount)) ? (result * kMultiplier) : result); \
  }

  GEMMLOWP_EXP_BARREL_SHIFTER(-2, 1672461947);
  GEMMLOWP_EXP_BARREL_SHIFTER(-1, 1302514674);
  GEMMLOWP_EXP_BARREL_SHIFTER(+0, 790015084);
  GEMMLOWP_EXP_BARREL_SHIFTER(+1, 290630308);
  GEMMLOWP_EXP_BARREL_SHIFTER(+2, 39332535);
  GEMMLOWP_EXP_BARREL_SHIFTER(+3, 720401);
  GEMMLOWP_EXP_BARREL_SHIFTER(+4, 242);

#undef GEMMLOWP_EXP_BARREL_SHIFTER

  static constexpr int clampB = ((kIntegerBits > 5) ? (36 - kIntegerBits) : 0);
  if (kIntegerBits > 5)
  {
    const InputF clamp = InputF::FromScalarRaw(RoundingDivideByPOT(-(1 << clampB), 0));
    result.raw() = ((a.raw() < clamp.raw()) ? ResultF::Zero().raw() : result.raw());
  }

  result.raw() = (a.raw() ? result.raw() : ResultF::One().raw());
  return result;
}

// Returns 1 / (1 + x) for x in (0, 1).
inline FixedPoint<0> one_over_one_plus_x_for_x_in_0_1(FixedPoint<0> a)
{
  typedef FixedPoint<0> F0;
  typedef FixedPoint<2> F2;
  F0 half_denominator = F0::FromScalarRaw(RoundingHalfSum(a.raw(), F0::One().raw()));
  // Newton-Raphson division
  // https://en.wikipedia.org/wiki/Division_algorithm#Newton.E2.80.93Raphson_division
  // Refer to that page for the logic behind the 48/17 and 32/17 constants.
  const F2 constant_48_over_17 = F2::FromScalarRaw(RoundingDivideByPOT(1515870810, 0));
  const F2 constant_neg_32_over_17 = F2::FromScalarRaw(RoundingDivideByPOT(-1010580540, 0));
  F2 x = constant_48_over_17 + half_denominator * constant_neg_32_over_17;
  for (int i = 0; i < 3; i++)
  {
    F2 half_denominator_times_x = half_denominator * x;
    F2 one_minus_half_denominator_times_x = F2::One() - half_denominator_times_x;
    x = x + Rescale<2>(x * one_minus_half_denominator_times_x);
  }
  return Rescale<0>(ExactMulByPot<-1>(x));
}

} // namespace gemmlowp
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_GEMMLOWP_FIXED_POINT_H__
