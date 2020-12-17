/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _NNC_CORE_BACKEND_INTERPRETER_QUANTIZATION_HELPERS_
#define _NNC_CORE_BACKEND_INTERPRETER_QUANTIZATION_HELPERS_

#include <cmath>
#include <limits>

namespace mir_interpreter
{

inline void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift)
{
  if (double_multiplier == 0.)
  {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }

  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(round(q * (1ll << 31)));

  assert(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31))
  {
    q_fixed /= 2;
    ++*shift;
  }
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31)
  {
    *shift = 0;
    q_fixed = 0;
  }
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

inline void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                                int32_t *quantized_multiplier, int *left_shift)
{
  assert(double_multiplier < 1.0);
  assert(double_multiplier > 0.0);
  int shift;
  QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
  assert(shift <= 0);
  *left_shift = shift;
}

inline int32_t MaskIfNonZero(int32_t a)
{
  static const int32_t zero = 0;
  return a ? ~zero : zero;
}

inline int32_t MaskIfZero(int32_t a) { return MaskIfNonZero(!a); }

inline int32_t MaskIfLessThan(int32_t a, int32_t b) { return MaskIfNonZero(a < b); }

inline int32_t MaskIfGreaterThan(int32_t a, int32_t b) { return MaskIfNonZero(a > b); }

inline int32_t RoundingDivideByPOT(int32_t x, int exponent)
{
  assert(exponent >= 0);
  assert(exponent <= 31);
  const int32_t mask = (1ll << exponent) - 1;
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + (MaskIfLessThan(x, 0) & 1);
  return (x >> exponent) + (MaskIfGreaterThan(remainder, threshold) & 1);
}

inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b)
{
  bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  std::int64_t a_64(a);
  std::int64_t b_64(b);
  std::int64_t ab_64 = a_64 * b_64;
  std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  std::int32_t ab_x2_high32 = static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift)
{
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  return RoundingDivideByPOT(
    SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier), right_shift);
}

inline int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(int32_t x,
                                                              int32_t quantized_multiplier,
                                                              int left_shift)
{
  return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quantized_multiplier),
                             -left_shift);
}

} // namespace mir_interpreter

#endif //_NNC_CORE_BACKEND_INTERPRETER_QUANTIZATION_HELPERS_
