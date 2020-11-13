/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_RUY_TYPES_H__
#define __NNFW_RUY_TYPES_H__

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <limits>
#include <string>
#include "Shape.h"

namespace nnfw
{
namespace ruy
{

enum class FusedActivationFunctionType
{
  kNone = 0,
  kRelu6 = 1,
  kRelu1 = 2,
  kRelu = 3,
  kTanh = 4,
  kSigmoid = 6,
};

enum class PaddingType
{
  kNone = 0,
  kSame = 1,
  kValid = 2,
};

struct PaddingValues
{
  int16_t width;
  int16_t height;
};

struct ConvParams
{
  PaddingType padding_type;
  PaddingValues padding_values;
  // TODO(starka): This was just "stride", so check that width+height is OK.
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  // uint8_t inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
  bool is_replaced_weights{false};
};

struct FullyConnectedParams
{
  FusedActivationFunctionType activation{FusedActivationFunctionType::kNone};
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  float weights_scale;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params - no one use this params, but ruy might use them later.
  // float float_activation_min;
  // float float_activation_max;
  // FullyConnectedWeightsFormat weights_format;
};

enum class Order
{
  kColMajor,
  kRowMajor
};

enum class CachePolicy : std::uint8_t
{
  kNeverCache,
  kCacheIfLargeSpeedup,
  kAlwaysCache,
};

// MatrixParams encapsulates the parameters that Gemm needs about each
// matrix, besides the buffer data pointer.
// Compare to ruy::Matrix, which also encapsulates the data pointer.
// Rationale for leaving the data pointer out of here: doing so
// requires complicated const-correctness mechanics. See
// ruy::ConstCheckingPtr.
template <typename Scalar> struct MatrixParams
{
  // Storage layout order. For now we only do plain linear non-strided
  // layout. It would be easy to support a stride if needed.
  Order order = Order::kColMajor;
  // Number of rows of the matrix.
  int rows = 0;
  // Number of columns of the matrix.
  int cols = 0;
  // The zero_point, i.e. which Scalar value is to be interpreted as zero.
  // When Scalar is floating-point, this must be 0.
  Scalar zero_point = 0;
  // When the data pointed to by this matrix is constant data, so that it is
  // valid to assume that equality of pointers implies equality of data,
  // a CachePolicy may be used instead of the default kNeverCache,
  // which will enable ruy to take advantage of this constancy of the data to
  // cache the packing work, which can be a large speedup in matrix*vector
  // and other narrow shapes.
  CachePolicy cache_policy = CachePolicy::kNeverCache;
};

// Enumeration of broad categories of Gemm.
//
// The primary reason for this to exist is to allow Gemm to compile
// only uniform-quantized or only per-channel-quantized code paths.
// This is unneeded with ruy as the back-end, as this is only a runtime
// difference in ruy, but with gemmlowp these really are separate code
// paths and templatizing in a QuantizationFlavor is necessary to avoid
// compiling unused gemmlowp code. Indeed, TFLite currently uses
// uint8 with uniform quantization and int8 with per-channel quantization,
// and does not use uint8 with per-channel. We want to avoid compiling
// the gemmlowp uint8 per-channel path when gemmlowp is the back-end.
//
// It's possible to drop this in the future if gemmlowp goes away and no
// other then-relevant backend library handles quantized paths in a way that
// requires knowing this at compile-time.
enum class QuantizationFlavor
{
  // Floating-point Gemm: the accumulators are not multiplied by any
  // 'multiplier'.
  kFloatingPoint,
  // Quantized Gemm using a single multiplier for all accumulators.
  kIntegerWithUniformMultiplier,
  // Quantized Gemm using a separate multipliers for accumulators of each
  // row of the destination matrix. This is what is called 'per-channel'
  // in GemmParams. Here we use the more specific 'per-row' terminology
  // to allow for the possibility of 'per-column' in the future, and to
  // allow for that to be a separate code path in some back-end such as
  // gemmlowp.
  kIntegerWithPerRowMultiplier
};

// Additional parameters that Gemm needs, beyond what falls into
// the MatrixParams that it takes. Compare to ruy::Spec.
//
// Decoupling AccumScalar from DstScalar (rather than deducing it from that)
// is useful future-proofing. Think of a float16 path using float32 accum.
//
// QuantizationFlavor is passed here even though it's technically not used
// in this class. This is so that we retain the ability in the future to
// specialize this class for quantization flavor, and this allows for
// Gemm to be templatized in quantization_flavor via the GemmParams that it
// takes, allowing for automatic template parameter deduction to take place,
// so that most call sites don't need to specify a QuantizationFlavor
// (only those that need perchannel quantization do).
template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor =
              std::is_floating_point<AccumScalar>::value
                  ? QuantizationFlavor::kFloatingPoint
                  : QuantizationFlavor::kIntegerWithUniformMultiplier>
struct GemmParams
{
  // Only for non-floating-point cases. The fixed-point part (i.e. the mantissa)
  // of the multiplier by which accumulators are multiplied before being casted
  // to the destination type.
  AccumScalar multiplier_fixedpoint = 0;
  // Only for non-floating-point cases. The exponent part of the aforementioned
  // multiplier.
  int multiplier_exponent = 0;
  // Per-channel variant of multiplier_fixedpoint. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_fixedpoint.
  const AccumScalar *multiplier_fixedpoint_perchannel = nullptr;
  // Per-channel variant of multiplier_exponent. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_exponent.
  //
  // Either none or both of multiplier_exponent_perchannel and
  // multiplier_fixedpoint_perchannel must be nullptr.
  const int *multiplier_exponent_perchannel = nullptr;
  // The bias vector data, if not null.
  const AccumScalar *bias = nullptr;
  // min clamp bound of destination values.
  DstScalar clamp_min = std::is_floating_point<DstScalar>::value
                            ? -std::numeric_limits<DstScalar>::infinity()
                            : std::numeric_limits<DstScalar>::lowest();
  // max clamp bound of destination values.
  DstScalar clamp_max = std::is_floating_point<DstScalar>::value
                            ? std::numeric_limits<DstScalar>::infinity()
                            : std::numeric_limits<DstScalar>::max();
};

// Validates self-consistency of GemmParams.
template <typename AccumScalar, typename DstScalar, QuantizationFlavor quantization_flavor>
void ValidateGemmParams(const GemmParams<AccumScalar, DstScalar, quantization_flavor> &params)
{
  // Guard consistency of the quantized multiplier fields.
  if (quantization_flavor == QuantizationFlavor::kFloatingPoint)
  {
    assert(!params.multiplier_fixedpoint);
    assert(!params.multiplier_exponent);
    assert(!params.multiplier_fixedpoint_perchannel);
    assert(!params.multiplier_exponent_perchannel);
  }
  else if (quantization_flavor == QuantizationFlavor::kIntegerWithUniformMultiplier &&
           !std::is_same<DstScalar, int32_t>::value)
  {
    assert(params.multiplier_fixedpoint);
    // Nothing to check about multiplier_exponent
    assert(!params.multiplier_fixedpoint_perchannel);
    assert(!params.multiplier_exponent_perchannel);
  }
  else if (quantization_flavor == QuantizationFlavor::kIntegerWithPerRowMultiplier &&
           !std::is_same<DstScalar, int32_t>::value)
  {
    assert(!params.multiplier_fixedpoint);
    assert(!params.multiplier_exponent);
    assert(params.multiplier_fixedpoint_perchannel);
    assert(params.multiplier_exponent_perchannel);
  }
  else
  {
    // For the get raw accumulator case, we should make sure none of the
    // quantization params are set.
    assert(!params.multiplier_fixedpoint);
    assert(!params.multiplier_exponent);
    assert(!params.multiplier_fixedpoint_perchannel);
    assert(!params.multiplier_exponent_perchannel);
  }
  UNUSED_RELEASE(params);
}

} // namespace ruy
} // namespace nnfw

#endif // __NNFW_RUY_TYPES_H__
