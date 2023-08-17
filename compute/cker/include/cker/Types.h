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

#ifndef __NNFW_CKER_TYPES_H__
#define __NNFW_CKER_TYPES_H__

#include <cstdint>
#include <type_traits>
#include <limits>
#include <string>

namespace nnfw
{
namespace cker
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

enum class BinaryArithmeticOpType
{
  ADD = 0,
  SUB = 1,
  MUL = 2,
  DIV = 3,
  POW = 4,
};

enum class ComparisonOpType
{
  Equal,
  NotEqual,
  Greater,
  GreaterEqual,
  Less,
  LessEqual
};

struct PaddingValues
{
  int16_t width;
  int16_t height;
};

enum class BroadcastableOpCategory : uint8_t
{
  kNone,
  kNonBroadcast,              // Matching input shapes.
  kFirstInputBroadcastsFast,  // Fivefold nested loops.
  kSecondInputBroadcastsFast, // Fivefold nested loops.
  kGenericBroadcast,          // Fall-back.
};

struct PoolParams
{
  PaddingValues padding_values;
  int stride_height;
  int stride_width;
  int filter_height;
  int filter_width;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct SoftmaxParams
{
  // beta is not really used (not a Tensorflow parameter) and not implemented
  // for LogSoftmax.
  double beta;
  int axis;
  // uint8 inference params.  Used even when beta defaults to 1.0.
  int32_t input_multiplier;
  int32_t input_left_shift;
  // Reverse scaling is only used by LogSoftmax.
  int32_t reverse_scaling_divisor;
  int32_t reverse_scaling_right_shift;
  int diff_min;
  int32_t zero_point;
  float scale;
  float *table;
  uint8_t *uint8_table1;
  uint8_t *uint8_table2;
};

struct PackParams
{
  int8_t axis;
  // zeropoint and scale were only used to implement PackWithScaling in the legacy code of
  // tensorflow
  // const int32_t* input_zeropoint;
  // const float* input_scale;
  uint16_t inputs_count;
  // int32_t output_zeropoint;
  // float output_scale;
};

struct UnpackParams
{
  uint16_t num_split;
  int16_t axis;
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

struct ComparisonParams
{
  ComparisonOpType type;
  int left_shift;
  int input1_shift;
  int input2_shift;
  int32_t input1_offset;
  int32_t input1_multiplier;
  int32_t input2_offset;
  int32_t input2_multiplier;
  bool is_broadcast;
};

struct BinaryArithmeticOpParam
{
  // Shape dependent / common to data / op types.
  BroadcastableOpCategory broadcast_category{BroadcastableOpCategory::kNone};
  // uint8 inference params.
  int32_t input1_offset = 0;
  int32_t input2_offset = 0;
  int32_t output_offset = 0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  // Add / Sub, not Mul, uint8 inference params.
  int32_t left_shift = 0;
  int32_t input1_multiplier = 0;
  int32_t input1_shift = 0;
  int32_t input2_multiplier = 0;
  int32_t input2_shift = 0;
  // uint8, etc, activation params.
  int32_t quantized_activation_min = 0;
  int32_t quantized_activation_max = 0;
  // float activation params.
  float float_activation_min = 0;
  float float_activation_max = 0;

  // Processed output dimensions.
  // Let input "a" be the one that broadcasts in the faster-changing dimension.
  // Then, after coalescing, for shapes {a0, a1, a2, a3, a4} and
  // {b0, b1, b2, b3, b4},
  // broadcast_shape[4] = b0 = a0.
  // broadcast_shape[3] = b1; a1 = 1.
  // broadcast_shape[2] = b2 = a2.
  // broadcast_shape[1] = a3; b3 = 1.
  // broadcast_shape[0] = b4 = a4.
  int broadcast_shape[5] = {};
};

struct TransposeParams
{
  int8_t perm_count;
  int32_t perm[4];
};

struct ConcatenationParams
{
  int8_t axis;
  const int32_t *input_zeropoint;
  const float *input_scale;
  uint16_t inputs_count;
  int32_t output_zeropoint;
  float output_scale;
};

struct DepthwiseConvParams
{
  PaddingType padding_type;
  PaddingValues padding_values;
  int16_t stride_width;
  int16_t stride_height;
  int16_t dilation_width_factor;
  int16_t dilation_height_factor;
  int16_t depth_multiplier;
  // uint8 inference params.
  // TODO(b/65838351): Use smaller types if appropriate.
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
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
  // float activation params
  float float_activation_min;
  float float_activation_max;
  // Mark the operands as cacheable if they are unchanging, e.g. weights.
  bool lhs_cacheable;
  bool rhs_cacheable;
  // FullyConnectedWeightsFormat weights_format;
};

struct L2NormParams
{
  // uint8 inference params.
  int32_t input_zero_point;
};

enum LSTMKernelType
{
  kTfLiteLSTMFullKernel = 0,
  kTfLiteLSTMBasicKernel
};

struct LSTMParams
{
  // Parameters for LSTM version 1.
  FusedActivationFunctionType activation{FusedActivationFunctionType::kNone};
  float cell_clip;
  float proj_clip;

  // Parameters for LSTM version 2.
  // kTfLiteLSTMBasicKernel is only supported in version 2 or above.
  LSTMKernelType kernel_type;

  // Parameters for LSTM version 4.
  bool asymmetric_quantize_inputs;
};

struct GatherParams
{
  int32_t axis;
};

struct InstanceNormParams
{
  float epsilon;
  float float_activation_min;
  float float_activation_max;
};

struct ResizeBilinearParams
{
  int32_t output_height;
  int32_t output_width;
  bool align_corners;
  bool half_pixel_centers;
};

struct TransposeConvParams
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
};

struct SliceParams
{
  int8_t begin_count;
  int32_t begin[4];
  int8_t size_count;
  int32_t size[4];
};

struct StridedSliceParams
{
  int8_t start_indices_count;
  int16_t start_indices[4];
  int8_t stop_indices_count;
  int16_t stop_indices[4];
  int8_t strides_count;
  int16_t strides[4];

  int16_t begin_mask;
  int16_t ellipsis_mask;
  int16_t end_mask;
  int16_t new_axis_mask;
  int16_t shrink_axis_mask;
};

struct SplitParams
{
  uint16_t num_split;
  int16_t axis;
};

struct SplitVParams
{
  uint16_t num_split;
  int16_t axis;
};

struct FusedBatchNormParams
{
  bool is_training;
  std::string data_format; // UNKNOWN(0), NHWC(1), NCHW(2)
  float epsilon;
};

struct SpaceToBatchParams
{
  // "Zero" padding for uint8 means padding with the output offset.
  int32_t output_offset;
};

struct SpaceToDepthParams
{
  int32_t block_size;
};

struct LeakyReluParams
{
  float alpha;
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

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TYPES_H__
