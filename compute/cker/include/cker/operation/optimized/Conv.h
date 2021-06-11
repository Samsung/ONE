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

#ifndef __NNFW_CKER_OPTIMIZED_CONV_H__
#define __NNFW_CKER_OPTIMIZED_CONV_H__

#include "OptimizedUtils.h"

#include "cker/eigen/EigenSupport.h"
#include "cker/eigen/Utils.h"
#include "cker/gemmlowp/GEMMSupport.h"
#include "cker/neon/neon_check.h"
#include "cker/operation/Common.h"
#include "cker/Shape.h"
#include "cker/Types.h"

#include <public/gemmlowp.h>
#include <public/map.h>
#include <fixedpoint/fixedpoint.h>

#include <vector>
#include <tuple>

namespace nnfw
{
namespace cker
{
namespace optimized
{

std::mutex _gemmlowp_mutex;

struct GemmlowpOutputPipeline
{
  typedef gemmlowp::VectorMap<const int32_t, gemmlowp::VectorShape::Col> ColVectorMap;
  typedef std::tuple<gemmlowp::OutputStageBiasAddition<ColVectorMap>,
                     gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent,
                     gemmlowp::OutputStageClamp, gemmlowp::OutputStageSaturatingCastToUint8>
    Pipeline;
  static Pipeline MakeExp(const int32_t *bias_data, int output_rows, int32_t output_offset,
                          int32_t output_multiplier, int output_left_shift,
                          int32_t output_activation_min, int32_t output_activation_max)
  {
    ColVectorMap bias_vector(bias_data, output_rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent quantize_down_stage;
    quantize_down_stage.result_offset_after_shift = output_offset;
    quantize_down_stage.result_fixedpoint_multiplier = output_multiplier;
    quantize_down_stage.result_exponent = output_left_shift;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = output_activation_min;
    clamp_stage.max = output_activation_max;
    gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;
    return std::make_tuple(bias_addition_stage, quantize_down_stage, clamp_stage,
                           saturating_cast_stage);
  }
};

inline void AddBiasAndEvalActivationFunction(float output_activation_min,
                                             float output_activation_max, const Shape &bias_shape,
                                             const float *bias_data, const Shape &array_shape,
                                             float *array_data)
{
  BiasAndClamp(output_activation_min, output_activation_max, bias_shape.FlatSize(), bias_data,
               array_shape.FlatSize(), array_data);
}

inline void Conv(const ConvParams &params, const Shape &input_shape, const uint8_t *input_data,
                 const Shape &filter_shape, const uint8_t *filter_data, const Shape &bias_shape,
                 const int32_t *bias_data, const Shape &output_shape, uint8_t *output_data,
                 const Shape &im2col_shape, uint8_t *im2col_data)
{
  gemmlowp::GemmContext *gemm_context = gemm_support::GetGemmLowpContext();

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int32_t input_offset = params.input_offset;
  const int32_t filter_offset = params.weights_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);

  const uint8_t *gemm_input_data = nullptr;
  const Shape *gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_dilated_im2col = dilation_width_factor != 1 || dilation_height_factor != 1;
  const bool need_im2col =
    stride_width != 1 || stride_height != 1 || filter_width != 1 || filter_height != 1;
  if (need_dilated_im2col)
  {
    assert(im2col_data);
    const int input_zero_point = -input_offset;
    assert(input_zero_point >= 0);
    assert(input_zero_point <= 255);
    DilatedIm2col(params, input_zero_point, input_shape, input_data, filter_shape, output_shape,
                  im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  }
  else if (need_im2col)
  {
    assert(im2col_data);
    const int input_zero_point = -input_offset;
    assert(input_zero_point >= 0);
    assert(input_zero_point <= 255);
    Im2col(params, filter_height, filter_width, input_zero_point, input_shape, input_data,
           im2col_shape, im2col_data);
    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  }
  else
  {
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  const int gemm_input_rows = gemm_input_shape->Dims(3);
  // Using FlatSizeSkipDim causes segfault in some contexts (see b/79927784).
  // The root cause has not yet been identified though. Same applies below for
  // the other calls commented out. This is a partial rollback of cl/196819423.
  // const int gemm_input_cols = FlatSizeSkipDim(*gemm_input_shape, 3);
  const int gemm_input_cols =
    gemm_input_shape->Dims(0) * gemm_input_shape->Dims(1) * gemm_input_shape->Dims(2);
  const int filter_rows = filter_shape.Dims(0);
  // See b/79927784.
  // const int filter_cols = FlatSizeSkipDim(filter_shape, 0);
  const int filter_cols = filter_shape.Dims(1) * filter_shape.Dims(2) * filter_shape.Dims(3);
  const int output_rows = output_shape.Dims(3);
  // See b/79927784.
  // const int output_cols = FlatSizeSkipDim(output_shape, 3);
  const int output_cols = output_shape.Dims(0) * output_shape.Dims(1) * output_shape.Dims(2);
  assert(output_rows == filter_rows);
  assert(output_cols == gemm_input_cols);
  assert(filter_cols == gemm_input_rows);
  assert(bias_shape.FlatSize() == output_rows);
  UNUSED_RELEASE(bias_shape);
  gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::RowMajor> filter_matrix(
    filter_data, filter_rows, filter_cols);
  gemmlowp::MatrixMap<const uint8_t, gemmlowp::MapOrder::ColMajor> input_matrix(
    gemm_input_data, gemm_input_rows, gemm_input_cols);
  gemmlowp::MatrixMap<uint8_t, gemmlowp::MapOrder::ColMajor> output_matrix(output_data, output_rows,
                                                                           output_cols);
  const auto &output_pipeline =
    GemmlowpOutputPipeline::MakeExp(bias_data, output_rows, output_offset, output_multiplier,
                                    output_shift, output_activation_min, output_activation_max);

  std::lock_guard<std::mutex> lock_guard(_gemmlowp_mutex);
  gemmlowp::GemmWithOutputPipeline<uint8_t, uint8_t, gemmlowp::L8R8WithLhsNonzeroBitDepthParams>(
    gemm_context, filter_matrix, input_matrix, &output_matrix, filter_offset, input_offset,
    output_pipeline);
}

} // namespace optimized

namespace multithreaded
{
namespace
{
template <class T> class EigenTensorConvFunctor
{
private:
  Eigen::PaddingType RuntimePadding2EigenPadding(PaddingType padding)
  {
    switch (padding)
    {
      case PaddingType::kValid:
        return Eigen::PADDING_VALID;
      case PaddingType::kSame:
        return Eigen::PADDING_SAME;
      case PaddingType::kNone:
        assert(false); // should never get here.
        return Eigen::PADDING_VALID;
    }
    return Eigen::PADDING_SAME; // Prevent compiler warning about missing
                                // return
  }

public:
  void operator()(const Eigen::ThreadPoolDevice &device, const T *input_data, int input_batches,
                  int input_height, int input_width, int input_depth, const T *filter_data,
                  int filter_height, int filter_width, int filter_count, int stride_rows,
                  int stride_cols, int pad_height, int pad_width, nnfw::cker::PaddingType padding,
                  T *output_data, int output_height, int output_width)
  {
    const bool is_1x1_kernel =
      (filter_height == 1 && filter_width == 1 && stride_rows == 1 && stride_cols == 1);
    const bool is_same_height_width =
      (filter_height == input_height && filter_width == input_width && pad_width == 0 &&
       pad_height == 0);
    if (is_1x1_kernel || is_same_height_width)
    {
      // is_1x1_kernel: For 1x1 kernel, the 2D convolution is reduced to matrix multiplication.
      //  - output (input_batches * conv_width, filter_count)
      //  - input (input_batches * conv_width, input_depth)
      //  - filter (input_depth, filter_count)
      // is_same_height_width: If the input data and filter have the same height/width, the 2D
      // convolution is reduced to matrix multiplication.
      //  - output (input_batches, filter_count)
      //  - input (input_batches, filter_width * filter_height * input_depth)
      //  - filter (filter_width * filter_height * input_depth, filter_count)
      const int conv_width = output_height * output_width;
      int io_col = input_batches;
      int filter_col = input_depth * filter_width * filter_height;
      if (is_1x1_kernel)
      {
        io_col *= conv_width;
      }
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      eigen_support::EigenMatrix output(output_data, io_col, filter_count);
      eigen_support::ConstEigenMatrix input(input_data, io_col, filter_col);
      eigen_support::ConstEigenMatrix filter(filter_data, filter_col, filter_count);
      eigen_support::MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input, filter,
                                                                     dim_pair);
    }
    else
    {
      eigen_support::EigenTensor output(output_data, input_batches, output_height, output_width,
                                        filter_count);
      eigen_support::ConstEigenTensor input(input_data, input_batches, input_height, input_width,
                                            input_depth);
      eigen_support::ConstEigenTensor filter(filter_data, filter_height, filter_width, input_depth,
                                             filter_count);
      output.device(device) = Eigen::SpatialConvolution(input, filter, stride_cols, stride_rows,
                                                        RuntimePadding2EigenPadding(padding));
    }
  }
};
} // namespace

inline void Conv(const ConvParams &params, const Shape &input_shape, const float *input_data,
                 const Shape &filter_shape, const float *filter_data, const Shape &bias_shape,
                 const float *bias_data, const Shape &output_shape, float *output_data)
{
  const Eigen::ThreadPoolDevice &device = *eigen_support::GetThreadPoolDevice();

  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const PaddingType padding = params.padding_type;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  assert(input_shape.DimensionsCount() == 4);
  assert(filter_shape.DimensionsCount() == 4);
  assert(output_shape.DimensionsCount() == 4);

  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  EigenTensorConvFunctor<float> conv_functor;
  conv_functor(device, input_data, batches, input_height, input_width, input_depth, filter_data,
               filter_height, filter_width, output_depth, stride_height, stride_width, pad_height,
               pad_width, padding, output_data, output_height, output_width);

  optimized::AddBiasAndEvalActivationFunction(output_activation_min, output_activation_max,
                                              bias_shape, bias_data, output_shape, output_data);
}

} // namespace multithreaded
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_OPTIMIZED_CONV_H__
