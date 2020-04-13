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

#include "cker/eigen/EigenSupport.h"
#include "cker/eigen/Utils.h"
#include "cker/neon/neon_check.h"
#include "cker/operation/Common.h"
#include "cker/Shape.h"
#include "cker/Types.h"
#include <vector>

namespace nnfw
{
namespace cker
{
namespace optimized
{
inline void AddBiasAndEvalActivationFunction(float output_activation_min,
                                             float output_activation_max, const Shape &bias_shape,
                                             const float *bias_data, const Shape &array_shape,
                                             float *array_data)
{
  BiasAndClamp(output_activation_min, output_activation_max, bias_shape.FlatSize(), bias_data,
               array_shape.FlatSize(), array_data);
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
