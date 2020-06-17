/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_FUSEDBATCHNORM_H__
#define __NNFW_CKER_FUSEDBATCHNORM_H__

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"

#include "cker/operation/EinsumHelper/Tensor.h"
#include "cker/operation/EinsumHelper/MatmulBCast.h"

#include "Transpose.h"
#include "BatchMatMul.h"

#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>

namespace nnfw
{
namespace cker
{

class FusedBatchNorm
{
public:
  FusedBatchNorm() : _prepared(false)
  {
    // DO NOTHING
  }

  void prepare() { _prepared = true; }

  void operator()(const std::vector<Shape> &input_shapes,
                  const std::vector<const float *> &input_data, const Shape &output_shape,
                  float *output_data, FusedBatchNormParams param)
  {
    // TODO: support fused_batch_norm if is_traninig is false
    assert(param.is_training == true);

    // TODO: support case where dim[1] != 1 or dim[3] !=1.
    // Here we only support input tensor of [B, 1, X, 1] shape
    assert(input_shapes[0].Dims(1) == 1 && input_shapes[0].Dims(3) == 1);

    if (!_prepared)

    {
      prepare();
    }

    Tensor transformed_input[5];
    Tensor transformed_output;

    const int num_inputs = input_shapes.size();
    std::vector<InputTensor> inputs(num_inputs);
    for (int i = 0; i < num_inputs; i++)
    {
      inputs[i].shape.ReplaceWith(input_shapes[i].DimensionsCount(), input_shapes[i].DimsData());
      inputs[i].buffer = input_data[i];
      copyFrom(inputs[i], inputs[i].shape, &transformed_input[i]);
    }

    InputTensor output;
    output.shape.ReplaceWith(output_shape.DimensionsCount(), output_shape.DimsData());
    output.buffer = output_data;
    copyFrom(output, output.shape, &transformed_output);

    // TODO: support transpose if data_format is NCHW
    // Here, Eigen use RowMajor kernel(NHWC)

    typename TTypes<float, 4>::Tensor x(transformed_input[0].shaped<4>());
    typename TTypes<float, 4>::Tensor y(transformed_output.shaped<4>());
    typename TTypes<float, 1>::Tensor scale(transformed_input[1].shaped<1>());
    typename TTypes<float, 1>::Tensor offset(transformed_input[2].shaped<1>());

    const int depth = x.dimension(3);
    const int size = x.size();
    const int rest_size = size / depth;
    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);

    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);
    Eigen::array<int, 1> reduce_dims({0});
    Eigen::array<int, 2> bcast_spec({rest_size, 1});

    auto x_rest_by_depth = x.reshape(rest_by_depth).template cast<float>();
    const int rest_size_minus_one = (rest_size > 1) ? (rest_size - 1) : 1;
    float rest_size_inv = static_cast<float>(1.0f / static_cast<float>(rest_size));
    // This adjustment is for Bessel's correction
    float rest_size_adjust =
        static_cast<float>(rest_size) / static_cast<float>(rest_size_minus_one);

    Eigen::Tensor<float, 1, Eigen::RowMajor> batch_mean(depth);
    Eigen::Tensor<float, 1, Eigen::RowMajor> batch_variance(depth);

    const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();

    batch_mean.device(d) = (x_rest_by_depth.sum(reduce_dims) * rest_size_inv);
    auto x_centered = x_rest_by_depth - batch_mean.reshape(one_by_depth).broadcast(bcast_spec);

    batch_variance.device(d) = x_centered.square().sum(reduce_dims) * rest_size_inv;
    auto scaling_factor = ((batch_variance + param.epsilon).rsqrt() * scale)
                              .eval()
                              .reshape(one_by_depth)
                              .broadcast(bcast_spec);
    auto x_scaled = x_centered * scaling_factor;
    auto x_shifted =
        (x_scaled + offset.reshape(one_by_depth).broadcast(bcast_spec)).template cast<float>();

    UNUSED_RELEASE(rest_size_adjust);

    y.reshape(rest_by_depth).device(d) = x_shifted;

    memcpy(output_data, y.data(), output_shape.FlatSize() * sizeof(float));
  }

  void copyFrom(const InputTensor &input, const Shape &shape, Tensor *output)
  {
    Tensor temp_tensor;
    temp_tensor.shape.ReplaceWith(input.shape.DimensionsCount(), input.shape.DimsData());
    temp_operand.emplace_back(std::make_unique<float[]>(input.shape.FlatSize()));
    temp_tensor.buffer = temp_operand.back().get();
    memcpy(temp_tensor.buffer, input.buffer, input.shape.FlatSize() * sizeof(float));

    copyFrom(temp_tensor, shape, output);
  }

  void copyFrom(const Tensor &input, const Shape &shape, Tensor *output)
  {
    if (output->copyFrom(input, shape))
      return;

    throw std::runtime_error{"Einsum: Encountered error while reshaping a Tensor"};
  }

private:
  bool _prepared;
  std::vector<std::unique_ptr<float[]>> temp_operand;
};

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_FUSEDBATCHNORM_H__
