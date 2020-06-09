/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/TransposeConv.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

TransposeConv::TransposeConv(const Tensor *outputShape, const Tensor *weights,
                             const Tensor *inputData, Tensor *output,
                             const TransposeConvParams &params)
    : KernelWithParams<TransposeConvParams>(params), _output_shape(outputShape), _weights(weights),
      _input_data(inputData), _output(output)
{
}

void TransposeConv::configure()
{
  assert(_output_shape->shape().num_dims() == 1);
  assert(_input_data->shape().num_dims() == 4);
  assert(_weights->shape().num_dims() == 4);
  assert(_input_data->element_type() == DataType::FLOAT32 ||
         _input_data->element_type() == DataType::U8);
  assert(_input_data->element_type() == _output->element_type());
  assert(_input_data->shape().dim(3) == _weights->shape().dim(3));
  if (_input_data->element_type() == DataType::U8)
  {
    _scratch_tensor =
        std::make_unique<Tensor>(DataType::S32, _output->shape(), AffineQuantization{}, "");
    double real_multiplier = 0.0;
    const double input_product_scale = _input_data->scale() * _weights->scale();
    assert(input_product_scale >= 0);
    real_multiplier = input_product_scale / _output->scale();
    int exponent;
    quantizeMultiplier(real_multiplier, &_output_multiplier, &exponent);
    _output_shift = -exponent;
    _output_activation_min = std::numeric_limits<uint8_t>::min();
    _output_activation_max = std::numeric_limits<uint8_t>::max();
  }

  int dims = _output_shape->shape().dim(0);
  Shape output_shape(dims);
  const int32_t *shape_data = getTensorData<int32_t>(_output_shape);
  for (int i = 0; i < dims; i++)
    output_shape.dim(i) = shape_data[i];
  _output->resize(output_shape);
}

void TransposeConv::execute() const
{
  switch (_input_data->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void TransposeConv::evalFloat() const
{
  const int width = _output->shape().dim(2);
  const int height = _output->shape().dim(1);

  const int filter_width = _weights->shape().dim(2);
  const int filter_height = _weights->shape().dim(1);

  int unused_output_height, unused_output_width;
  unused_output_width =
      computeOutputSize(params().padding, width, filter_width, params().stride_width, 1);
  unused_output_height =
      computeOutputSize(params().padding, height, filter_height, params().stride_height, 1);
  int32_t offset = 0;
  tflite::ConvParams op_params;
  op_params.padding_type = tflite::PaddingType::kSame;
  op_params.padding_values.height = computePaddingWithOffset(
      params().stride_height, 1, height, filter_height, unused_output_height, &offset);
  op_params.padding_values.height_offset = offset;
  op_params.padding_values.width = computePaddingWithOffset(
      params().stride_width, 1, width, filter_width, unused_output_width, &offset);
  op_params.padding_values.width_offset = offset;
  op_params.stride_height = params().stride_height;
  op_params.stride_width = params().stride_width;
  op_params.output_multiplier = _output_multiplier;
  tflite::reference_ops::TransposeConv(
      op_params, getTensorShape(_input_data), getTensorData<float>(_input_data),
      getTensorShape(_weights), getTensorData<float>(_weights), getTensorShape(_output),
      getTensorData<float>(_output), tflite::RuntimeShape(), (float *)nullptr);
}

void TransposeConv::evalQuantized() const
{
  int32_t input_offset = -_input_data->zero_point();
  int32_t filter_offset = -_weights->zero_point();
  int32_t output_offset = _weights->zero_point();
  const int width = _output->shape().dim(2);
  const int height = _output->shape().dim(1);

  const int filter_width = _weights->shape().dim(2);
  const int filter_height = _weights->shape().dim(1);

  int unused_output_height, unused_output_width;
  unused_output_width =
      computeOutputSize(params().padding, width, filter_width, params().stride_width, 1);
  unused_output_height =
      computeOutputSize(params().padding, height, filter_height, params().stride_height, 1);
  int32_t offset = 0;
  tflite::ConvParams op_params;
  op_params.padding_type = tflite::PaddingType::kSame;
  op_params.padding_values.height = computePaddingWithOffset(
      params().stride_height, 1, height, filter_height, unused_output_height, &offset);
  op_params.padding_values.width = computePaddingWithOffset(
      params().stride_width, 1, width, filter_width, unused_output_width, &offset);
  op_params.stride_height = params().stride_height;
  op_params.stride_width = params().stride_width;
  op_params.input_offset = input_offset;
  op_params.output_offset = output_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_multiplier = _output_multiplier;
  op_params.output_shift = -_output_shift;
  op_params.quantized_activation_min = _output_activation_min;
  op_params.quantized_activation_max = _output_activation_max;

  tflite::reference_ops::TransposeConv(
      op_params, getTensorShape(_input_data), getTensorData<uint8>(_input_data),
      getTensorShape(_weights), getTensorData<uint8>(_weights), getTensorShape(_output),
      getTensorData<uint8>(_output), tflite::RuntimeShape(), (uint8 *)nullptr,
      getTensorData<int32_t>(_scratch_tensor.get()));
}

} // namespace kernels
} // namespace luci_interpreter
