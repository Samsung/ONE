/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
    : KernelWithParams<TransposeConvParams>(params), _outputShape(outputShape), _weights(weights),
      _inputData(inputData), _output(output)
{
}

void TransposeConv::configure()
{
  assert(_outputShape->shape().num_dims() == 1);
  assert(_inputData->shape().num_dims() == 4);
  assert(_weights->shape().num_dims() == 4);
  assert(_inputData->element_type() == DataType::FLOAT32 ||
         _inputData->element_type() == DataType::U8);
  assert(_inputData->element_type() == _output->element_type());
  assert(_inputData->shape().dim(3) == _weights->shape().dim(3));
  Shape im2col_shape(2);
  im2col_shape.dim(0) = _inputData->shape().dim(1) * _inputData->shape().dim(2);
  im2col_shape.dim(1) =
      _weights->shape().dim(0) * _weights->shape().dim(1) * _weights->shape().dim(2);
  _im2col =
      std::make_unique<Tensor>(_inputData->element_type(), im2col_shape, AffineQuantization{}, "");
  if (_inputData->element_type() == DataType::U8)
  {
    _scratch_tensor =
        std::make_unique<Tensor>(DataType::S32, _output->shape(), AffineQuantization{}, "");
    double real_multiplier = 0.0;
    const double input_product_scale = _inputData->scale() * _weights->scale();
    assert(input_product_scale >= 0);
    real_multiplier = input_product_scale / _output->scale();
    int exponent;
    quantizeMultiplier(real_multiplier, &_output_multiplier, &exponent);
    _output_shift = -exponent;
    _output_activation_min = std::numeric_limits<uint8_t>::min();
    _output_activation_max = std::numeric_limits<uint8_t>::max();
  }

  int dims = _outputShape->shape().dim(0);
  Shape output_shape(dims);
  if (_inputData->element_type() == DataType::FLOAT32)
  {
    const float *shape_data = getTensorData<float>(_outputShape);
    for (int i = 0; i < dims; i++)
      output_shape.dim(i) = shape_data[i];
  }
  else
  {
    const uint8_t *shape_data = getTensorData<uint8_t>(_outputShape);
    for (int i = 0; i < dims; i++)
      output_shape.dim(i) = shape_data[i];
  }
  _output->resize(output_shape);
}

void TransposeConv::execute() const
{
  switch (_inputData->element_type())
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
  tflite::reference_ops::TransposeConv(op_params, getTensorShape(_inputData),
                                       getTensorData<float>(_inputData), getTensorShape(_weights),
                                       getTensorData<float>(_weights), getTensorShape(_output),
                                       getTensorData<float>(_output), getTensorShape(_im2col.get()),
                                       getTensorData<float>(_im2col.get()));
}

void TransposeConv::evalQuantized() const
{
  int32_t input_offset = -_inputData->zero_point();
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
      op_params, getTensorShape(_inputData), getTensorData<uint8>(_inputData),
      getTensorShape(_weights), getTensorData<uint8>(_weights), getTensorShape(_output),
      getTensorData<uint8>(_output), getTensorShape(_im2col.get()),
      getTensorData<uint8>(_im2col.get()), getTensorData<int32_t>(_scratch_tensor.get()));
}

} // namespace kernels
} // namespace luci_interpreter
