/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "DepthwiseConvolutionLayer.h"

#include <cker/operation/DepthwiseConv.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

DepthwiseConvolutionLayer::DepthwiseConvolutionLayer()
    : _input(nullptr), _kernel(nullptr), _bias(nullptr), _output(nullptr), _paddingLeft(0),
      _paddingTop(0), _paddingRight(0), _paddingBottom(0), _strideWidth(0), _strideHeight(0),
      _multiplier(0), _activation(ir::Activation::NONE)
{
  // DO NOTHING
}

void DepthwiseConvolutionLayer::convFloat32()
{
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  nnfw::cker::DepthwiseConvParams op_params;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.depth_multiplier = _multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  nnfw::cker::DepthwiseConv(
      op_params, getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
      getTensorShape(_kernel), reinterpret_cast<const float *>(_kernel->buffer()),
      getTensorShape(_bias), reinterpret_cast<const float *>(_bias->buffer()),
      getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()));
}

void DepthwiseConvolutionLayer::convQuant8()
{
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  CalculateActivationRangeUint8(_activation, _output, &output_activation_min,
                                &output_activation_max);

  double real_multiplier = 0.0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  GetQuantizedConvolutionMultiplier(_input, _kernel, _bias, _output, &real_multiplier);
  QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  nnfw::cker::DepthwiseConvParams op_params;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = 1;
  op_params.dilation_height_factor = 1;
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.depth_multiplier = _multiplier;
  op_params.input_offset = -_input->data_offset();
  op_params.weights_offset = -_kernel->data_offset();
  op_params.output_offset = _output->data_offset();
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  nnfw::cker::DepthwiseConv(
      op_params, getTensorShape(_input), reinterpret_cast<const uint8_t *>(_input->buffer()),
      getTensorShape(_kernel), reinterpret_cast<const uint8_t *>(_kernel->buffer()),
      getTensorShape(_bias), reinterpret_cast<const int32_t *>(_bias->buffer()),
      getTensorShape(_output), reinterpret_cast<uint8_t *>(_output->buffer()));
}

void DepthwiseConvolutionLayer::configure(const IPortableTensor *input,
                                          const IPortableTensor *kernel,
                                          const IPortableTensor *bias, const uint32_t paddingLeft,
                                          const uint32_t paddingRight, const uint32_t paddingTop,
                                          const uint32_t paddingBottom, const uint32_t strideWidth,
                                          const uint32_t strideHeight, const uint32_t multiplier,
                                          const ir::Activation activation, IPortableTensor *output)
{
  _input = input;
  _kernel = kernel;
  _bias = bias;
  _paddingLeft = paddingLeft;
  _paddingRight = paddingRight;
  _paddingTop = paddingTop;
  _paddingBottom = paddingBottom;
  _strideWidth = strideWidth;
  _strideHeight = strideHeight;
  _multiplier = multiplier;
  _activation = activation;
  _output = output;
}

void DepthwiseConvolutionLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    convFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    convQuant8();
  }
  else
  {
    throw std::runtime_error{"DepthwiseConv: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
