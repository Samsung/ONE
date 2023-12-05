/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/Padding.h"

namespace onert
{
namespace backend
{
namespace xnnpack
{
namespace ops
{

DepthwiseConvolutionLayer::DepthwiseConvolutionLayer(
  const std::shared_ptr<ExternalContext> external_context)
  : Layer(external_context), _input(nullptr), _kernel(nullptr), _bias(nullptr), _output(nullptr),
    _padding_type(ir::PaddingType::EXPLICIT), _padding_left(0), _padding_top(0), _padding_right(0),
    _padding_bottom(0), _stride_width(0), _stride_height(0), _multiplier(1),
    _dilation_width_factor(1), _dilation_height_factor(1), _activation(ir::Activation::NONE)
{
  // DO NOTHING
}

void DepthwiseConvolutionLayer::configure(
  const IPortableTensor *input, const IPortableTensor *kernel, const IPortableTensor *bias,
  ir::PaddingType padding_type, const uint32_t padding_left, const uint32_t padding_right,
  const uint32_t padding_top, const uint32_t padding_bottom, const uint32_t stride_width,
  const uint32_t stride_height, const uint32_t multiplier, const uint32_t dilation_width_factor,
  const uint32_t dilation_height_factor, const ir::Activation activation, IPortableTensor *output)
{
  _input = input;
  _kernel = kernel;
  _bias = bias;
  _padding_type = padding_type;
  _padding_left = padding_left;
  _padding_right = padding_right;
  _padding_top = padding_top;
  _padding_bottom = padding_bottom;
  _stride_width = stride_width;
  _stride_height = stride_height;
  _multiplier = multiplier;
  _dilation_width_factor = dilation_width_factor;
  _dilation_height_factor = dilation_height_factor;
  _activation = activation;
  _output = output;

  // TODO Support not nhwc layer
  assert(_input->layout() == ir::Layout::NHWC || _input->layout() == ir::Layout::UNKNOWN);

  assert(_activation == ir::Activation::NONE || _activation == ir::Activation::RELU ||
         _activation == ir::Activation::RELU1 || _activation == ir::Activation::RELU6);
}

void DepthwiseConvolutionLayer::run()
{
  assert(_external_context && _external_context->getThreadPool());
  if (!_setup)
  {
    _setup = setup();
    assert(_setup);
  }

  if (_input->data_type() == OperandType::FLOAT32)
  {
    enum xnn_status status = xnn_run_operator(_kernel_op, _external_context->getThreadPool());
    if (status != xnn_status_success)
    {
      throw std::runtime_error{"failed to run FP32 DepthwiseConvolution operator"};
    }
  }
  else
  {
    throw std::runtime_error{"XNNPACK DepthwiseConv: unsupported data type"};
  }
}

bool DepthwiseConvolutionLayer::create()
{
  float output_activation_min = 0.f, output_activation_max = 0.f;
  CalculateActivationRange<float>(_activation, &output_activation_min, &output_activation_max);

  // NHWC
  // Kernel format is [1, kernel_height, kernel_width, depth_out].
  const auto &kernel_shape = _kernel->getShape();
  uint32_t kernel_height = kernel_shape.dim(1);
  uint32_t kernel_width = kernel_shape.dim(2);
  uint32_t output_channels = kernel_shape.dim(3);
  uint32_t input_channels = _input->getShape().dim(3);
  assert(static_cast<uint32_t>(_output->getShape().dim(3)) == output_channels);
  assert(output_channels == input_channels * _multiplier);

  enum xnn_status status = xnn_create_convolution2d_nhwc_f32(
    _padding_top, _padding_right, _padding_bottom, _padding_left, kernel_height, kernel_width,
    _stride_height, _stride_width, _dilation_height_factor, _dilation_width_factor,
    input_channels /* groups */, 1 /* group_input_channels */,
    _multiplier /* group_output_channels */, input_channels /* input_channel_stride */,
    output_channels /* output_channel_stride */, reinterpret_cast<const float *>(_kernel->buffer()),
    reinterpret_cast<const float *>(_bias->buffer()), output_activation_min, output_activation_max,
    XNN_FLAG_DEPTHWISE_CONVOLUTION, &_kernel_op);
  if (status != xnn_status_success)
  {
    throw std::runtime_error{"failed to create FP32 DepthwiseConvolution operator"};
  }
  assert(_kernel_op != nullptr);
  return true;
}

bool DepthwiseConvolutionLayer::setup()
{
  if (_input->buffer() == nullptr || _output->buffer() == nullptr)
  {
    // it could be models's input or output
    return false;
  }

  uint32_t input_width = _input->getShape().dim(2);
  uint32_t input_height = _input->getShape().dim(1);
  uint32_t batch_size = _input->getShape().dim(0);
  enum xnn_status status = xnn_setup_convolution2d_nhwc_f32(
    _kernel_op, batch_size, input_height, input_width,
    reinterpret_cast<const float *>(_input->buffer()), reinterpret_cast<float *>(_output->buffer()),
    _external_context->getThreadPool());
  if (status != xnn_status_success)
  {
    throw std::runtime_error{"failed to create FP32 DepthwiseConvolution operator"};
  }
  return true;
}

} // namespace ops
} // namespace xnnpack
} // namespace backend
} // namespace onert
