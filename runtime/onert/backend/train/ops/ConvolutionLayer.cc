/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConvolutionLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Conv.h>
#include <cker/operation/TransposeConv.h>
#include <cker/operation/Transpose.h>
#include <cker/train/operation/Conv.h>
#include <cker/train/operation/ReLU.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{
ConvolutionLayer::ConvolutionLayer()
  : cpu::ops::ConvolutionLayer(), _grad_weights{nullptr}, _grad_bias{nullptr},
    _deriv_input{nullptr}, _deriv_output{nullptr}, _transposed_weights{nullptr}
{
  // DO NOTHING
}

ConvolutionLayer::~ConvolutionLayer() = default;

void ConvolutionLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                 const IPortableTensor *bias, IPortableTensor *output,
                                 IPortableTensor *deriv_input, IPortableTensor *grad_weights,
                                 IPortableTensor *grad_bias, const IPortableTensor *deriv_output,
                                 ir::PaddingType paddingType, const uint32_t paddingLeft,
                                 const uint32_t paddingRight, const uint32_t paddingTop,
                                 const uint32_t paddingBottom, const uint32_t strideWidth,
                                 const uint32_t strideHeight, const uint32_t dilationWidthFactor,
                                 const uint32_t dilationHeightFactor,
                                 const ir::Activation activation)
{
  cpu::ops::ConvolutionLayer::configure(
    input, weights, bias, paddingType, paddingLeft, paddingRight, paddingTop, paddingBottom,
    strideWidth, strideHeight, dilationWidthFactor, dilationHeightFactor, activation, output);

  _deriv_input = deriv_input;
  _grad_weights = grad_weights;
  _grad_bias = grad_bias;
  _deriv_output = deriv_output;

  _transposed_weights = std::make_unique<Tensor>(weights->get_info(), weights->layout());
  _transposed_weights->setBuffer(std::make_shared<basic::Allocator>(weights->total_size()));

  _conv_deriv_output =
    std::make_unique<DerivativeTensor>(deriv_output->get_info(), deriv_output->layout());
  _conv_deriv_output->setBuffer(std::make_shared<basic::Allocator>(deriv_output->total_size()));

  if (activation != ir::Activation::NONE)
  {
    _act_deriv_output =
      std::make_unique<Tensor>(_deriv_output->get_info(), _deriv_output->layout());
    _act_deriv_output->setBuffer(std::make_shared<basic::Allocator>(_deriv_output->total_size()));
  }
}

void ConvolutionLayer::forward(bool) { cpu::ops::ConvolutionLayer::run(); }
void ConvolutionLayer::backward()
{
  const auto data_type = _deriv_output->data_type();
  assert(data_type == _input->data_type());
  switch (data_type)
  {
    case OperandType::FLOAT32:
    {
      assert(data_type == _grad_bias->data_type());
      backwardFloat32();
      break;
    }
    default:
      throw std::runtime_error{"train ConvolutionLayer: unsupported data type"};
  }
}

void ConvolutionLayer::backwardFloat32()
{
  // Calculate gradient for activation
  const IPortableTensor *backprop_act;
  switch (_activation)
  {
    case ir::Activation::NONE:
      backprop_act = _deriv_output;
      break;
    case ir::Activation::RELU:
      nnfw::cker::train::ReLUGrad(getShape(_output), getBuffer<float>(_output),
                                  getShape(_deriv_output), getBuffer<float>(_deriv_output),
                                  getShape(_act_deriv_output.get()),
                                  getBuffer<float>(_act_deriv_output.get()));
      backprop_act = _act_deriv_output.get();
      break;
    default:
      throw std::runtime_error("train FullyConnectedLayer: Unsupported activation type yet");
  }

  // Transpose weights from OHWI to IHWO
  auto transposed_weights = _transposed_weights.get();
  assert(transposed_weights->getShape().rank() == 4);
  nnfw::cker::TransposeParams transpose_param;
  transpose_param.perm_count = transposed_weights->getShape().rank();
  transpose_param.perm[0] = 3;
  transpose_param.perm[1] = 1;
  transpose_param.perm[2] = 2;
  transpose_param.perm[3] = 0;
  nnfw::cker::Transpose(transpose_param, getShape(_kernel), getBuffer<float>(_kernel),
                        getShape(transposed_weights), getBuffer<float>(transposed_weights));

  // Calculate gradient for input
  // nnfw::cker::TransposeConvParams tconv_op_params;
  // tconv_op_params.padding_type = getPaddingType(_paddingType);
  // tconv_op_params.padding_values.width = _paddingLeft;
  // tconv_op_params.padding_values.height = _paddingTop;
  // tconv_op_params.stride_width = _strideWidth;
  // tconv_op_params.stride_height = _strideHeight;
  // nnfw::cker::TransposeConv(tconv_op_params, getShape(backprop_act),
  //                           getBuffer<float>(backprop_act), getShape(transposed_weights),
  //                           getBuffer<float>(transposed_weights), getShape(_deriv_input),
  //                           getBuffer<float>(_deriv_input));

  // Calculate gradient for weights
  if (_dilationHeightFactor != 1 || _dilationWidthFactor != 1)
    throw std::runtime_error("Convolution: Unsupported dilation yet");
  if (_strideHeight != 1 || _strideWidth != 1)
    throw std::runtime_error("Convolution: Unsupported stride yet");

  nnfw::cker::ConvParams conv_op_params;
  conv_op_params.padding_type = getPaddingType(_paddingType);
  conv_op_params.padding_values.width = _paddingLeft;
  conv_op_params.padding_values.height = _paddingTop;
  conv_op_params.stride_width = _strideWidth;
  conv_op_params.stride_height = _strideHeight;
  conv_op_params.dilation_width_factor = _dilationWidthFactor;
  conv_op_params.dilation_height_factor = _dilationHeightFactor;
  // Ignore activation min/max
  conv_op_params.float_activation_min = std::numeric_limits<float>::min();
  conv_op_params.float_activation_max = std::numeric_limits<float>::max();

  assert(_grad_weights->getShape().rank() == 4);
  const auto nums_channels = _grad_weights->getShape().asFeature(_grad_weights->layout()).C;
  std::vector<float> zeros(nums_channels);
  memset(zeros.data(), 0, nums_channels * sizeof(float));

  nnfw::cker::Conv cal_weights_grad_kernel;
  cal_weights_grad_kernel(conv_op_params, getShape(_input), getBuffer<float>(_input),
                          getShape(backprop_act), getBuffer<float>(backprop_act),
                          nnfw::cker::Shape{nums_channels}, zeros.data(), getShape(_grad_weights),
                          getBuffer<float>(_grad_weights));

  // Calculate gradient for bias
  const auto incomming_shape = backprop_act->getShape();
  assert(incomming_shape.rank() == 4);
  assert(incomming_shape.dim(3) == _grad_bias->getShape().dim(0));
  nnfw::cker::train::ConvBiasGrad<float>(getShape(backprop_act), getBuffer<float>(backprop_act),
                                         getShape(_grad_bias), getBuffer<float>(_grad_bias));
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
