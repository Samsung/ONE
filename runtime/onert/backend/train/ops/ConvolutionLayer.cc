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
#include <cker/operation/Transpose.h>
#include <cker/train/operation/Conv.h>
#include <cker/train/operation/ReLU.h>

#include <cker/operation/TransposeConv.h>

#include "util/logging.h"

namespace
{

using namespace onert;

template <typename Tensor>
std::unique_ptr<Tensor> createTransposedWeights(const backend::IPortableTensor *origin_weights)
{
  const auto &origin_shape = origin_weights->getShape();
  assert(origin_shape.rank() == 4);

  auto transposed_info = origin_weights->get_info();
  // OHWI to HWIO
  auto transposed_shape =
    ir::Shape{origin_shape.dim(1), origin_shape.dim(2), origin_shape.dim(3), origin_shape.dim(0)};
  transposed_info.shape(transposed_shape);

  return std::make_unique<Tensor>(transposed_info, origin_weights->layout());
}

} // namespace

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
    _back_prop_input{nullptr}, _back_prop_output{nullptr}, _transposed_weights{nullptr}
{
  // DO NOTHING
}

ConvolutionLayer::~ConvolutionLayer() = default;

void ConvolutionLayer::configure(
  const IPortableTensor *input, const IPortableTensor *weights, const IPortableTensor *bias,
  IPortableTensor *output, IPortableTensor *back_prop_input, IPortableTensor *grad_weights,
  IPortableTensor *grad_bias, const IPortableTensor *back_prop_output, ir::PaddingType paddingType,
  const uint32_t paddingLeft, const uint32_t paddingRight, const uint32_t paddingTop,
  const uint32_t paddingBottom, const uint32_t strideWidth, const uint32_t strideHeight,
  const uint32_t dilationWidthFactor, const uint32_t dilationHeightFactor,
  const ir::Activation activation)
{
  const bool is_cacheable_weights = false;
  cpu::ops::ConvolutionLayer::configure(input, weights, bias, paddingType, paddingLeft,
                                        paddingRight, paddingTop, paddingBottom, strideWidth,
                                        strideHeight, dilationWidthFactor, dilationHeightFactor,
                                        activation, output, is_cacheable_weights);

  _back_prop_input = back_prop_input;
  _grad_weights = grad_weights;
  _grad_bias = grad_bias;
  _back_prop_output = back_prop_output;

  if (_dilationHeightFactor != 1 || _dilationWidthFactor != 1)
    throw std::runtime_error("train ConvolutionLayer: Unsupported dilation yet");

  // TODO Optimize transposed tensors
  _transposed_weights = createTransposedWeights<Tensor>(weights);
  _transposed_weights->setBuffer(
    std::make_shared<basic::Allocator>(_transposed_weights->total_size()));

  _conv_back_prop_output =
    std::make_unique<BackPropTensor>(back_prop_output->get_info(), back_prop_output->layout());
  _conv_back_prop_output->setBuffer(
    std::make_shared<basic::Allocator>(_conv_back_prop_output->total_size()));

  _transposed_grad_weights = createTransposedWeights<GradientTensor>(weights);
  _transposed_grad_weights->setBuffer(
    std::make_shared<basic::Allocator>(_transposed_grad_weights->total_size()));

  if (activation != ir::Activation::NONE)
  {
    _act_back_prop_output =
      std::make_unique<BackPropTensor>(_back_prop_output->get_info(), _back_prop_output->layout());
    _act_back_prop_output->setBuffer(
      std::make_shared<basic::Allocator>(_act_back_prop_output->total_size()));
  }
}

void ConvolutionLayer::forward(bool) { cpu::ops::ConvolutionLayer::run(); }
void ConvolutionLayer::backward()
{
  const auto data_type = _back_prop_output->data_type();
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
  try
  {
    backprop_act =
      backpropActivation(_activation, _output, _back_prop_output, _act_back_prop_output.get());
  }
  catch (const std::exception &e)
  {
    throw std::runtime_error{"train ConvolutionLayer: " + std::string(e.what())};
  }
  assert(backprop_act != nullptr);

  // Initialize conv params for training kernels
  nnfw::cker::ConvParams conv_train_params;
  conv_train_params.padding_type = getPaddingType(_paddingType);
  conv_train_params.padding_values.width = _paddingLeft;
  conv_train_params.padding_values.height = _paddingTop;
  conv_train_params.stride_width = _strideWidth;
  conv_train_params.stride_height = _strideHeight;
  conv_train_params.dilation_width_factor = _dilationWidthFactor;
  conv_train_params.dilation_height_factor = _dilationHeightFactor;

  // Transpose weights from OHWI to HWIO
  auto transposed_weights = _transposed_weights.get();
  assert(transposed_weights->getShape().rank() == 4);
  nnfw::cker::TransposeParams transpose_param;
  transpose_param.perm_count = transposed_weights->getShape().rank();
  transpose_param.perm[0] = 1;
  transpose_param.perm[1] = 2;
  transpose_param.perm[2] = 3;
  transpose_param.perm[3] = 0;
  MEASURE_TIME_START(Transpose1);
  nnfw::cker::Transpose(transpose_param, getShape(_kernel), getBuffer<float>(_kernel),
                        getShape(transposed_weights), getBuffer<float>(transposed_weights));
  MEASURE_TIME_END(Transpose1, "");

  // Calculate gradient for input
  MEASURE_TIME_START(ConvInputGrad);
  nnfw::cker::train::ConvInputGrad(
    conv_train_params, getShape(backprop_act), getBuffer<float>(backprop_act),
    getShape(transposed_weights), getBuffer<float>(transposed_weights), _paddingBottom,
    _paddingRight, getShape(_back_prop_input), getBuffer<float>(_back_prop_input));
  MEASURE_TIME_END(ConvInputGrad, "");

  // Calculate gradient for weights
  auto transposed_grad_weights = _transposed_grad_weights.get();
  assert(_grad_weights->getShape().rank() == 4);
  assert(transposed_grad_weights->getShape().rank() == 4);
  MEASURE_TIME_START(ConvFilterGrad);
  nnfw::cker::train::ConvFilterGrad(
    conv_train_params, getShape(backprop_act), getBuffer<float>(backprop_act), getShape(_input),
    getBuffer<float>(_input), _paddingBottom, _paddingRight, getShape(transposed_grad_weights),
    getBuffer<float>(transposed_grad_weights));
  MEASURE_TIME_END(ConvFilterGrad, "");

  // Transpose weights'gradient from HWIO to OHWI
  nnfw::cker::TransposeParams transpose_grad_param;
  transpose_grad_param.perm_count = transposed_grad_weights->getShape().rank();
  transpose_grad_param.perm[0] = 3;
  transpose_grad_param.perm[1] = 0;
  transpose_grad_param.perm[2] = 1;
  transpose_grad_param.perm[3] = 2;
  MEASURE_TIME_START(Transpose2);
  nnfw::cker::Transpose(transpose_grad_param, getShape(transposed_grad_weights),
                        getBuffer<float>(transposed_grad_weights), getShape(_grad_weights),
                        getBuffer<float>(_grad_weights));
  MEASURE_TIME_END(Transpose2, "");

  // Calculate gradient for bias
  if (_bias)
  {
    MEASURE_TIME_START(bias);
    biasGrad(backprop_act, _grad_bias);
    MEASURE_TIME_END(bias, "");
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
