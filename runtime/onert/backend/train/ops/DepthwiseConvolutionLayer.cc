/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "OperationUtils.h"

#include <cker/train/operation/DepthwiseConv.h>
#include <cker/train/operation/ReLU.h>
#include <cker/operation/Reduce.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

DepthwiseConvolutionLayer::DepthwiseConvolutionLayer()
  : cpu::ops::DepthwiseConvolutionLayer(), _grad_weights{nullptr}, _grad_bias{nullptr},
    _back_prop_input{nullptr}, _back_prop_output{nullptr}, _act_back_prop_output{nullptr}
{
  // DO NOTHING
}

DepthwiseConvolutionLayer::~DepthwiseConvolutionLayer() = default;

void DepthwiseConvolutionLayer::configure(
  const IPortableTensor *input, const IPortableTensor *kernel, const IPortableTensor *bias,
  IPortableTensor *output, IPortableTensor *back_prop_input, IPortableTensor *grad_weights,
  IPortableTensor *grad_bias, const IPortableTensor *back_prop_output, const uint32_t paddingLeft,
  const uint32_t paddingRight, const uint32_t paddingTop, const uint32_t paddingBottom,
  const uint32_t strideWidth, const uint32_t strideHeight, const uint32_t multiplier,
  const uint32_t dilationWidth, const uint32_t dilationHeight, const ir::Activation activation,
  const std::shared_ptr<ExternalContext> &external_context)
{
  cpu::ops::DepthwiseConvolutionLayer::configure(
    input, kernel, bias, paddingLeft, paddingRight, paddingTop, paddingBottom, strideWidth,
    strideHeight, multiplier, dilationWidth, dilationHeight, activation, output, external_context);
  _back_prop_input = back_prop_input;
  _back_prop_output = back_prop_output;
  _grad_weights = grad_weights;
  _grad_bias = grad_bias;

  if (activation != ir::Activation::NONE)
  {
    _act_back_prop_output =
      std::make_unique<BackPropTensor>(_back_prop_output->get_info(), _back_prop_output->layout());
    _act_back_prop_output->setBuffer(
      std::make_shared<basic::Allocator>(_act_back_prop_output->total_size()));
  }
}

void DepthwiseConvolutionLayer::forward(bool) { cpu::ops::DepthwiseConvolutionLayer::run(); }

void DepthwiseConvolutionLayer::backward()
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
      throw std::runtime_error{"train DepthwiseConvolutionLayer: unsupported data type"};
  }
}

void DepthwiseConvolutionLayer::backwardFloat32()
{
  // Calculate gradient for activation
  const IPortableTensor *backprop_act;
  switch (_activation)
  {
    case ir::Activation::NONE:
      backprop_act = _back_prop_output;
      break;
    case ir::Activation::RELU:
      nnfw::cker::train::ReLUGrad(getShape(_output), getBuffer<float>(_output),
                                  getShape(_back_prop_output), getBuffer<float>(_back_prop_output),
                                  getShape(_act_back_prop_output.get()),
                                  getBuffer<float>(_act_back_prop_output.get()));
      backprop_act = _act_back_prop_output.get();
      break;
    default:
      throw std::runtime_error("train DepthwiseConvolutionLayer: Unsupported activation type yet");
  }

  nnfw::cker::DepthwiseConvParams dconv_params;
  dconv_params.stride_width = _strideWidth;
  dconv_params.stride_height = _strideHeight;
  dconv_params.padding_values.width = _paddingLeft;
  dconv_params.padding_values.height = _paddingTop;
  dconv_params.depth_multiplier = _multiplier;

  // Calculate gradient for input
  nnfw::cker::train::DepthwiseConvInputGrad(
    dconv_params, getShape(backprop_act), getBuffer<float>(backprop_act), getShape(_kernel),
    getBuffer<float>(_kernel), getShape(_back_prop_input), getBuffer<float>(_back_prop_input));

  // Calculate gradient for weights
  nnfw::cker::train::DepthwiseConvFilterGradRef(
    dconv_params, getShape(backprop_act), getBuffer<float>(backprop_act), getShape(_input),
    getBuffer<float>(_input), getShape(_grad_weights), getBuffer<float>(_grad_weights));

  // Calculate gradient for bias
  if (_bias)
  {
    // TODO Use optimized kernel
    assert(_grad_bias);
    std::vector<int32_t> axes{0, 1, 2};
    nnfw::cker::Reduce reduce_kernel;
    reduce_kernel.prepare(backprop_act->getShape().rank(), axes.size());
    bool result = reduce_kernel.ReduceGeneric<float>(
      getShape(backprop_act), getBuffer<float>(backprop_act), getShape(_grad_bias),
      getBuffer<float>(_grad_bias), axes, false /* keep_dims */, 0.f,
      [](const float current, const float in) -> float { return in + current; });

    if (!result)
    {
      throw std::runtime_error{"train DepthwiseConvolutionLayer: Fail to caculate bias gradient"};
    }
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
