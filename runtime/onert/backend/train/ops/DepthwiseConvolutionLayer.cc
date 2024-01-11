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

#include <cker/train/operation/ReLU.h>
#include <cker/operation/Reduce.h>

#include <chrono>
#include <util/logging.h>

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
    _back_prop_input{nullptr}, _back_prop_output{nullptr}, _act_back_prop_output{nullptr},
    _use_padded_filter{false}, _padded_filter{nullptr}, _filter_buffers{nullptr},
    _filter_dim_buffers{nullptr}, _dconv_kernel{new nnfw::cker::train::DepthwiseConv()}
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

  auto beg = std::chrono::high_resolution_clock::now();

  int64_t kPacketSize;
  const auto data_type = _back_prop_output->data_type();
  assert(data_type == _input->data_type());
  switch (data_type)
  {
    case OperandType::FLOAT32:
    {
      kPacketSize = _dconv_kernel->kPacketSize<float>();
      break;
    }
    default:
      throw std::runtime_error("train DepthwiseConvolutionLayer: unsupported data type");
  }

  const auto incoming_shape = getShape(_back_prop_output);
  const auto filter_shape = getShape(_kernel);
  const int batch = incoming_shape.Dims(0);
  const int out_depth = incoming_shape.Dims(3);
  const int filter_rows = filter_shape.Dims(1);
  const int filter_cols = filter_shape.Dims(2);

  const int filter_spatial_size = filter_rows * filter_cols;
  const int padded_filter_inner_dim_size =
    ((out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

  _use_padded_filter = (out_depth % kPacketSize) == 0 ? false : true;

  // prepare padded_filter buffer for cker
  {
    auto padded_filter_info = ir::OperandInfo(_kernel->get_info());
    padded_filter_info.shape({batch, filter_spatial_size, padded_filter_inner_dim_size});
    _padded_filter = std::make_unique<Tensor>(padded_filter_info, _kernel->layout());
    _padded_filter->setBuffer(std::make_shared<basic::Allocator>(_padded_filter->total_size()));
  }

  // prepare out_bprop and in_bprop buffer for cker
  {
    const int thread_count = _dconv_kernel->getThreadCount() + 1;

    auto filter_buffers_info = ir::OperandInfo(_kernel->get_info());
    filter_buffers_info.shape({thread_count, filter_spatial_size, padded_filter_inner_dim_size});
    _filter_buffers = std::make_unique<Tensor>(filter_buffers_info, _kernel->layout());
    _filter_buffers->setBuffer(std::make_shared<basic::Allocator>(_filter_buffers->total_size()));

    auto filter_dim_buffers_info = ir::OperandInfo(_back_prop_input->get_info());
    filter_dim_buffers_info.shape({thread_count, padded_filter_inner_dim_size});
    _filter_dim_buffers =
      std::make_unique<Tensor>(filter_dim_buffers_info, _back_prop_input->layout());
    _filter_dim_buffers->setBuffer(
      std::make_shared<basic::Allocator>(_filter_dim_buffers->total_size()));
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
  VERBOSE(DepthwiseConvolutionLayer)
    << "DepthwiseConvolutionLayer configure time = " << duration.count() << std::endl;
}

void DepthwiseConvolutionLayer::forward(bool) { cpu::ops::DepthwiseConvolutionLayer::run(); }

void DepthwiseConvolutionLayer::backward()
{
  MEASURE_TIME_START(backward);

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

  MEASURE_TIME_END(backward);
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

  MEASURE_TIME_START(backpropInput);

  // Calculate gradient for input
  _dconv_kernel->backpropInput(
    dconv_params, getShape(backprop_act), getBuffer<float>(backprop_act), getShape(_kernel),
    getBuffer<float>(_kernel), getBuffer<float>(_padded_filter.get()), getShape(_back_prop_input),
    getBuffer<float>(_back_prop_input), _use_padded_filter, getBuffer<float>(_filter_buffers.get()),
    getBuffer<float>(_filter_dim_buffers.get()));

  MEASURE_TIME_END(backpropInput);

  MEASURE_TIME_START(backpropFilter);

  // Calculate gradient for weights
  _dconv_kernel->backpropFilter(
    dconv_params, getShape(backprop_act), getBuffer<float>(backprop_act), getShape(_input),
    getBuffer<float>(_input), getShape(_grad_weights), getBuffer<float>(_grad_weights),
    getBuffer<float>(_padded_filter.get()), getBuffer<float>(_filter_buffers.get()));

  MEASURE_TIME_END(backpropFilter);

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
