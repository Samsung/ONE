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

#include "FullyConnectedLayer.h"

#include "ir/Padding.h"

namespace onert
{
namespace backend
{
namespace xnnpack
{
namespace ops
{

FullyConnectedLayer::FullyConnectedLayer(const std::shared_ptr<ExternalContext> external_context)
  : Layer(external_context), _input(nullptr), _kernel(nullptr), _bias(nullptr), _output(nullptr),
    _activation(ir::Activation::NONE)
{
  // DO NOTHING
}

void FullyConnectedLayer::configure(const IPortableTensor *input, const IPortableTensor *weights,
                                    const IPortableTensor *bias, ir::Activation activation,
                                    IPortableTensor *output)
{
  _input = input;
  _kernel = weights;
  _bias = bias;
  _activation = activation;
  _output = output;

  // TODO Support not nhwc layer
  assert(_input->layout() == ir::Layout::NHWC || _input->layout() == ir::Layout::UNKNOWN);

  assert(_activation == ir::Activation::NONE || _activation == ir::Activation::RELU ||
         _activation == ir::Activation::RELU1 || _activation == ir::Activation::RELU6);
}

void FullyConnectedLayer::run()
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
      throw std::runtime_error{"failed to run FP32 FullyConnected operator"};
    }
  }
  else
  {
    throw std::runtime_error{"XNNPACK FC: unsupported data type"};
  }
}

bool FullyConnectedLayer::create()
{
  float output_activation_min = 0.f, output_activation_max = 0.f;
  CalculateActivationRange<float>(_activation, &output_activation_min, &output_activation_max);

  const auto &kernel_shape = _kernel->getShape();
  assert(kernel_shape.rank() == 2);
  uint32_t output_channels = kernel_shape.dim(0);
  uint32_t input_channels = kernel_shape.dim(1);

  const auto &input_shape = _input->getShape();
  const auto &output_shape = _output->getShape();
  uint32_t flag = 0;
  if (input_shape.rank() != output_shape.rank())
  {
    flag |= XNN_FLAG_TENSORFLOW_RESHAPE_2D;
    assert(input_shape.num_elements() % input_channels == 0);
  }
  else
  {
    assert(static_cast<uint32_t>(input_shape.dim(input_shape.rank() - 1)) == input_channels);
  }

  assert(_kernel && _kernel->buffer());
  const float *kernel_buffer = reinterpret_cast<const float *>(_kernel->buffer());
  const float *bias_buffer = (_bias) ? reinterpret_cast<const float *>(_bias->buffer()) : nullptr;

  enum xnn_status status = xnn_create_fully_connected_nc_f32(
    input_channels, output_channels, input_channels /* input stride */,
    output_channels /* output stride */, kernel_buffer, bias_buffer, output_activation_min,
    output_activation_max, flag, &_kernel_op);
  if (status != xnn_status_success)
  {
    throw std::runtime_error{"failed to create FP32 FullyConnected operator"};
  }
  assert(_kernel_op != nullptr);
  return true;
}

bool FullyConnectedLayer::setup()
{
  if (_input->buffer() == nullptr || _output->buffer() == nullptr)
  {
    // it could be models's input or output
    return false;
  }

  uint32_t batch_size = _input->getShape().num_elements() / _kernel->getShape().dim(1);
  enum xnn_status status = xnn_setup_fully_connected_nc_f32(
    _kernel_op, batch_size, reinterpret_cast<const float *>(_input->buffer()),
    reinterpret_cast<float *>(_output->buffer()), _external_context->getThreadPool());
  if (status != xnn_status_success)
  {
    throw std::runtime_error{"failed to create FP32 FullyConnected operator"};
  }
  return true;
}

} // namespace ops
} // namespace xnnpack
} // namespace backend
} // namespace onert
