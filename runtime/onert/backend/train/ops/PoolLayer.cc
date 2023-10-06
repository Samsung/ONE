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

#include "PoolLayer.h"
#include "OperationUtils.h"
#include "../Tensor.h"

#include <cker/Utils.h>
#include <cker/train/operation/MaxPool.h>
#include <cker/train/operation/ReLU.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

namespace
{

// Convert train op type to inter(cpu) op type
cpu::ops::PoolType convert_to_infer(const train::ops::PoolType &pool_type)
{
  switch (pool_type)
  {
    case train::ops::PoolType::kMax:
      return cpu::ops::PoolType::kMax;
    default:
      throw std::runtime_error("PoolLayer: Unsupported pool type yet");
  }
}

/**
 * MaxPool Training Kernel Registry
 *
 *  - provide forward, bacward interface
 *  - have addtioinal buffer needed to do backpropagate
 */
class MaxPool2D final : public TrainingKernelRegistry
{
private:
  const ir::Activation _activation;
  const IPortableTensor *_output;
  nnfw::cker::PoolParams _op_params;

  std::unique_ptr<Tensor> _act_deriv_output;
  std::unique_ptr<Tensor> _arg_max_index;

public:
  MaxPool2D(const uint32_t paddingLeft, const uint32_t, const uint32_t, const uint32_t paddingTop,
            const uint32_t strideWidth, const uint32_t strideHeight, const uint32_t kernelWidth,
            const uint32_t kernelHeight, const ir::Activation activation,
            const IPortableTensor *output)
    : _activation(activation), _output(output)
  {
    {
      _op_params.stride_height = strideHeight;
      _op_params.stride_width = strideWidth;
      _op_params.filter_height = kernelHeight;
      _op_params.filter_width = kernelWidth;
      _op_params.padding_values.height = (int8_t)paddingTop;
      _op_params.padding_values.width = (int8_t)paddingLeft;
      CalculateActivationRange<float>(activation, &_op_params.float_activation_min,
                                      &_op_params.float_activation_max);
    }

    _arg_max_index = std::make_unique<Tensor>(_output->get_info(), _output->layout());
    _arg_max_index->setBuffer(std::make_shared<basic::Allocator>(_output->total_size()));

    if (activation != ir::Activation::NONE)
    {
      _act_deriv_output = std::make_unique<Tensor>(_output->get_info(), _output->layout());
      _act_deriv_output->setBuffer(std::make_shared<basic::Allocator>(_output->total_size()));
    }
  };

  ~MaxPool2D() {}

public:
  void forward(const IPortableTensor *in, IPortableTensor *out)
  {
    assert(in->layout() == ir::Layout::NHWC);

    auto out_shape = getShape(out);
    auto out_data = getBuffer<float>(out);
    auto arg_max_index = _arg_max_index.get();

    // maxpool forward
    nnfw::cker::train::MaxPool2D(_op_params, getShape(in), getBuffer<float>(in), out_shape,
                                 out_data, getBuffer<int>(arg_max_index));

    // activation forward
    for (int i = 0; i < out_shape.FlatSize(); ++i)
    {
      out_data[i] = nnfw::cker::ActivationFunctionWithMinMax(
        out_data[i], _op_params.float_activation_min, _op_params.float_activation_max);
    }
  }

  void backward(const IPortableTensor *deriv_out, IPortableTensor *deriv_in)
  {
    assert(deriv_out->layout() == ir::Layout::NHWC);

    // activation bacward
    switch (_activation)
    {
      case ir::Activation::NONE:
        break;
      case ir::Activation::RELU:
        nnfw::cker::train::ReLUGrad(getShape(_output), getBuffer<float>(_output),
                                    getShape(deriv_out), getBuffer<float>(deriv_out),
                                    getShape(_act_deriv_output.get()),
                                    getBuffer<float>(_act_deriv_output.get()));
        deriv_out = _act_deriv_output.get();
        break;
      default:
        throw std::runtime_error("PoolLayer: Unsupported activation type yet");
    }

    // maxpool baackward
    auto arg_max_index = _arg_max_index.get();
    nnfw::cker::train::MaxPool2DGrad(getShape(deriv_out), getBuffer<float>(deriv_out),
                                     getBuffer<int>(arg_max_index), getShape(deriv_in),
                                     getBuffer<float>(deriv_in));
  }
};

} // namespace

PoolLayer::PoolLayer()
  : cpu::ops::PoolLayer(), _deriv_input(nullptr), _deriv_output(nullptr), _kernel(nullptr)
{
  // DO NOTHING
}

void PoolLayer::configure(const IPortableTensor *input, const uint32_t paddingLeft,
                          const uint32_t paddingRight, const uint32_t paddingTop,
                          const uint32_t paddingBottom, const uint32_t strideWidth,
                          const uint32_t strideHeight, const uint32_t kernelWidth,
                          const uint32_t kernelHeight, const ir::Activation activation,
                          IPortableTensor *output, const PoolType op_type,
                          IPortableTensor *deriv_input, const IPortableTensor *deriv_output)
{
  _input = input;
  _output = output;

  _deriv_output = deriv_output;
  _deriv_input = deriv_input;

  // ready to infer
  cpu::ops::PoolLayer::configure(input, paddingLeft, paddingRight, paddingTop, paddingBottom,
                                 strideWidth, strideHeight, kernelWidth, kernelHeight, activation,
                                 output, convert_to_infer(op_type));

  // ready to train
  if (output->data_type() != OperandType::FLOAT32)
  {
    throw std::runtime_error("PoolLayer : Unsupported data type for training");
  }

  switch (op_type)
  {
    case PoolType::kMax:
      _kernel = std::make_unique<MaxPool2D>(paddingLeft, paddingRight, paddingTop, paddingBottom,
                                            strideWidth, strideHeight, kernelWidth, kernelHeight,
                                            activation, output);
      break;
    default:
      throw std::runtime_error("PoolLayer: Unsupported pool type");
      break;
  }
}

void PoolLayer::forward(bool training)
{
  if (training)
  {
    _kernel->forward(_input, _output);
  }
  else
  {
    cpu::ops::PoolLayer::run();
  }
}

void PoolLayer::backward() { _kernel->backward(_deriv_output, _deriv_input); }

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
