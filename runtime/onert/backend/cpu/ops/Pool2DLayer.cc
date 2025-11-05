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

#include "Pool2DLayer.h"

#include "../KernelGenerator.h"
#include "../Validator.h"

#include <cker/operation/AveragePool.h>
#include <cker/operation/MaxPool.h>

#include <unordered_map>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Pool2D &) { _supported = true; }

ops::PoolType convertPoolType(ir::operation::Pool2D::PoolType type_ir)
{
  switch (type_ir)
  {
    case ir::operation::Pool2D::PoolType::AVG:
      return ops::PoolType::kAvg;
    case ir::operation::Pool2D::PoolType::MAX:
      return ops::PoolType::kMax;
    default:
      throw std::runtime_error("cpu KernelGenerator : Not supported operation yet");
  }
}

void KernelGenerator::visit(const ir::operation::Pool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Pool2D::Input::INPUT)};

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
  const auto padding =
    ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);
  const auto activation = node.param().activation;

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);

  auto fn = std::make_unique<ops::PoolLayer>();

  fn->configure(ifm_tensor, padding.left, padding.right, padding.top, padding.bottom,
                stride.horizontal, stride.vertical, kw, kh, activation, ofm_tensor,
                convertPoolType(node.param().op_type));

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

namespace
{
template <typename T>
void avgPool2D(const nnfw::cker::PoolParams &params, const IPortableTensor *input,
               IPortableTensor *output)
{
  nnfw::cker::AveragePool<T>(params, getShape(input), getBuffer<T>(input), getShape(output),
                             getBuffer<T>(output));
}

template <typename T>
void maxPool2D(const nnfw::cker::PoolParams &params, const IPortableTensor *input,
               IPortableTensor *output)
{
  nnfw::cker::MaxPool<T>(params, getShape(input), getBuffer<T>(input), getShape(output),
                         getBuffer<T>(output));
}

template <typename T>
std::function<void(const IPortableTensor *, IPortableTensor *)>
generateKernelGeneric(const nnfw::cker::PoolParams &params, PoolType op_type)
{
  if (op_type == PoolType::kAvg)
  {
    return std::bind(&avgPool2D<T>, params, std::placeholders::_1, std::placeholders::_2);
  }
  else if (op_type == PoolType::kMax)
  {
    return std::bind(&maxPool2D<T>, params, std::placeholders::_1, std::placeholders::_2);
  }
  else
  {
    throw std::runtime_error{"Pool: unsupported pool type"};
  }
}
} // namespace

PoolLayer::PoolLayer() : _input(nullptr), _output(nullptr), _kernel()
{
  // DO NOTHING
}

#define POOLING_PARAMETERS                              \
  nnfw::cker::PoolParams op_params;                     \
  op_params.stride_height = strideHeight;               \
  op_params.stride_width = strideWidth;                 \
  op_params.filter_height = kernelHeight;               \
  op_params.filter_width = kernelWidth;                 \
  op_params.padding_values.height = (int8_t)paddingTop; \
  op_params.padding_values.width = (int8_t)paddingLeft; \
  op_params.float_activation_min = 0;                   \
  op_params.float_activation_max = 0;                   \
  op_params.quantized_activation_min = 0;               \
  op_params.quantized_activation_max = 0;

void PoolLayer::configure(const IPortableTensor *input, const uint32_t paddingLeft, const uint32_t,
                          const uint32_t paddingTop, const uint32_t, const uint32_t strideWidth,
                          const uint32_t strideHeight, const uint32_t kernelWidth,
                          const uint32_t kernelHeight, const ir::Activation activation,
                          IPortableTensor *output, const PoolType op_type)
{
  assert(input != nullptr);
  assert(output != nullptr);

  _input = input;
  _output = output;

  POOLING_PARAMETERS

  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
    {
      float output_activation_min = 0;
      float output_activation_max = 0;
      CalculateActivationRange<float>(activation, &output_activation_min, &output_activation_max);
      op_params.float_activation_min = output_activation_min;
      op_params.float_activation_max = output_activation_max;

      _kernel = generateKernelGeneric<float>(op_params, op_type);
      break;
    }
    case OperandType::QUANT_UINT8_ASYMM:
    {
      int32_t output_activation_min = 0;
      int32_t output_activation_max = 0;
      CalculateActivationRangeQuantized(activation, _output, &output_activation_min,
                                        &output_activation_max);
      op_params.quantized_activation_min = output_activation_min;
      op_params.quantized_activation_max = output_activation_max;
      _kernel = generateKernelGeneric<uint8_t>(op_params, op_type);
      break;
    }
    case OperandType::QUANT_INT8_ASYMM:
    {
      int32_t output_activation_min = 0;
      int32_t output_activation_max = 0;
      CalculateActivationRangeQuantized(activation, _output, &output_activation_min,
                                        &output_activation_max);
      op_params.quantized_activation_min = output_activation_min;
      op_params.quantized_activation_max = output_activation_max;
      _kernel = generateKernelGeneric<int8_t>(op_params, op_type);
      break;
    }
    default:
      throw std::runtime_error{"Pool: unsupported data type"};
  }
}

void PoolLayer::run() { _kernel(_input, _output); }

#undef AVGPOOLING_PARAMETERS

} // namespace onert::backend::cpu::ops
