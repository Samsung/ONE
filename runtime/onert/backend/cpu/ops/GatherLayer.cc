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

#include "GatherLayer.h"

#include "OperationUtils.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

#include <cker/operation/Gather.h>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Gather &node)
{
  using ir::operation::Gather;

  const auto input_index{node.getInputs().at(Gather::Input::INPUT)};
  const auto input_node = &_graph.operands().at(input_index);

  _supported = false;

  if (input_node->typeInfo().type() == ir::DataType::QUANT_GGML_Q4_0)
    return;

  _supported = true;
}

void KernelGenerator::visit(const ir::operation::Gather &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto indices_tensor = _tensor_reg->getPortableTensor(indices_index);

  const auto rank = _ctx.at(input_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis);

  auto fn = std::make_unique<ops::GatherLayer>();

  fn->configure(input_tensor, indices_tensor, output_tensor, axis);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

void GatherLayer::configure(const IPortableTensor *input, const IPortableTensor *indices,
                            IPortableTensor *output, int32_t axis)
{
  _input = input;
  _indices = indices;
  _axis = axis;
  _output = output;
}

template <typename InputType> void GatherLayer::runByInputType()
{
  using OutputType = InputType;
  nnfw::cker::GatherParams op_params;
  op_params.axis = _axis;

  switch (_indices->data_type())
  {
    case OperandType::INT32:
    {
      using IndicesType = int32_t;

      nnfw::cker::Gather<InputType, IndicesType>(
        op_params, getShape(_input), getBuffer<InputType>(_input), getShape(_indices),
        getBuffer<IndicesType>(_indices), getShape(_output), getBuffer<OutputType>(_output));
      break;
    }
    case OperandType::INT64:
    {
      using IndicesType = int64_t;

      nnfw::cker::Gather<InputType, IndicesType>(
        op_params, getShape(_input), getBuffer<InputType>(_input), getShape(_indices),
        getBuffer<IndicesType>(_indices), getShape(_output), getBuffer<OutputType>(_output));
      break;
    }
    default:
      throw std::runtime_error("Gather: unsupported indices data type");
  }
}

void GatherLayer::run()
{
  switch (_input->data_type())
  {
    case OperandType::FLOAT32:
      runByInputType<float>();
      break;
    case OperandType::QUANT_UINT8_ASYMM:
      runByInputType<uint8_t>();
      break;
    case OperandType::INT32:
      runByInputType<int32_t>();
      break;
    case OperandType::BOOL8:
      runByInputType<bool>();
      break;
    default:
      throw std::runtime_error("Gather: unsupported input data type");
  }
}

} // namespace onert::backend::cpu::ops
