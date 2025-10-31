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

#include "UnpackLayer.h"

#include "OperationUtils.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

#include <cker/operation/Unpack.h>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Unpack &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(0)};

  const auto rank = _ctx.at(input_index).shape().rank();
  const auto axis = ops::getAxis(rank, node.param().axis);

  assert(rank == 0 || (-rank <= axis && axis < rank));

  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  std::vector<IPortableTensor *> output_tensors;
  for (const auto &output_idx : node.getOutputs())
    output_tensors.emplace_back(_tensor_reg->getPortableTensor(output_idx));

  auto fn = std::make_unique<ops::UnpackLayer>();

  uint32_t axis_resolved = (axis < 0 ? axis + rank : axis);

  fn->configure(input_tensor, axis_resolved, node.param().num, output_tensors);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

UnpackLayer::UnpackLayer() : _input(nullptr), _outputs(), _axis(0), _num_output(0)
{
  // DO NOTHING
}

template <typename T> void UnpackLayer::unpackImpl()
{
  nnfw::cker::UnpackParams op_params;
  op_params.axis = _axis;
  op_params.num_split = _num_output;

  std::vector<nnfw::cker::Shape *> outputDimsPtr;
  std::vector<nnfw::cker::Shape> outputDims;
  outputDimsPtr.reserve(_num_output);
  outputDims.reserve(_num_output);

  for (int32_t i = 0; i < _num_output; i++)
  {
    outputDims.push_back(getShape(_outputs[i]));
    outputDimsPtr.push_back(&outputDims[i]);
  }

  std::vector<T *> outputPtrs;

  for (const auto output : _outputs)
  {
    outputPtrs.emplace_back(getBuffer<T>(output));
  }

  nnfw::cker::Unpack<T>(op_params, getShape(_input), getBuffer<T>(_input), getShape(_outputs[0]),
                        outputPtrs.data());
}

void UnpackLayer::configure(const IPortableTensor *input, uint32_t axis, int32_t num,
                            std::vector<IPortableTensor *> &outputs)
{
  assert(input != nullptr);
  assert(outputs.size() > 0);
  assert(outputs.size() == (size_t)num);

  _input = input;
  _axis = axis;
  _num_output = num;
  _outputs = outputs;
}

void UnpackLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
    unpackImpl<float>();
  else if (_input->data_type() == OperandType::INT32)
    unpackImpl<int32_t>();
  else
    throw std::runtime_error{"Unpack: Unsupported data type"};
}

} // namespace onert::backend::cpu::ops
