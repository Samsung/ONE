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

#include "SplitLayer.h"

#include "OperationUtils.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

#include <cker/operation/Split.h>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Split &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Split &node)
{
  const auto num_splits = node.param().num_splits;
  assert(num_splits == static_cast<int>(node.getOutputs().size()));

  const auto input_idx{node.getInputs().at(ir::operation::Split::Input::INPUT)};
  const auto axis_idx{node.getInputs().at(ir::operation::Split::Input::AXIS)};

  auto in_tensor = _tensor_reg->getPortableTensor(input_idx);
  auto axis_tensor = _tensor_reg->getPortableTensor(axis_idx);

  std::vector<IPortableTensor *> out_tensors;
  for (const auto &output_idx : node.getOutputs())
    out_tensors.emplace_back(_tensor_reg->getPortableTensor(output_idx));

  auto fn = std::make_unique<ops::SplitLayer>();

  fn->configure(in_tensor, axis_tensor, num_splits, out_tensors);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

SplitLayer::SplitLayer() : _input(nullptr), _axis(nullptr), _num_splits(0), _outputs()
{
  // DO NOTHING
}

template <typename T> void SplitLayer::split(void)
{
  nnfw::cker::SplitParams op_params;
  if (_axis->total_size() != sizeof(int32_t))
  {
    throw std::runtime_error("ArgMinMax: wrong shape of axis");
  }
  auto axis = *getBuffer<int32_t>(_axis);
  if (axis < 0)
  {
    axis += _input->getShape().rank();
  }
  op_params.axis = axis;
  op_params.num_split = _num_splits;

  std::vector<T *> outputPtrs;

  for (const auto output : _outputs)
  {
    assert(output->total_size() == sizeOfData(output->data_type(), output->getShape().dims()));
    outputPtrs.emplace_back(getBuffer<T>(output));
  }

  assert(_input->total_size() == sizeOfData(_input->data_type(), _input->getShape().dims()));
  nnfw::cker::Split<T>(op_params, getShape(_input), getBuffer<T>(_input), getShape(_outputs[0]),
                       outputPtrs.data());
}

void SplitLayer::configure(const IPortableTensor *input, const IPortableTensor *axis,
                           uint16_t num_splits, std::vector<IPortableTensor *> &outputs)
{
  assert(input != nullptr);

  _num_splits = num_splits;
  _input = input;
  _axis = axis;
  _outputs = outputs;
}

void SplitLayer::run()
{
  if (_input->data_type() == OperandType::FLOAT32)
  {
    split<float>();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    split<uint8_t>();
  }
  else if (_input->data_type() == OperandType::INT32)
  {
    split<int32_t>();
  }
  else if (_input->data_type() == OperandType::INT64)
  {
    split<int64_t>();
  }
  else
  {
    throw std::runtime_error{"Split: unsupported input type"};
  }
}

} // namespace onert::backend::cpu::ops
