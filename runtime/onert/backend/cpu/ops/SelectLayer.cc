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

#include "SelectLayer.h"

#include "OperationUtils.h"

#include <cker/operation/Select.h>

#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void KernelGenerator::visit(const ir::operation::Select &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto condition_index{node.getInputs().at(ir::operation::Select::Input::CONDITION)};
  const auto true_index{node.getInputs().at(ir::operation::Select::Input::INPUT_TRUE)};
  const auto false_index{node.getInputs().at(ir::operation::Select::Input::INPUT_FALSE)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto condition_tensor = _tensor_reg->getPortableTensor(condition_index);
  auto true_tensor = _tensor_reg->getPortableTensor(true_index);
  auto false_tensor = _tensor_reg->getPortableTensor(false_index);

  auto fn = std::make_unique<ops::SelectLayer>();

  fn->configure(condition_tensor, true_tensor, false_tensor, output_tensor);

  _return_fn = std::move(fn);
}

void Validator::visit(const ir::operation::Select &) { _supported = true; }

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

SelectLayer::SelectLayer()
  : _cond(nullptr), _input_true(nullptr), _input_false(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void SelectLayer::configure(const IPortableTensor *cond, const IPortableTensor *input_true,
                            const IPortableTensor *input_false, IPortableTensor *output)
{
  _cond = cond;
  _input_true = input_true;
  _input_false = input_false;
  _output = output;
}

void SelectLayer::run()
{

#define KERNEL_SELECT(type, op)                                                     \
  nnfw::cker::op(getShape(_cond), getBuffer<uint8_t>(_cond), getShape(_input_true), \
                 getBuffer<type>(_input_true), getShape(_input_false),              \
                 getBuffer<type>(_input_false), getShape(_output), getBuffer<type>(_output));

#define KERNEL_SWITCH(type, op)                                  \
  switch (type)                                                  \
  {                                                              \
    break;                                                       \
    case OperandType::FLOAT32:                                   \
      KERNEL_SELECT(float, op);                                  \
      break;                                                     \
    default:                                                     \
      throw std::runtime_error{"Select: unsupported data type"}; \
  }

  auto input_type = _input_true->data_type();
  bool require_broadcast =
    !HaveSameShapes(_input_true, _cond) || !HaveSameShapes(_input_false, _cond);
  bool rank_one_select = ((_input_true->getShape().rank() == 1) && !require_broadcast);

  if (rank_one_select)
  {
    KERNEL_SWITCH(input_type, RankOneSelect);
  }
  else if (require_broadcast)
  {
    KERNEL_SWITCH(input_type, BroadcastSelect4DSlow);
  }
  else
  {
    KERNEL_SWITCH(input_type, Select);
  }
}

} // namespace onert::backend::cpu::ops
