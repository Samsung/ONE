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

#include "SquaredDifferenceLayer.h"

#include "OperationUtils.h"

#include <cker/operation/SqDiff.h>

#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void KernelGenerator::visit(const ir::operation::SquaredDifference &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::RHS)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  auto fn = std::make_unique<ops::SqDiffLayer>();

  fn->configure(lhs_tensor, rhs_tensor, ofm_tensor);
  _return_fn = std::move(fn);
}

void Validator::visit(const ir::operation::SquaredDifference &) { _supported = true; }

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

SqDiffLayer::SqDiffLayer() : _input1(nullptr), _input2(nullptr), _output(nullptr)
{
  // DO NOTHING
}

void SqDiffLayer::SqDiffFloat32()
{
  nnfw::cker::SqDiff(getShape(_input1), getBuffer<float>(_input1), getShape(_input2),
                     getBuffer<float>(_input2), getShape(_output), getBuffer<float>(_output));
}

void SqDiffLayer::configure(const IPortableTensor *input1, const IPortableTensor *input2,
                            IPortableTensor *output)
{
  _input1 = input1;
  _input2 = input2;
  _output = output;
}

void SqDiffLayer::run()
{
  if (_input1->data_type() == OperandType::FLOAT32)
  {
    SqDiffFloat32();
  }
  else
  {
    throw std::runtime_error{"SquaredDiff: unsupported data type"};
  }
}
} // namespace onert::backend::cpu::ops
