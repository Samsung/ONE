/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ANN/IR/OperandInventory.h"

#include <memory>

using std::make_unique;

namespace ann
{

OperandID OperandInventory::create(const DType &dtype)
{
  uint32_t id = _operands.size();

  auto operand = make_unique<ScalarOperand>();
  operand->dtype(dtype);

  _operands.emplace_back(std::move(operand));

  return OperandID{id};
}

OperandID OperandInventory::create(const DType &dtype, const nncc::core::ADT::tensor::Shape &shape)
{
  uint32_t id = _operands.size();

  auto operand = make_unique<TensorOperand>(shape);
  operand->dtype(dtype);

  _operands.emplace_back(std::move(operand));

  return OperandID{id};
}

Operand *OperandInventory::at(const OperandID &id) { return _operands.at(id.value()).get(); }

const Operand *OperandInventory::at(const OperandID &id) const
{
  return _operands.at(id.value()).get();
}

} // namespace ann
