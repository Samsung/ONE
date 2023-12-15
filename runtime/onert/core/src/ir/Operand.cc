/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ir/Operand.h"

namespace onert
{
namespace ir
{

size_t Operand::operandSize(void) const
{
  const uint32_t ranks = shape().rank();
  int32_t elements = 1;

  for (uint32_t rank = 0; rank < ranks; rank++)
  {
    elements *= shape().dim(rank);
  }

  DataType type = typeInfo().type();
  size_t element_size = sizeOfDataType(type);

  // Value of type is matched with OperandCode enum in NeuralNetworks.h
  return element_size * elements;
}

void Operand::insertUse(const OperationIndex &idx) { _uses.insert(idx); }

void Operand::removeUse(const OperationIndex &idx) { _uses.remove(idx); }

void Operand::clearUses() { _uses.clear(); }

void Operand::setDef(const OperationIndex &idx) { _def = idx; }

void Operand::unsetDef() { _def = OperationIndex{}; }

void Operand::clearDefUse()
{
  unsetDef();
  clearUses();
}

} // namespace ir
} // namespace onert
