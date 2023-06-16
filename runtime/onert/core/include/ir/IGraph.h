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

#ifndef __ONERT_IR_IGRAPH_H__
#define __ONERT_IR_IGRAPH_H__

#include "ir/Operands.h"
#include "ir/Operations.h"

namespace onert
{
namespace ir
{

struct IGraph
{
  virtual ~IGraph() = default;

  // Accessors
  virtual const OperandIndexSequence &getInputs() const = 0;
  virtual const OperandIndexSequence &getOutputs() const = 0;
  virtual IOIndex getInputIndex(const std::string &name) const = 0;
  virtual IOIndex getOutputIndex(const std::string &name) const = 0;
  virtual const Operands &operands() const = 0;
  virtual const Operations &operations() const = 0;

  // Methods that can change graph
  virtual void changeShape(const OperandIndex &index, const ir::Shape &new_shape) = 0;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_IGRAPH_H__
