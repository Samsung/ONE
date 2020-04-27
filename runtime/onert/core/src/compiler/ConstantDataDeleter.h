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

#ifndef __ONERT_COMPILER_CACHED_DATA_DELETER_H__
#define __ONERT_COMPILER_CACHED_DATA_DELETER_H__

#include "ir/Index.h"
#include "ir/OperationVisitor.h"
#include "ir/OpSequences.h"
#include "ir/Operands.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{

class ConstantDataDeleter : public ir::OperationVisitor
{
public:
  ConstantDataDeleter(ir::Operands &operands) : _operands(operands)
  {
    // DO NOTHING
  }

  virtual ~ConstantDataDeleter() = default;

public:
  void run()
  {
    _operands.iterate(
        [&](const ir::OperandIndex &ind, const ir::Operand &) { deleteConstantData(ind); });
  }

  void run(const ir::OpSequence &op_seq)
  {
    for (const auto &e : op_seq.operations())
    {
      const auto &node = *(e.node);
      node.accept(*this);
    }
  }

  // NOTE: Almost layers that have the big size constants are conv and fc.
  void visit(const ir::operation::Conv2D &node) override
  {
    using ir::operation::Conv2D;
    const auto ker_index{node.getInputs().at(Conv2D::Input::KERNEL)};
    const auto bias_index{node.getInputs().at(Conv2D::Input::BIAS)};
    deleteConstantData(ker_index);
    deleteConstantData(bias_index);
  }

  void visit(const ir::operation::DepthwiseConv2D &node) override
  {
    using ir::operation::DepthwiseConv2D;
    const auto ker_index{node.getInputs().at(DepthwiseConv2D::Input::KERNEL)};
    const auto bias_index{node.getInputs().at(DepthwiseConv2D::Input::BIAS)};
    deleteConstantData(ker_index);
    deleteConstantData(bias_index);
  }

  void visit(const ir::operation::FullyConnected &node) override
  {
    using ir::operation::FullyConnected;
    const auto weight_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
    const auto bias_index{node.getInputs().at(FullyConnected::Input::BIAS)};
    deleteConstantData(weight_index);
    deleteConstantData(bias_index);
  }

private:
  void deleteConstantData(const ir::OperandIndex &ind)
  {
    auto &obj = _operands.at(ind);
    if (obj.isConstant())
    {
      assert(obj.data() != nullptr);
      obj.deleteData();
    }
  }

private:
  ir::Operands &_operands;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_CACHED_DATA_DELETER_H__
