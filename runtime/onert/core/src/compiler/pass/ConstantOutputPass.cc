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

#include "ConstantOutputPass.h"

#include "ir/Graph.h"
#include "ir/operation/Permute.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{
namespace pass
{

void ConstantOutputPass::callback(const ir::OperandIndex &ind, ir::Operand &obj)
{
  if (!_graph.getOutputs().contains(ind) || !obj.isConstant())
    return;

  auto permute_input_ind = _graph.addOperand(obj.shape(), obj.typeInfo());
  auto &permute_input_obj = _graph.operands().at(permute_input_ind);

  // Move the const data
  permute_input_obj.data(obj.shareData());
  obj.releaseData();
  obj.info().setAsNonConst();

  using ir::operation::Permute;
  auto permute_obj = std::make_unique<Permute>(permute_input_ind, ind, Permute::Type::COPY);
  auto permute_ind = _graph.operations().push(std::move(permute_obj));

  permute_input_obj.insertUse(permute_ind);
  obj.setDef(permute_ind);

  // Make the operations that uses this operand to use the generated operand
  auto orig_uses = obj.getUses();
  for (auto &&use : orig_uses)
  {
    permute_input_obj.insertUse(use);
    obj.removeUse(use);
    _graph.operations().at(use).replaceInputs(ind, permute_input_ind);
  }

  VERBOSE(ConstantOutputPass) << "Permute Op inserted for a constant ouput, node index : "
                              << permute_ind << std::endl;
  VERBOSE(ConstantOutputPass) << "  - Input (inserted) Operand : " << permute_input_ind
                              << std::endl;
  VERBOSE(ConstantOutputPass) << "  - Output(original) Operand : " << ind << std::endl;
}

} // namespace pass
} // namespace compiler
} // namespace onert
