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

#include "OddOutputPass.h"

#include "ir/Graph.h"
#include "ir/operation/Permute.h"
#include "util/logging.h"
#include "util/Utils.h"

namespace onert
{
namespace compiler
{
namespace pass
{

void OddOutputPass::run()
{
  auto &outputs = _graph.getOutputs();

  VERBOSE(OddOutputPass) << "Case 1 : An operand which is a model output and a model input"
                         << std::endl;
  for (auto &ind : outputs)
  {
    if (_graph.getInputs().contains(ind))
    {
      auto permute_output_ind = insertPermute(ind);
      // Update the output to be newly added operand
      _graph.getOutputs().replace(ind, permute_output_ind);
    }
  }

  VERBOSE(OddOutputPass) << "Case 2 : Two or more duplicated outputs" << std::endl;
  std::unordered_set<ir::OperandIndex> occurence;
  for (auto &ind : outputs)
  {
    auto &obj = _graph.operands().at(ind);
    if (occurence.count(ind) == 0)
    {
      occurence.insert(ind);
      continue;
    }

    // Panic when it is const, it must have been handled earlier in another pass
    UNUSED_RELEASE(obj);
    assert(!obj.isConstant());

    auto permute_output_ind = insertPermute(ind);
    ind = permute_output_ind; // Replace output index to fix output duplication
  }
}

ir::OperandIndex OddOutputPass::insertPermute(ir::OperandIndex ind)
{
  auto &obj = _graph.operands().at(ind);
  auto output_ind = _graph.addOperand(obj.shape(), obj.typeInfo());
  auto &output_obj = _graph.operands().at(output_ind);

  using ir::operation::Permute;
  auto permute_obj = std::make_unique<Permute>(ind, output_ind, Permute::Type::COPY);
  auto permute_ind = _graph.operations().push(std::move(permute_obj));

  output_obj.setDef(permute_ind);
  obj.insertUse(permute_ind);

  VERBOSE(OddOutputPass) << "Permute Op inserted for a constant output, node index : "
                         << permute_ind << std::endl;
  VERBOSE(OddOutputPass) << "  - Input (original) Operand : " << ind << std::endl;
  VERBOSE(OddOutputPass) << "  - Output(inserted) Operand : " << output_ind << std::endl;

  return output_ind;
}

} // namespace pass
} // namespace compiler
} // namespace onert
