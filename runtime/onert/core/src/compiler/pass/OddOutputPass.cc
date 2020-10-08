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

namespace onert
{
namespace compiler
{
namespace pass
{

void OddOutputPass::run()
{
  auto &outputs = _graph.getOutputs();

  // Case 1 : An operand which is a model output and a model input
  for (auto &ind : outputs)
  {
    if (_graph.getInputs().contains(ind))
    {
      auto &obj = _graph.operands().at(ind);

      auto permute_output_ind = _graph.addOperand(obj.shape(), obj.typeInfo());
      auto &permute_output_obj = _graph.operands().at(permute_output_ind);

      using ir::operation::Permute;
      auto permute_obj = std::make_unique<Permute>(ind, permute_output_ind, Permute::Type::COPY);
      auto permute_ind = _graph.operations().push(std::move(permute_obj));

      permute_output_obj.setDef(permute_ind);
      obj.insertUse(permute_ind);

      VERBOSE(OddOutputPass) << "Permute Op inserted for a constant output, node index : "
                             << permute_ind << std::endl;
      VERBOSE(OddOutputPass) << "  - Input (original) Operand : " << ind << std::endl;
      VERBOSE(OddOutputPass) << "  - Output(inserted) Operand : " << permute_output_ind
                             << std::endl;

      // Update the output to be newly added operand
      _graph.getOutputs().replace(ind, permute_output_ind);
    }
  }
}

} // namespace pass
} // namespace compiler
} // namespace onert
