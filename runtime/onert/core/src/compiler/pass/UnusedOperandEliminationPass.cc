/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Pass.h"

#include "UnusedOperandEliminationPass.h"
#include "ir/Index.h"
#include "util/Set.h"
#include "ir/Graph.h"

/**
 * @file  UnusedOperandEliminationPass.cc
 * @brief This file contains UnusedOperandEliminationPass class implementation
 */

namespace onert
{
namespace compiler
{
namespace pass
{

void UnusedOperandEliminationPass::run()
{
  util::Set<ir::OperandIndex> used;

  _graph.operations().iterate([&](const ir::OperationIndex &, const ir::IOperation &node) {
    for (auto &&ind : (node.getInputs() + node.getOutputs()) | ir::Remove::UNDEFINED)
    {
      used.add(ind);
    }
  });

  // Graph's inputs/outputs are always considered as used
  for (auto &&ind : (_graph.getInputs() + _graph.getOutputs()) | ir::Remove::UNDEFINED)
  {
    used.add(ind);
  }

  _graph.operands().iterate([&](const ir::OperandIndex &ind, const ir::Operand &) {
    if (!used.contains(ind))
    {
      VERBOSE() << "Remove unused operand " << ind << std::endl;
      _graph.operands().remove(ind);
    }
  });
}

} // namespace pass
} // namespace compiler
} // namespace onert
