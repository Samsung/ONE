/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "passes/optimizations/OptimizationUtils.h"

namespace nnc
{
namespace opt_util
{

void swapAdjacent(mir::Graph *g, mir::Operation *top, mir::Operation *bottom)
{
  assert(top->getNumInputs() == bottom->getNumInputs() && top->getNumInputs() == 1 &&
         top->getNumInputs() == top->getNumOutputs() &&
         top->getNumInputs() == bottom->getNumOutputs() && "incompatible ops");
  const auto &ins = top->getInputs();
  std::vector<mir::Operation::Output *> prods;
  prods.reserve(top->getNumInputs());
  for (mir::Operation::Output *in : ins)
  {
    prods.emplace_back(in);
  }
  mir::Operation *new_bottom = g->copyOpWithInputs(bottom, prods);
  prods.clear();
  prods.reserve(new_bottom->getNumOutputs());
  for (mir::Operation::Output &out : new_bottom->getOutputs())
  {
    prods.emplace_back(&out);
  }
  mir::Operation *new_top = g->copyOpWithInputs(top, prods);
  g->replaceNode(bottom, new_top);
  g->replaceNode(top, new_bottom);
}

// TODO: this function and it's usages should be removed, after DCE optimization will be implemented
void removeNodeIfUnused(mir::Graph *g, mir::Operation *op)
{
  if (op->getOutput(0)->getUses().empty())
    g->removeNode(op);
}

} // namespace opt_util
} // namespace nnc
