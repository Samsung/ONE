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

#include "MergeConcatNodesPass.h"
#include "Dialect/IR/TFLNodes.h"

#include <oops/InternalExn.h>

#include <vector>

namespace
{

bool canMerge(locoex::TFLConcatenation *node1, locoex::TFLConcatenation *node2)
{
  if (node1->fusedActivationFunction() != node2->fusedActivationFunction())
    return false;

  if (node1->axis() != node2->axis())
    return false;

  switch (node1->fusedActivationFunction())
  {
    case locoex::FusedActFunc::NONE:
    case locoex::FusedActFunc::RELU:
    case locoex::FusedActFunc::RELU6:
      return true;

      // case locoex::FusedActFunc::TANH:
      //   return false;

    default:
      INTERNAL_EXN_V("Unknown FusedActFunc", oops::to_uint32(node1->fusedActivationFunction()));
  }
}

/**
 * @brief Collect all the inputs of newly created TFLConcatenation nodes
 *
 *        in:0 -------------------------------\
 *        in:1 ---- TFLConcatenation:0 -------- TFLConcatenation:3 --- C
 *                  (axis = 0, NONE)            (axis = 0, NONE)
 *        in:2 ---/                           /
 *        in:3 ---- TFLConcatenation:1 ------/
 *                  (axis = 1, NONE)        /
 *        in:4 ---/                        /
 *        in:5 ---- TFLConcatenation:2 ---/
 *                  (axis = 0, RELU)
 *        in:6 ---/
 *
 *        For exmaple, if graph is like above, dfs(TFLConcatenation:3) will
 *        return [in:0, in:1, in:2, TFLConcatenation:1, TFLConcatenation:2]
 *
 *        TFLConcatenation:0 can be merged to TFLConcatenation:3,
 *        because axis and fusedActivationFunction are same.
 *        It means that [in:1, in:2] will be linked as inputs of new TFLConcatenation.
 *
 *        However, TFLConcatenation:1 and TFLConcatenation:2 cannot be merged to
 *        TFLConcatenation:3 because axis and fusedActivationFunction of each are different.
 *        So [in:3, in:4, in:5, in:6] will not be linked as inputs of new TFLConcatenation
 *        and [TFLConcatenation:1, TFLConcatenation:2] will be linked instead.
 *
 *        Therefore, inputs of newly created TFLConcatenation node for merging
 *        TFLConcatenation:3 will be [in:0, in:1, in:2, TFLConcatenation:1, TFLConcatenation:2]
 *        and dfs(TFLConcatenation:3) will return it.
 *
 *
 * @note The input nodes should be traversed by LRV,
 *       which is from left to right (input:0 --> input:N)
 */
std::vector<loco::Node *> dfs(locoex::TFLConcatenation *root)
{
  std::vector<loco::Node *> res;

  for (uint32_t i = 0; i < root->numValues(); ++i)
  {
    auto input = dynamic_cast<locoex::TFLConcatenation *>(root->values(i));
    if (input != nullptr && canMerge(input, root))
    {
      auto children = dfs(input);
      for (auto child : children)
        res.push_back(child);
    }
    else
    {
      res.push_back(root->values(i));
    }
  }

  return res;
}

} // namespace

namespace exo
{

/**
 * @brief Merge TFLConcatenate nodes whose axis and fusedActivationFunction are same
 *
 * [Before]
 *    in:0 -------------------------------\
 *    in:1 ---- TFLConcatenation:0 -------- TFLConcatenation:3 --- C
 *              (axis = 0, NONE)            (axis = 0, NONE)
 *    in:2 ---/                           /
 *    in:3 ---- TFLConcatenation:1 ------/
 *              (axis = 1, NONE)        /
 *    in:4 ---/                        /
 *    in:5 ---- TFLConcatenation:2 ---/
 *              (axis = 0, RELU)
 *    in:6 ---/
 *
 * [After]
 *    in:0 -------------------------------\
 *    in:1 -------------------------------- TFLConcatenation:4 --- C
 *                                          (axis = 0, NONE)
 *    in:2 -------------------------------/
 *    in:3 ---- TFLConcatenation:1 ------/
 *              (axis = 1, NONE)        /
 *    in:4 ---/                        /
 *    in:5 ---- TFLConcatenation:2 ---/
 *              (axis = 0, RELU)
 *    in:6 ---/
 *
 *
 *    in:1 ---- TFLConcatenation:0 ----
 *              (axis = 0, NONE)
 *    in:2 ---/
 *
 *
 *         ---- TFLConcatenation:3 ----
 *              (axis = 0, NONE)
 */
bool MergeConcatNodesPass::run(loco::Graph *graph)
{
  // Let's enumerate nodes required to compute output nodes
  auto active_nodes = loco::active_nodes(loco::output_nodes(graph));

  // Find TFLConcatenation nodes which have another TFLConcatenation nodes
  // as inputs, with same axis and same fusedActivationFunction
  std::vector<locoex::TFLConcatenation *> candidates;
  for (auto node : active_nodes)
  {
    if (auto concat = dynamic_cast<locoex::TFLConcatenation *>(node))
    {
      for (uint32_t i = 0; i < concat->numValues(); ++i)
      {
        auto input = dynamic_cast<locoex::TFLConcatenation *>(concat->values(i));
        if (input != nullptr && canMerge(input, concat))
        {
          candidates.push_back(concat);
          break;
        }
      }
    }
  }

  // Merge multiple TFLConcatenation nodes as one TFLConcatenation node
  for (auto node : candidates)
  {
    auto inputs = dfs(node);

    auto new_concat = graph->nodes()->create<locoex::TFLConcatenation>(inputs.size());
    new_concat->axis(node->axis());
    new_concat->fusedActivationFunction(node->fusedActivationFunction());

    for (uint32_t i = 0; i < inputs.size(); ++i)
      new_concat->values(i, inputs.at(i));

    loco::replace(node).with(new_concat);
    for (uint32_t i = 0; i < node->numValues(); ++i)
      node->values(i, nullptr);
  }

  return candidates.size() > 0;
}

} // namespace exo
