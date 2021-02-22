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

#include "moco/Pass/Passes/ResolveSquaredDifference.h"

#include <moco/IR/TFDialect.h>
#include <moco/IR/TFNodes.h>
#include <moco/IR/TFNodeVisitor.h>
#include <moco/IR/TFNodeImpl.h>

#include <loco/IR/NodeShape.h>
#include <loco/Service/ShapeInference.h>

namespace
{

bool decompose_sqdiff(moco::TFSquaredDifference *node)
{
  /**
   * @note This will decompose TFSquaredDifference node into TFSub and TFMul
   *
   *       Before
   *                 A --- TFSquaredDifference -- C
   *                 B --/
   *       After
   *                 A --- TFSquaredDifference --
   *                 B --/
   *                 A --- TFSub == TFMul -- C
   *                 B --/
   *       Where
   *                 A : x of TFSquaredDifference
   *                 B : y of TFSquaredDifference
   *                 C : a node that uses TFSquaredDifference as an input
   *                 TFSquaredDifference is disconnected from C
   *                 A and B are drawn multiple times to simplify the diagram
   */

  auto node_A = node->x();
  auto node_B = node->y();

  auto sub_node = node->graph()->nodes()->create<moco::TFSub>();
  auto mul_node = node->graph()->nodes()->create<moco::TFMul>();

  // update connections
  sub_node->x(node_A);
  sub_node->y(node_B);
  mul_node->x(sub_node);
  mul_node->y(sub_node);

  // replace node
  replace(node).with(mul_node);

  return true;
}

} // namespace

namespace moco
{

bool ResolveSquaredDifference::run(loco::Graph *graph)
{
  auto active_nodes = loco::active_nodes(loco::output_nodes(graph));
  bool changed = false;

  for (auto node : active_nodes)
  {
    if (node->dialect() == TFDialect::get())
    {
      auto tf_node = dynamic_cast<moco::TFSquaredDifference *>(node);
      if (tf_node != nullptr)
      {
        if (decompose_sqdiff(tf_node))
          changed = true;
      }
    }
  }

  return changed;
}

} // namespace moco
