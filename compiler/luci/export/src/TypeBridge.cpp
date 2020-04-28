/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TypeBridge.h"

#include "CircleExporterUtils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Service/CircleTypeInference.h>
#include <luci/Service/CircleShapeInference.h>

#include <loco/Service/TypeInference.h>
#include <loco/Service/ShapeInference.h>

namespace
{

/**
 * @brief CopySelector will return condition of copy shape/type inference to node
 */
struct CopySelector final : public luci::CircleNodeVisitor<bool>
{
  // return false(don't copy) for nodes that provides shape/type from nature
  bool visit(const luci::CircleInput *) final { return false; }
  bool visit(const luci::CircleConst *) final { return false; }

  // default is copy attributes
  bool visit(const luci::CircleNode *) { return true; }
};

} // namespace

namespace luci
{

loco::TensorShape node_shape(CircleNode *node)
{
  loco::TensorShape shape;

  shape.rank(node->rank());
  for (uint32_t r = 0; r < node->rank(); ++r)
  {
    shape.dim(r) = loco::Dimension(node->dim(r).value());
  }
  return shape;
}

loco::DataType node_dtype(CircleNode *node) { return node->dtype(); }

void copy_shape_dtype(loco::Graph *graph)
{
  /**
   * @note We will iterate all the nodes in the graph to include dangle nodes
   */
  auto nodes = graph->nodes();
  for (uint32_t n = 0; n < nodes->size(); ++n)
  {
    auto node = dynamic_cast<luci::CircleNode *>(nodes->at(n));
    assert(node != nullptr);

    CopySelector cs;
    if (node->accept(&cs))
    {
      assert(loco::shape_known(node));
      assert(loco::dtype_known(node));

      node->dtype(loco::dtype_get(node));

      auto shape = loco::shape_get(node).as<loco::TensorShape>();
      node->rank(shape.rank());
      for (uint32_t r = 0; r < shape.rank(); ++r)
      {
        node->dim(r) = loco::Dimension(shape.dim(r).value());
      }
    }
  }
}

} // namespace luci
