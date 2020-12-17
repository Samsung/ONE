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

#include "passes/transformations/LowerConv2D.h"

#include "mir/ops/Conv2DOp.h"
#include "mir/ops/DepthwiseConv2DOp.h"
#include "mir/ops/TransposeOp.h"

namespace nnc
{

static void lowerConv2D(mir::Graph *graph, mir::ops::Conv2DOp *op)
{
  mir::Operation::Output *input = op->getInput(0);
  mir::Operation::Output *kernel = op->getInput(1);

  const std::int32_t in_group_size = kernel->getShape().dim(3);
  const std::int32_t out_group_size = kernel->getShape().dim(0) / op->getNumGroups();

  if (in_group_size == 1 && out_group_size == 1)
  {
    // [O, H, W, I / M] == [M, H, W, 1] -> [H, W, M, 1]
    std::vector<std::size_t> perm{1, 2, 0, 3};
    mir::Operation::Output *new_kernel =
      graph->create<mir::ops::TransposeOp>(kernel, perm)->getOutput(0);
    mir::Conv2DOpAttributes attributes = op->getAttributes();
    attributes.num_groups = 1;
    mir::Operation::Output *new_result =
      graph->create<mir::ops::DepthwiseConv2DOp>(input, new_kernel, attributes)->getOutput(0);
    graph->replaceNode(op, new_result->getNode());
  }
}

LowerConv2D::LowerConv2D() = default;

PassData LowerConv2D::run(PassData data)
{
  auto *graph = static_cast<mir::Graph *>(data);

  // Collect candidate ops before actual transformation because the graph will be changed.
  std::vector<mir::ops::Conv2DOp *> group_conv_ops;
  for (mir::Operation *op : graph->getNodes())
  {
    auto *conv_op = dynamic_cast<mir::ops::Conv2DOp *>(op);
    if (conv_op != nullptr && conv_op->getNumGroups() != 1)
    {
      group_conv_ops.push_back(conv_op);
    }
  }

  for (mir::ops::Conv2DOp *op : group_conv_ops)
  {
    lowerConv2D(graph, op);
  }

  return graph;
}

void LowerConv2D::cleanup() {}

} // namespace nnc
