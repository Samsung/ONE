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

#ifndef LUCI_INTERPRETER_LOADER_KERNELBUILDER_H
#define LUCI_INTERPRETER_LOADER_KERNELBUILDER_H

#include "loader/KernelBuilderHelper.h"

#include "core/Kernel.h"
#include "core/RuntimeGraph.h"

#include <luci/IR/CircleNodeVisitor.h>

#include <memory>
#include <unordered_map>

namespace luci_interpreter
{

class KernelBuilder : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>,
                      public KernelBuilderHelper
{
public:
  KernelBuilder(
    const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
    const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
    : KernelBuilderHelper(graph_to_runtime_graph, node_to_tensor)
  {
  }

  std::unique_ptr<Kernel> visit(const luci::CircleNode *node) override;

  std::unique_ptr<Kernel> visit(const luci::CircleOutput *node) override;
  std::unique_ptr<Kernel> visit(const luci::CirclePRelu *node) override;
  std::unique_ptr<Kernel> visit(const luci::CirclePack *node) override;
  std::unique_ptr<Kernel> visit(const luci::CirclePad *node) override;
  std::unique_ptr<Kernel> visit(const luci::CirclePadV2 *node) override;
  std::unique_ptr<Kernel> visit(const luci::CirclePow *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleRelu *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleRelu6 *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleReshape *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleResizeBilinear *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleResizeNearestNeighbor *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleReverseV2 *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleRsqrt *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSlice *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSoftmax *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSpaceToBatchND *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSpaceToDepth *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSplit *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSqrt *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSquaredDifference *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSqueeze *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleStridedSlice *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSub *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleTanh *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleTranspose *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleTransposeConv *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleUnpack *node) override;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_KERNELBUILDER_H
