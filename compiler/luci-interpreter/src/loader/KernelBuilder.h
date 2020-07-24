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

#include "core/Kernel.h"
#include "core/RuntimeGraph.h"

#include <luci/IR/CircleNodeVisitor.h>

#include <memory>
#include <vector>
#include <unordered_map>

namespace luci_interpreter
{

class KernelBuilder : public luci::CircleNodeVisitor<std::unique_ptr<Kernel>>
{
public:
  KernelBuilder(
      const std::unordered_map<const loco::Graph *, RuntimeGraph *> &graph_to_runtime_graph,
      const std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor)
      : _graph_to_runtime_graph(graph_to_runtime_graph), _node_to_tensor(node_to_tensor)
  {
  }

  std::unique_ptr<Kernel> visit(const luci::CircleAdd *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleArgMax *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleAveragePool2D *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleConcatenation *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleConv2D *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleConst *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleDepthwiseConv2D *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleElu *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleFullyConnected *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleIf *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleL2Normalize *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleL2Pool2D *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleLeakyRelu *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleLocalResponseNormalization *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleLogistic *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleInput *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleMaxPool2D *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleMean *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleMul *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleOutput *node) override;
  std::unique_ptr<Kernel> visit(const luci::CirclePad *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleReshape *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleReverseV2 *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSlice *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSoftmax *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSpaceToDepth *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSplit *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleStridedSlice *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleSqueeze *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleTranspose *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleTransposeConv *node) override;
  std::unique_ptr<Kernel> visit(const luci::CircleUnpack *node) override;

private:
  const Tensor *getInputTensor(const loco::Node *node) const;

  const Tensor *getOptionalInputTensor(const loco::Node *node) const;

  Tensor *getOutputTensor(const loco::Node *node) const;

  std::vector<Tensor *> getOutputTensors(const std::vector<const loco::Node *> &nodes) const;

  RuntimeGraph *getRuntimeGraph(const loco::Graph *graph) const;

private:
  const std::unordered_map<const loco::Graph *, RuntimeGraph *> &_graph_to_runtime_graph;
  const std::unordered_map<const loco::Node *, Tensor *> &_node_to_tensor;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_KERNELBUILDER_H
