/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "loader/KernelBuilderVisitor.h"
#include "kernels/Unpack.h"

namespace
{

// TODO move to Utils
template <typename CircleNodeOut>
std::vector<const loco::Node *> collectOutputNodes(const luci::CircleNode *node)
{
  std::vector<const CircleNodeOut *> output_nodes;
  for (const loco::Node *loco_node : loco::succs(node))
  {
    output_nodes.push_back(loco::must_cast<const CircleNodeOut *>(loco_node));
  }
  std::sort(output_nodes.begin(), output_nodes.end(),
            [](const CircleNodeOut *node1, const CircleNodeOut *node2) {
              return node1->index() < node2->index();
            });
  return {output_nodes.cbegin(), output_nodes.cend()};
}

} // namespace

namespace luci_interpreter
{

std::unique_ptr<Kernel> KernelBuilderVisitor::visit(const luci::CircleUnpack *node)
{
  auto output_nodes = collectOutputNodes<luci::CircleUnpackOut>(node);
  assert(node->arity() == 1);
  assert(output_nodes.size() == static_cast<size_t>(node->num()));

  const Tensor *input = getInputTensor(node->value());
  std::vector<Tensor *> outputs = getOutputTensors(output_nodes);

  UnpackParams params{};
  params.axis = node->axis();

  // NOTE 'num' attribute is ignored.
  return std::make_unique<kernels::Unpack>(input, std::move(outputs), params);
}

} // namespace luci_interpreter