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

#include <logo/ResolveDuplicateReshapePass.h>

#include <loco.h>

#include <cassert>

namespace
{

/// @return  true when 'node' and its input node are both FixedReshapes
bool is_duplicate_reshape(loco::Node *node)
{
  auto node_as_reshape = dynamic_cast<loco::FixedReshape *>(node);

  if (!node_as_reshape)
    return false;

  auto input_as_reshape = dynamic_cast<loco::FixedReshape *>(node_as_reshape->input());

  if (!input_as_reshape)
    return false;

  return true;
}

/**
 * @brief  Remap reshape's input to its input's input, i.e. bypass input reshape
 *
 * Before:
 *
 *   In ----- FixedReshape_1 ----- [Out_1]*
 *                        \
 *                         ------- FixedReshape_2 --- [Out_2]*
 *                                 ('reshape' arg)
 *
 * After:
 *
 *   In ----- FixedReshape_1 ----- [Out_1]*
 *    \
 *     --------------------------- FixedReshape_2 --- [Out_2]*
 *
 * Note: In case of no Out_1, FixedReshape_1 becomes dead node.
 *       Out_1 can be another FixedReshape as well, which would be resolved in
 *       another occurance of this transform pass.
 */
void remap_input(loco::FixedReshape *reshape)
{
  auto input_reshape = loco::must_cast<loco::FixedReshape *>(reshape->input());

  auto volume = [](loco::FixedReshape *node) {
    uint32_t vol = 1;
    for (uint32_t axis = 0; axis < node->rank(); ++axis)
    {
      assert(node->dim(axis).known());
      vol *= node->dim(axis).value();
    }
    return vol;
  };

  // Volume mismatch between duplicate reshapes is pointless
  assert(volume(reshape) == volume(input_reshape));

  // Set node's input as input's input, i.e. bypass
  reshape->input(input_reshape->input());
}

} // namespace

namespace logo
{

bool ResolveDuplicateReshapePass::run(loco::Graph *graph)
{
  auto outputs = loco::output_nodes(graph);

  bool changed = false;
  for (auto node : loco::postorder_traversal(outputs))
  {
    if (is_duplicate_reshape(node))
    {
      auto node_as_reshape = loco::must_cast<loco::FixedReshape *>(node);

      remap_input(node_as_reshape);

      changed = true;
    }
  }

  return changed;
}

} // namespace logo
