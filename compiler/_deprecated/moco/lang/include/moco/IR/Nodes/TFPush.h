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

#ifndef __MOCO_IR_TFPUSH_H__
#define __MOCO_IR_TFPUSH_H__

#include "moco/IR/TFNodeDecl.h"

#include <loco.h>

namespace moco
{

/**
 * @brief Make a value visible to user
 *
 * @note  TFPush is a virtual node that does not corresponds to real TensorFlow node
 *        Why this node is introduced:
 *        - Any TensorFlow Nodes can be an output.
 *        - So let any TFNode type can provide OutputIndex using Annotation.
 *        - Problem comes when in transformation, output node can be replaced.
 *        - This causes that OutputIndex Annotation should be copied to new node.
 *        - This makes every transformation in any Dialect code change.
 *        - And even worse, this makes every new transformation follow this rule.
 *        - Which is not good.
 *        - Thus, like loco Canonical does, follow loco::Push.
 */
class TFPush /* to user */ final : public FixedArityNode<1, TFNodeImpl<TFOpcode::TFPush>>
{
public:
  TFPush() = default;

public:
  loco::Node *from(void) const { return at(0)->node(); }
  void from(loco::Node *node) { at(0)->node(node); }

public:
  void index(const loco::GraphOutputIndex &index);

  /**
   * @brief Get associated output index
   *
   * The behavior of this method is undefined when "index" is not set before.
   *
   * NOTE This method intentionally returns "GraphOutputIndex" instead of "const GraphOutputIndex &"
   *      not to expose the internal implementation details.
   */
  loco::GraphOutputIndex index(void) const;

  /**
   * @brief Check whether index is initialized
   *
   * NOTE "indexed" method does not validate whether index is in a valid range
   */
  bool indexed(void) const { return _index != -1; }

  /**
   * @brief  Reset output index
   */
  void index_reset(void) { _index = -1; }

private:
  int64_t _index = -1; // Uninitialized
};

/// @brief Find a TFPush node with a given output index
TFPush *push_node(loco::Graph *g, const loco::GraphOutputIndex &index);

} // namespace moco

#endif // __MOCO_IR_TFPUSH_H__
