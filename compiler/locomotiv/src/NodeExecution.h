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

#ifndef _LOCOMOTIV_NODEEXECUTION_H_
#define _LOCOMOTIV_NODEEXECUTION_H_

#include <loco.h>

namespace locomotiv
{

struct UnaryFunc
{
  virtual ~UnaryFunc() = default;

  virtual float apply(float) const;
  virtual int32_t apply(int32_t) const;
};

// Q. How to support mixed precision binary operators?
struct BinaryFunc
{
  virtual ~BinaryFunc() = default;

  virtual float apply(float, float) const;
  virtual int32_t apply(int32_t, int32_t) const;
};

/**
 * @brief Helper class for Session, responsible to process one node calculation.
 */
class NodeExecution
{
public:
  /// @brief Run calculation for one unspecified Node
  void run(loco::Node *node);

  static NodeExecution &get()
  {
    static NodeExecution me;
    return me;
  }

private:
  NodeExecution() {}

  template <typename Derived> Derived *as(loco::Node *node)
  {
    return dynamic_cast<Derived *>(node);
  }

// clang-format off
  /**
   * @brief Calculate for one specified node and update its result as NodeData.
   *        Abort program when its ingredients are not ready or not supported.
   *
   * @note Definitions of overloaded execute() are in 'Node/' directory
   */
// clang-format on
#define NODE(Name) void execute(loco::Name *);
#include "Node.lst"
#undef NODE

  void eltwise_unary(loco::Node *node, const UnaryFunc &f);
  void eltwise_binary(loco::Node *node, const BinaryFunc &f);
};

} // namespace locomotiv

#endif // _LOCOMOTIV_NODEEXECUTION_H_
