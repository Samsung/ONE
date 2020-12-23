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

#ifndef __LOCO_IR_USE_H__
#define __LOCO_IR_USE_H__

#include "loco/IR/Node.forward.h"

namespace loco
{

/**
 * @brief The edge between a node definition and its user.
 *
 * Note that this "Use" denotes **one** edge between a node and its users,
 * and thus there are unique node and user for each Use.
 *
 * There will be multiple "Use" edges for the same node if there are multiple
 * users.
 *
 * This class design is heavily inspired from "Use" class in LLVM.
 */
class Use final
{
public:
  /**
   * @brief Construct Use with its user
   * @note user SHOULD BE set on construction.
   */
  Use(Node *user) : _user{user}
  {
    // DO NOTHING
  }

  Use(const Use &) = delete;
  Use(Use &&) = delete;

  ~Use()
  {
    // Unlink itself from the node
    node(nullptr);
  }

public:
  Node *node(void) const { return _node; }
  void node(Node *node);

public:
  Node *user(void) const { return _user; }

private:
  Node *_node{nullptr};
  Node *_user{nullptr};
};

} // namespace loco

#endif // __LOCO_IR_USE_H__
