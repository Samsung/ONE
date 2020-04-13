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

#ifndef __LOCO_IR_ALGORITHM_H__
#define __LOCO_IR_ALGORITHM_H__

#include "loco/IR/Node.h"

#include <set>
#include <vector>

namespace loco
{

/**
 * @brief Generate postorder traversal sequence starting from "roots"
 *
 * HOW TO USE
 *
 *  for (auto node : postorder_traversal(...))
 *  {
 *    ... node->do_something() ...
 *  }
 *
 */
std::vector<loco::Node *> postorder_traversal(const std::vector<loco::Node *> &roots);

/**
 * @brief Enumerate all the nodes required to compute "roots"
 */
std::set<loco::Node *> active_nodes(const std::vector<loco::Node *> &roots);

} // namespace loco

#endif // __LOCO_IR_ALGORITHM_H__
