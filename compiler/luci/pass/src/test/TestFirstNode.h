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

#ifndef __LUCI_PASS_TEST_FIRST_NODE_H__
#define __LUCI_PASS_TEST_FIRST_NODE_H__

#include <luci/IR/CircleNodes.h>

#include <loco.h>

namespace luci
{
namespace test
{

template <class T> T *first_node(loco::Graph *g)
{
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto target_node = dynamic_cast<T *>(node);
    if (target_node != nullptr)
      return target_node;
  }
  return nullptr;
}

} // namespace test
} // namespace luci

#endif // __LUCI_PASS_TEST_FIRST_NODE_H__
