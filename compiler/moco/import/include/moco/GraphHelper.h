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

#ifndef __MOCO_GRAPH_HELPER_H__
#define __MOCO_GRAPH_HELPER_H__

#include <moco/IR/TFNode.h>

#include <loco.h>

namespace moco
{

/**
 * @brief  find_node_byname() will return a node with type T with given name
 *         in graph g
 *
 * @note   this uses simple linear search, but can speed up with better
 *         algorithms when needed.
 */
template <typename T> T *find_node_byname(loco::Graph *g, const char *name)
{
  T *first_node = nullptr;
  loco::Graph::NodeContext *nodes = g->nodes();
  uint32_t count = nodes->size();

  for (uint32_t i = 0; i < count; ++i)
  {
    auto tfnode = dynamic_cast<TFNode *>(nodes->at(i));
    if (tfnode != nullptr)
    {
      if (tfnode->name() == name)
      {
        // if tfnode is NOT type of T then return will be nullptr
        // this is OK cause the user wanted to get type T but it isn't
        return dynamic_cast<T *>(tfnode);
      }
    }
  }

  return nullptr;
}

} // namespace moco

#endif // __MOCO_GRAPH_HELPER_H__
