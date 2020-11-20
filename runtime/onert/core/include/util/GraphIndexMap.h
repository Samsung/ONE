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

#ifndef __ONERT_UTIL_GRAPH_INDEX_MAP_H__
#define __ONERT_UTIL_GRAPH_INDEX_MAP_H__

#include "ir/Graph.h"
#include "ir/Index.h"

#include <unordered_map>

namespace onert
{
namespace util
{

/**
 * @brief Class to maintain ir::Graph* and its subgraph index
 *        Subgraph index comes normally from circle or tflite file.
 *        When there is no concept of subgraph index like nnapi API, getIndex() always returns 0.
 */
class GraphIndexMap
{
public:
  GraphIndexMap() = default;

public:
  static GraphIndexMap &get()
  {
    static GraphIndexMap me;
    return me;
  }

  /**
   * @brief Set subgraph index of a graph
   */
  void setIndex(const ir::Graph *g, uint32_t index) { _map.emplace(g, index); }

  /**
   * @brief Get subgraph index of a graph. If there is no such key (g),
   *        ir::SubgraphIndex(0) will be returned.
   */
  ir::SubgraphIndex getIndex(const ir::Graph *g) const
  {
    constexpr int DEFAULT_SUBG_INDEX = 0;

    auto find = _map.find(g);

    return find == _map.end() ? ir::SubgraphIndex{DEFAULT_SUBG_INDEX} : find->second;
  }

  /**
   * @brief Copies [graph, subgraph index] to [graph', subgraph index]
   *        This is useful when source graph is cloned into another graph. Both graphs will have
   *        same subgraph index.
   */
  void copyIndex(const ir::Graph *src, const ir::Graph *dst)
  {
    auto index = getIndex(src);
    _map.emplace(dst, index);
  }

private:
  std::unordered_map<const ir::Graph *, ir::SubgraphIndex> _map;
};

} // namespace util
} // namespace onert

#endif // __ONERT_UTIL_GRAPH_INDEX_MAP_H__
