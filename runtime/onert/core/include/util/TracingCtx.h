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

#ifndef __ONERT_UTIL_TRACING_CTX_H__
#define __ONERT_UTIL_TRACING_CTX_H__

#include "ir/Graph.h"
#include "ir/Index.h"

#include <unordered_map>
#include <mutex>

namespace onert
{
namespace util
{

/**
 * @brief Class to maintain information about profiling per session
 */
class TracingCtx
{
public:
  // create and store unique session id managed by this class
  void makeSessionId()
  {
    std::unique_lock<std::mutex> lock{_session_id_mutex};

    static uint32_t session_id = 0;
    _session_id = session_id++;

    lock.unlock();
  }

  uint32_t getSessionId() { return _session_id; }

  /**
   * @brief Set subgraph index of a graph
   */
  void setSubgraphIndex(const ir::Graph *g, uint32_t index) { _subgraph_indices.emplace(g, index); }

  /**
   * @brief Get subgraph index of a graph. If there is no such key (g),
   *        ir::SubgraphIndex(0) will be returned.
   */
  ir::SubgraphIndex getSubgraphIndex(const ir::Graph *g) const
  {
    auto find = _subgraph_indices.find(g);

    return find == _subgraph_indices.end() ? getDefaultSubgraphIndex() : find->second;
  }

  /**
   * @brief Get the Default Subgraph Index object. Call this when there is no subgraph index
   *        information or when TracingCtx object is nullptr
   *
   * @return const ir::SubgraphIndex
   */
  static const ir::SubgraphIndex getDefaultSubgraphIndex() { return ir::SubgraphIndex{0}; }

private:
  std::unordered_map<const ir::Graph *, ir::SubgraphIndex> _subgraph_indices;
  uint32_t _session_id;
  std::mutex _session_id_mutex;
};

} // namespace util
} // namespace onert

#endif // __ONERT_UTIL_TRACING_CTX_H__
