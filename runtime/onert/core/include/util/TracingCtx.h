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
#include "ir/Subgraphs.h"

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
  /**
   * @brief Create and store unique session id managed by this class
   *        Note that this constructor can be called by multiple sessions running in parallely.
   *        Use this constructor only when there is only one subgraph in a model.
   */
  TracingCtx(const ir::Graph *primary_subgraph)
  {
    decideSessionID();
    _subgraph_indices.emplace(primary_subgraph, 0);
  }

  /**
   * @brief Create and store unique session id managed by this class
   *        Note that this constructor can be called by multiple sessions running in parallely.
   */
  TracingCtx(const onert::ir::Subgraphs *subgraphs)
  {
    assert(subgraphs);

    decideSessionID();

    auto count = subgraphs->count();
    for (size_t i = 0; i < count; i++)
      _subgraph_indices.emplace(subgraphs->at(onert::ir::SubgraphIndex(i)).get(), i);
  }

  uint32_t getSessionId() const { return _session_id; }

  /**
   * @brief Set subgraph index of a graph
   */
  void setSubgraphIndex(const ir::Graph *g, uint32_t index) { _subgraph_indices.emplace(g, index); }

  /**
   * @brief Get subgraph index of a graph.
   */
  ir::SubgraphIndex getSubgraphIndex(const ir::Graph *g) const { return _subgraph_indices.at(g); }

private:
  void decideSessionID()
  {
    std::unique_lock<std::mutex> lock{_session_id_mutex};

    static uint32_t next_session_id = 0;
    _session_id = next_session_id++;
  }

private:
  std::unordered_map<const ir::Graph *, ir::SubgraphIndex> _subgraph_indices;
  uint32_t _session_id;
  static std::mutex _session_id_mutex;
};

} // namespace util
} // namespace onert

#endif // __ONERT_UTIL_TRACING_CTX_H__
