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
#include "ir/Model.h"

#include <unordered_map>
#include <mutex>

namespace onert::util
{

/**
 * @brief Class to maintain information about profiling per session
 */
class TracingCtx
{
public:
  /**
   * @brief Create and store unique session id managed by this class
   * @note  This constructor can be called by multiple session running in parallely.
   */
  TracingCtx(void) { decideSessionID(); }

  uint32_t getSessionId() const { return _session_id; }

  /**
   * @brief Return true if more than 1 session exist
   *
   * @note  This method is NOT thread-safe. Call this in thread-safe situation.
   */
  bool hasMultipleSessions() const { return _next_session_id > 1; }

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

    _session_id = _next_session_id++;
  }

private:
  std::unordered_map<const ir::Graph *, ir::SubgraphIndex> _subgraph_indices;
  uint32_t _session_id;
  static inline std::mutex _session_id_mutex;
  static inline uint32_t _next_session_id = 0;
};

} // namespace onert::util

#endif // __ONERT_UTIL_TRACING_CTX_H__
