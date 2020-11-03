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

#ifndef __ONERT_UTIL_EVENT_COLLECTOR_H__
#define __ONERT_UTIL_EVENT_COLLECTOR_H__

#include "util/EventRecorder.h"
#include "util/TracingCtx.h"

#include <vector>
#include <utility>
#include <string>

class EventCollector
{
public:
  enum class Edge
  {
    BEGIN,
    END
  };

  struct Event
  {
    const onert::util::TracingCtx *tracing_ctx;

    Edge edge;
    uint32_t session_index;
    uint32_t subg_index;

    // user-defined data: pairs of (key, value)
    std::vector<std::pair<std::string, std::string>> userData;

  protected:
    Event(const onert::util::TracingCtx *a_tracing_ctx, Edge a_edge, uint32_t a_subg_index)
      : tracing_ctx(a_tracing_ctx), edge(a_edge), session_index(tracing_ctx->getSessionId()),
        subg_index(a_subg_index)
    { /* empty */
    }

    virtual ~Event() = default;
  };

  struct SubgEvent : public Event
  {
    // constructor for subgraph start and end event
    SubgEvent(const onert::util::TracingCtx *a_tracing_ctx, Edge a_edge, uint32_t a_subg_index)
      : Event(a_tracing_ctx, a_edge, a_subg_index)
    { /* empty */
    }
  };

  struct OpSeqEvent : public Event
  {
    std::string backend;
    uint32_t op_index;
    std::string op_name;
    uint32_t op_seq_size; // if this event is for an operation sequence of multiple operations

    OpSeqEvent(const onert::util::TracingCtx *a_tracing_ctx, Edge a_edge, uint32_t a_subg_index,
               const std::string a_backend, uint32_t a_op_index, const std::string a_op_name,
               uint32_t a_op_seq_size)
      : Event(a_tracing_ctx, a_edge, a_subg_index)
    {
      backend.assign(a_backend);
      op_index = a_op_index;
      op_name.assign(a_op_name);
      op_seq_size = a_op_seq_size;
    }
  };

public:
  EventCollector(EventRecorder *rec) : _rec{rec}
  {
    // DO NOTHING
  }

public:
  template <typename EventT> void onEvent(const EventT &event);

protected:
  EventRecorder *_rec;
};

#endif // __ONERT_UTIL_EVENT_COLLECTOR_H__
