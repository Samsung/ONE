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
    Edge edge;
    uint32_t session_index;
    uint32_t subg_index;
    std::string backend;
    uint32_t op_index;
    std::string op_name;
    uint32_t op_seq_size; // if this event is for an operation sequence of multiple operations

    // TODO deprecate this. label can be differ by writer. So let the writer decide label.
    std::string label;

    // user-defined data: pairs of (key, value)
    std::vector<std::pair<std::string, std::string>> userData;

    Event(Edge a_edge, const std::string &a_backend, const std::string &a_label)
        : edge(a_edge), backend(a_backend), label(a_label)
    { /* empty */
    }
  };

public:
  EventCollector(EventRecorder *rec) : _rec{rec}
  {
    // DO NOTHING
  }

public:
  void onEvent(const Event &event);

protected:
  EventRecorder *_rec;
};

#endif // __ONERT_UTIL_EVENT_COLLECTOR_H__
