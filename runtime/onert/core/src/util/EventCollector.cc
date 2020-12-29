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

#include "util/EventCollector.h"

// C++ standard libraries
#include <chrono>

// POSIX standard libraries
#include <sys/time.h>
#include <sys/resource.h>

namespace
{

std::string timestamp(void)
{
  auto now = std::chrono::steady_clock::now();
  return std::to_string(
    std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count());
}

class DurationEventBuilder : public EventCollector::EventVisitor
{
public:
  DurationEventBuilder(const std::string &ts) : _ts{ts} {}

  std::unique_ptr<SubgDurationEvent> build(const EventCollector::SubgEvent &evt_collected,
                                           const std::string &ph) const
  {
    auto dur_evt = std::make_unique<SubgDurationEvent>();

    // The following will be set by a child of EventsWriter:
    // dur_evt.name, dur_evt.tid
    dur_evt->ph = ph;
    dur_evt->ts = _ts;
    dur_evt->tracing_ctx = evt_collected.tracing_ctx;

    dur_evt->session_index = evt_collected.session_index;
    dur_evt->subg_index = evt_collected.subg_index;

    dur_evt->args = evt_collected.userData;
    {
      dur_evt->args.emplace_back("session", std::to_string(evt_collected.session_index));
      dur_evt->args.emplace_back("subgraph", std::to_string(evt_collected.subg_index));
    }

    return dur_evt;
  }

  std::unique_ptr<OpSeqDurationEvent> build(const EventCollector::OpSeqEvent &evt_collected,
                                            const std::string &ph) const
  {
    auto dur_evt = std::make_unique<OpSeqDurationEvent>();

    // The following will be set by a child of EventsWriter:
    // dur_evt.name, dur_evt.tid
    dur_evt->ph = ph;
    dur_evt->ts = _ts;
    dur_evt->tracing_ctx = evt_collected.tracing_ctx;

    dur_evt->session_index = evt_collected.session_index;
    dur_evt->subg_index = evt_collected.subg_index;

    dur_evt->backend = evt_collected.backend;
    dur_evt->op_index = evt_collected.op_index;
    dur_evt->op_name = evt_collected.op_name;

    dur_evt->args = evt_collected.userData;
    {
      dur_evt->args.emplace_back("session", std::to_string(evt_collected.session_index));
      dur_evt->args.emplace_back("subgraph", std::to_string(evt_collected.subg_index));
    }

    return dur_evt;
  }

private:
  std::string _ts;
};

#ifdef DEBUG
inline void emit_rusage(EventRecorder *rec, const std::string &ts)
{
  struct rusage ru;

  getrusage(RUSAGE_SELF, &ru);
  {
    CounterEvent evt;

    evt.name = "maxrss";
    evt.ph = "C";
    evt.ts = ts;
    evt.values["value"] = std::to_string(ru.ru_maxrss);

    rec->emit(evt);
  }

  {
    CounterEvent evt;

    evt.name = "minflt";
    evt.ph = "C";
    evt.ts = ts;
    evt.values["value"] = std::to_string(ru.ru_minflt);

    rec->emit(evt);
  }
}
#endif

} // namespace

template <typename EventT> void EventCollector::onEvent(const EventT &event)
{
  auto ts = timestamp();

  DurationEventBuilder builder(ts);

  switch (event.edge)
  {
    case Edge::BEGIN:
    {
      auto duration_evt = builder.build(event, "B");
      _rec->emit(std::move(duration_evt));
      break;
    }
    case Edge::END:
    {
      auto duration_evt = builder.build(event, "E");
      _rec->emit(std::move(duration_evt));
      break;
    }
  }

// TODO: Add resurece measurement(e.g. RSS)
// when ready with low overhead in release build
#ifdef DEBUG
  emit_rusage(_rec, ts);
#endif
}

// template instantiation
template void EventCollector::onEvent<EventCollector::SubgEvent>(const SubgEvent &event);
template void EventCollector::onEvent<EventCollector::OpSeqEvent>(const OpSeqEvent &event);
