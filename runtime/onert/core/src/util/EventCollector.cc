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

class DurationEventBuilder
{
public:
  DurationEventBuilder(const std::string &ts) : _ts{ts} {}

  DurationEvent build(const EventCollector::Event &evt_collected, const std::string &ph) const
  {
    DurationEvent evt;

    evt.name = evt_collected.label;
    evt.tid = evt_collected.backend;
    evt.ph = ph;
    evt.ts = _ts;

    evt.args = evt_collected.userData;

    return evt;
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

void EventCollector::onEvent(const Event &event)
{
  auto ts = timestamp();

  switch (event.edge)
  {
    case Edge::BEGIN:
      _rec->emit(DurationEventBuilder(ts).build(event, "B"));
      break;

    case Edge::END:
      _rec->emit(DurationEventBuilder(ts).build(event, "E"));
      break;
  }

// TODO: Add resurece measurement(e.g. RSS)
// when ready with low overhead in release build
#ifdef DEBUG
  emit_rusage(_rec, ts);
#endif
}
