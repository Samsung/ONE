/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "util/EventCollectorGlobal.h"

#include <cassert>
#include <fstream>
#include <iostream>

#include "util/ConfigSource.h"

namespace onert
{
namespace util
{

EventCollectorGlobal::EventCollectorGlobal() : _recorder{}, _collector{&_recorder}
{
  // DO NOTHING
}

EventCollectorGlobal::~EventCollectorGlobal()
{
  if (!_recorder.empty())
  {
    try
    {
      // TODO Need better way for saved file path than the hardcoded path
      std::ofstream ofs{"trace.global.json"};
      _recorder.writeToFile(ofs);
    }
    catch (const std::exception &e)
    {
      std::cerr << "E: Fail to record event in EventCollectorGlobal: " << e.what() << std::endl;
    }
  }
}

EventCollectorGlobal &EventCollectorGlobal::get()
{
  static EventCollectorGlobal instance;
  return instance;
}

EventDurationBlock::EventDurationBlock(const std::string &tag) : _tag{tag}
{
  auto &glob = EventCollectorGlobal::get();
  glob.collector().onEvent(EventCollector::Event{EventCollector::Edge::BEGIN, "0", _tag});
}
EventDurationBlock::~EventDurationBlock()
{
  auto &glob = EventCollectorGlobal::get();
  glob.collector().onEvent(EventCollector::Event{EventCollector::Edge::END, "0", _tag});
}

EventDurationManual::EventDurationManual(const std::string &tag) : _tag{tag}, _pair{true} {}

EventDurationManual::~EventDurationManual()
{
  // Check if it has called begin-end pair
  assert(_pair);
}

void EventDurationManual::begin()
{
  _pair = false;
  auto &glob = EventCollectorGlobal::get();
  glob.collector().onEvent(EventCollector::Event{EventCollector::Edge::BEGIN, "0", _tag});
}

void EventDurationManual::end()
{
  assert(!_pair);
  _pair = true;
  auto &glob = EventCollectorGlobal::get();
  glob.collector().onEvent(EventCollector::Event{EventCollector::Edge::END, "0", _tag});
}

} // namespace util
} // namespace onert
