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

#ifndef __ONERT_UTIL_EVENT_RECORDER_H__
#define __ONERT_UTIL_EVENT_RECORDER_H__

#include <map>
#include <memory>
#include <mutex>

#include <vector>

struct Event
{
  std::string name;
  std::string tid;
  std::string ph;                                        /* REQUIRED */
  std::string ts;                                        /* REQUIRED */
  std::vector<std::pair<std::string, std::string>> args; // user-defined data: pairs of (key, value)
};

struct DurationEvent : public Event
{
  // TO BE FILLED
};

struct CounterEvent : public Event
{
  std::map<std::string, std::string> values;
};

//
// Record Event as Chrome Trace Event File Format
//
// Refrence: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit
//
class EventRecorder
{
public:
  EventRecorder() = default;

public:
  void emit(const DurationEvent &evt);
  void emit(const CounterEvent &evt);

public:
  bool empty() { return _duration_events.empty() && _counter_events.empty(); }
  const std::vector<DurationEvent> &duration_events() const { return _duration_events; }
  const std::vector<CounterEvent> &counter_events() const { return _counter_events; }

private:
  std::mutex _mu;
  std::vector<DurationEvent> _duration_events;
  std::vector<CounterEvent> _counter_events;
};

#endif // __ONERT_UTIL_EVENT_RECORDER_H__
