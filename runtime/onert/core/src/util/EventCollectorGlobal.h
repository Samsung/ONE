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

#ifndef __ONERT_UTIL_EVENT_COLLECTOR_GLOBAL_H__
#define __ONERT_UTIL_EVENT_COLLECTOR_GLOBAL_H__

#include "util/EventRecorder.h"
#include "util/EventCollector.h"

namespace onert
{
namespace util
{

/**
 * @brief Singleton class for event collection from anywhere in code
 *
 */
class EventCollectorGlobal
{
public:
  /**
   * @brief Get the singleton object of this class
   *
   * @return EventCollectorGlobal& Singleton object
   */
  static EventCollectorGlobal &get();

public:
  /**
   * @brief Getter for event collector object
   *
   * @return EventCollector& Collector object
   */
  EventCollector &collector() { return _collector; }

private:
  EventCollectorGlobal();
  ~EventCollectorGlobal();

private:
  EventRecorder _recorder;
  EventCollector _collector;
};

/**
 * @brief Helper class for emitting duration event which is handled automatically with ctor/dtor
 *
 */
class EventDurationBlock
{
public:
  /**
   * @brief Raise a duration event with type of BEGIN
   *
   * @param tag A label for the duration event
   */
  EventDurationBlock(const std::string &tag);
  /**
   * @brief Raise a duration event with type of END
   *
   */
  ~EventDurationBlock();

private:
  std::string _tag;
};

/**
 * @brief Helper class for emitting duration event which is handled manually
 *
 *        Usage:
 *        {
 *          ...
 *          EventDurationManual duration("some tag");
 *          duration.begin();
 *          ...
 *          ... // Code for duration
 *          ...
 *          duration.end();
 *        }
 *
 */
class EventDurationManual
{
public:
  /**
   * @brief Construct a new Event Duration Manual object
   *
   * @param tag A label for the duration object
   */
  EventDurationManual(const std::string &tag);
  /**
   * @brief Destroy the Event Duration Manual object
   *
   */
  ~EventDurationManual();

  /**
   * @brief Raise a duration event with type of BEGIN
   *
   */
  void begin();
  /**
   * @brief Raise a duration event with type of END
   *
   */
  void end();

private:
  std::string _tag;
  bool _pair;
};

} // namespace util
} // namespace onert

/**
 * Helper Macro Definitions
 *
 * HOW TO USE
 *
 * void f(args)
 * {
 *   EVENT_DURATION_FUNCTION();
 *   ...
 *   if(cond)
 *   {
 *     EVENT_DURATION_REGION("if branch");
 *     ...
 *   }
 *   ...
 * }
 */

#define EVENT_DURATION_FUNCTION() \
  ::onert::util::EventDurationBlock __event_duration__##__LINE__ { __FUNCTION__ }

#define EVENT_DURATION_REGION(tag) \
  ::onert::util::EventDurationBlock __event_duration__##__LINE__ { tag }

#endif // __ONERT_UTIL_EVENT_COLLECTOR_GLOBAL_H__
