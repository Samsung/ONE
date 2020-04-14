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

#ifndef __MLAPSE_BENCHMARK_OBSERVER_H__
#define __MLAPSE_BENCHMARK_OBSERVER_H__

#include <cstdint>
#include <chrono>

namespace mlapse
{

enum Phase : int32_t
{
  // 0 denotes "uninitialized value"
  Warmup = 1,
  Record = 2,
};

Phase uninitialized_phase(void);

enum Notification
{
  PhaseBegin,
  PhaseEnd,
  IterationBegin,
  IterationEnd,
};

template <Notification N> struct NotificationArg;

template <> struct NotificationArg<PhaseBegin>
{
  Phase phase;
  uint32_t count;
};

template <> struct NotificationArg<PhaseEnd>
{
};

template <> struct NotificationArg<IterationBegin>
{
  uint32_t index;
};

template <> struct NotificationArg<IterationEnd>
{
  std::chrono::milliseconds latency;
};

struct BenchmarkObserver
{
  virtual ~BenchmarkObserver() = default;

  virtual void notify(const NotificationArg<PhaseBegin> &arg) = 0;
  virtual void notify(const NotificationArg<PhaseEnd> &arg) = 0;
  virtual void notify(const NotificationArg<IterationBegin> &arg) = 0;
  virtual void notify(const NotificationArg<IterationEnd> &arg) = 0;
};

} // namespace mlapse

#endif // __MLAPSE_BENCHMARK_OBSERVER_H__
