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

#ifndef __MLAPSE_BENCHMARK_RUNNER_H__
#define __MLAPSE_BENCHMARK_RUNNER_H__

#include "mlapse/benchmark_observer.h"

#include <tflite/Session.h>

#include <chrono>
#include <map>

namespace mlapse
{

class BenchmarkRunner final
{
public:
  BenchmarkRunner(uint32_t warmup_count, uint32_t record_count)
  {
    _count[Warmup] = warmup_count;
    _count[Record] = record_count;
  }

public:
  void attach(BenchmarkObserver *observer);

public:
  void run(nnfw::tflite::Session *sess) const;

public:
  void run(const std::unique_ptr<nnfw::tflite::Session> &sess) const { run(sess.get()); }

private:
  void notify(const NotificationArg<PhaseBegin> &arg) const;
  void notify(const NotificationArg<PhaseEnd> &arg) const;
  void notify(const NotificationArg<IterationBegin> &arg) const;
  void notify(const NotificationArg<IterationEnd> &arg) const;

private:
  std::map<Phase, uint32_t> _count;

private:
  BenchmarkObserver *_observer = nullptr;
};

} // namespace mlapse

#endif // __MLAPSE_BENCHMARK_RUNNER_H__
