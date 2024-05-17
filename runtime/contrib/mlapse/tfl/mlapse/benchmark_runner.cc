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

#include "mlapse/benchmark_runner.h"

// From 'nnfw_lib_benchmark'
#include <benchmark/Accumulator.h>

// From C++ Standard Library
#include <cassert>
#include <stdexcept>

namespace mlapse
{
void BenchmarkRunner::attach(BenchmarkObserver *observer)
{
  assert(_observer == nullptr);
  _observer = observer;
}

void BenchmarkRunner::run(nnfw::tflite::Session *sess) const
{
  for (auto phase : {Warmup, Record})
  {
    uint32_t const count = _count.at(phase);

    // Notify when each phase begins
    {
      NotificationArg<PhaseBegin> arg;

      arg.phase = phase;
      arg.count = count;

      notify(arg);
    }

    for (uint32_t n = 0; n < count; ++n)
    {
      std::chrono::milliseconds elapsed(0);

      sess->prepare();

      // Notify when each iteration begins
      {
        NotificationArg<IterationBegin> arg;

        arg.index = n;

        notify(arg);
      };

      benchmark::measure(elapsed) << [&](void) {
        if (!sess->run())
        {
          throw std::runtime_error{"run failed"};
        }
      };

      // Notify when each iteration ends
      {
        NotificationArg<IterationEnd> arg;

        arg.latency = elapsed;

        notify(arg);
      };

      sess->teardown();
    }

    // Notify when each phase ends
    {
      NotificationArg<PhaseEnd> arg;

      notify(arg);
    }
  }
}

void BenchmarkRunner::notify(const NotificationArg<PhaseBegin> &arg) const
{
  if (_observer)
  {
    _observer->notify(arg);
  }
}

void BenchmarkRunner::notify(const NotificationArg<PhaseEnd> &arg) const
{
  if (_observer)
  {
    _observer->notify(arg);
  }
}

void BenchmarkRunner::notify(const NotificationArg<IterationBegin> &arg) const
{
  if (_observer)
  {
    _observer->notify(arg);
  }
}

void BenchmarkRunner::notify(const NotificationArg<IterationEnd> &arg) const
{
  if (_observer)
  {
    _observer->notify(arg);
  }
}

} // namespace mlapse
