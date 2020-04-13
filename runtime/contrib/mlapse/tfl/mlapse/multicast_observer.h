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

#ifndef __MLAPSE_MULTICAST_OBSERER_H__
#define __MLAPSE_MULTICAST_OBSERER_H__

#include "mlapse/benchmark_observer.h"

#include <memory>
#include <vector>

namespace mlapse
{

class MulticastObserver final : public BenchmarkObserver
{
public:
  MulticastObserver() = default;

public:
  void append(std::unique_ptr<BenchmarkObserver> &&o) { _observers.emplace_back(std::move(o)); }

public:
  void notify(const NotificationArg<PhaseBegin> &arg) final
  {
    for (const auto &o : _observers)
    {
      o->notify(arg);
    }
  }

  void notify(const NotificationArg<PhaseEnd> &arg) final
  {
    for (const auto &o : _observers)
    {
      o->notify(arg);
    }
  }

  void notify(const NotificationArg<IterationBegin> &arg) final
  {
    for (const auto &o : _observers)
    {
      o->notify(arg);
    }
  }

  void notify(const NotificationArg<IterationEnd> &arg) final
  {
    for (const auto &o : _observers)
    {
      o->notify(arg);
    }
  }

private:
  std::vector<std::unique_ptr<BenchmarkObserver>> _observers;
};

} // namespace mlapse

#endif // __MLAPSE_MULTICAST_OBSERER_H__
