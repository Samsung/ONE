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

#ifndef __NNFW_BENCHMARK_MEMORY_POLLER_H__
#define __NNFW_BENCHMARK_MEMORY_POLLER_H__

#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <list>

#include "Types.h"

namespace benchmark
{

// NOTE. gpu_poll is not necessary on general targets. This is used on the only tv targets.
// TODO finally should be separated from data
// TODO Use ctor() and dtor() instead of start() and end()
class MemoryPoller
{
public:
  MemoryPoller(std::chrono::milliseconds duration = std::chrono::milliseconds(5),
               bool gpu_poll = false);

  virtual ~MemoryPoller()
  {
    _term = true;
    _cond_var_started.notify_all();
    _thread.join();
  }

  bool start(PhaseEnum phase);
  bool end(PhaseEnum phase);
  const std::unordered_map<PhaseEnum, uint32_t> &getRssMap() const { return _rss_map; }
  const std::unordered_map<PhaseEnum, uint32_t> &getHwmMap() const { return _hwm_map; }
  const std::unordered_map<PhaseEnum, uint32_t> &getPssMap() const { return _pss_map; }

private:
  void process();
  bool prepareMemoryPolling();

private:
  std::chrono::milliseconds _duration;
  std::thread _thread;
  std::list<PhaseEnum> _phases;
  std::unordered_map<PhaseEnum, uint32_t> _rss_map;
  std::unordered_map<PhaseEnum, uint32_t> _hwm_map;
  std::unordered_map<PhaseEnum, uint32_t> _pss_map;

  std::mutex _mutex;
  std::mutex _mutex_started;
  std::condition_variable _cond_var_started;

  bool _term;
  bool _run;
  bool _gpu_poll;
  std::string _process_name;
};

} // namespace benchmark

#endif // __NNFW_BENCHMARK_MEMORY_POLLER_H__
