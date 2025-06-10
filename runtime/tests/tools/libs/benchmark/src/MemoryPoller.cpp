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

#include "benchmark/MemoryPoller.h"
#include "benchmark/Types.h"
#include "benchmark/MemoryInfo.h"

#include <vector>
#include <stdexcept>
#include <cassert>
#include <iostream>

namespace benchmark
{

MemoryPoller::MemoryPoller(std::chrono::milliseconds duration, bool gpu_poll)
  : _duration(duration), _run(false), _term(false), _gpu_poll(gpu_poll)
{
  if (prepareMemoryPolling() == false)
    throw std::runtime_error("failed to prepare memory pooling");

  _thread = std::thread{&MemoryPoller::process, this};
}

bool MemoryPoller::start(PhaseEnum phase)
{
  if (std::find(_phases.begin(), _phases.end(), phase) != _phases.end())
  {
    std::cerr << getPhaseString(phase) << " is already processing/processed..." << std::endl;
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(_mutex);
    _phases.emplace_back(phase);
    _rss_map[phase] = 0;
    _hwm_map[phase] = 0;
    _pss_map[phase] = 0;
  }

  _run = true;
  _cond_var_started.notify_all();
  return true;
}

bool MemoryPoller::end(PhaseEnum phase)
{
  if (std::find(_phases.begin(), _phases.end(), phase) == _phases.end())
  {
    std::cerr << getPhaseString(phase) << " is not started..." << std::endl;
    return false;
  }

  uint32_t mem = 0;
  bool stop = false;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _phases.remove(phase);
    stop = (_phases.size() == 0);
  }

  mem = getVmRSS();
  if (_gpu_poll)
  {
    mem += getGpuMemory(_process_name);
  }
  if (mem > _rss_map[phase])
    _rss_map[phase] = mem;

  mem = getVmHWM();
  if (_gpu_poll)
  {
    mem += getGpuMemory(_process_name);
  }
  _hwm_map[phase] = mem;

  mem = getPssSum();
  if (mem > _pss_map[phase])
    _pss_map[phase] = mem;

  if (stop)
  {
    _run = false;
    _cond_var_started.notify_all();
  }

  return true;
}

void MemoryPoller::process()
{
  std::unique_lock<std::mutex> lock_started(_mutex_started);
  while (true)
  {
    _cond_var_started.wait(lock_started, [&]() { return _run || _term; });
    if (_term)
      break;

    std::unique_lock<std::mutex> lock(_mutex);

    uint32_t cur_rss = getVmRSS();
    uint32_t cur_hwm = getVmHWM();
    if (_gpu_poll)
    {
      auto gpu_mem = getGpuMemory(_process_name);
      cur_rss += gpu_mem;
      cur_hwm += gpu_mem;
    }
    uint32_t cur_pss = getPssSum();

    for (const auto &phase : _phases)
    {
      auto &rss = _rss_map.at(phase);
      if (rss < cur_rss)
        rss = cur_rss;
      // hwm is gradually increasing
      auto &hwm = _hwm_map.at(phase);
      hwm = cur_hwm;
      auto &pss = _pss_map.at(phase);
      if (pss < cur_pss)
        pss = cur_pss;
    }

    lock.unlock();

    std::this_thread::sleep_for(std::chrono::milliseconds(_duration));
  }
}

bool MemoryPoller::prepareMemoryPolling()
{
  // VmRSS
  if (!prepareVmRSS())
  {
    std::cerr << "failed to prepare parsing vmrss" << std::endl;
    return false;
  }

  // (Additionally) GpuMemory
  if (_gpu_poll)
  {
    if (!prepareGpuMemory())
    {
      std::cerr << "failed to prepare parsing gpu memory" << std::endl;
      return false;
    }

    // Needs process name
    _process_name = getProcessName();
  }

  // PSS
  if (!preparePssSum())
  {
    std::cerr << "failed to prepare parsing pss sum" << std::endl;
    return false;
  }

  return true;
}

} // namespace benchmark
