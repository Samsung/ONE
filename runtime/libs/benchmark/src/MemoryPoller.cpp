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

#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <iostream>

namespace
{

const std::string proc_status_path("/proc/self/status");
const std::string gpu_memory_path("/sys/kernel/debug/mali0/gpu_memory");
const std::string proc_smaps_path("/proc/self/smaps");

bool isStrNumber(const std::string &s)
{
  return !s.empty() &&
         std::find_if(s.begin(), s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
}

std::vector<std::string> splitLine(std::string line, std::string delimiters = " \n\t")
{
  std::vector<std::string> words;
  size_t prev = 0, pos;

  while ((pos = line.find_first_of(delimiters, prev)) != std::string::npos)
  {
    if (pos > prev)
      words.emplace_back(line.substr(prev, pos - prev));
    prev = pos + 1;
  }

  if (prev < line.length())
    words.emplace_back(line.substr(prev, std::string::npos));

  return words;
}

std::vector<std::string> getValueFromFileStatus(const std::string &file, const std::string &key)
{
  std::ifstream ifs(file);
  assert(ifs.is_open());

  std::string line;
  std::vector<std::string> val;

  bool found = false;
  while (std::getline(ifs, line))
  {
    if (line.find(key) != std::string::npos)
    {
      found = true;
      break;
    }
  }
  ifs.close();

  if (!found)
  {
    // NOTE. the process which uses gpu resources cannot be there yet at the model-load phase.
    // At that time, just return empty.
    return val;
  }

  val = splitLine(line);
  return val;
}

// Because of smaps' structure, returns sum value as uint32_t
uint32_t getSumValueFromFileSmaps(const std::string &file, const std::string &key)
{
  std::ifstream ifs(file);
  assert(ifs.is_open());

  std::string line;
  uint32_t sum = 0;
  while (std::getline(ifs, line))
  {
    if (line.find(key) != std::string::npos)
    {
      // an example by splitLine()
      // `Pss:                   0 kB`
      // val[0]: "Pss:", val[1]: "0" val[2]: "kB"
      auto val = splitLine(line);
      assert(val.size() != 0);
      // SwapPss could show so that check where Pss is at the beginning
      if (val[0].find("Pss") != 0)
      {
        continue;
      }
      sum += std::stoul(val[1]);
    }
  }

  return sum;
}

} // namespace

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
    mem += getGpuMemory();
  }
  if (mem > _rss_map[phase])
    _rss_map[phase] = mem;

  mem = getVmHWM();
  if (_gpu_poll)
  {
    mem += getGpuMemory();
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
      auto gpu_mem = getGpuMemory();
      cur_rss += gpu_mem;
      cur_hwm += gpu_mem;
    }
    uint32_t cur_pss = getPssSum();

    for (auto &phase : _phases)
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
  {
    std::ifstream ifs(proc_status_path);
    if (!ifs.is_open())
    {
      std::cerr << "failed to open " << proc_status_path << std::endl;
      return false;
    }
    ifs.close();
  }

  // (Additionally) GpuMemory
  if (_gpu_poll)
  {
    std::ifstream ifs(gpu_memory_path);
    if (!ifs.is_open())
    {
      std::cerr << "failed to open " << gpu_memory_path << std::endl;
      return false;
    }
    ifs.close();

    // Needs process name
    auto val = getValueFromFileStatus(proc_status_path, "Name");
    assert(val.size() != 0);
    _process_name = val[1];
  }

  // PSS
  {
    std::ifstream ifs(proc_smaps_path);
    if (!ifs.is_open())
    {
      std::cerr << "failed to open " << proc_smaps_path << std::endl;
      return false;
    }
    ifs.close();
  }

  return true;
}

uint32_t MemoryPoller::getVmRSS()
{
  auto val = getValueFromFileStatus(proc_status_path, "VmRSS");
  if (val.size() == 0)
    return 0;
  assert(isStrNumber(val[1]));
  return std::stoul(val[1]);
}

uint32_t MemoryPoller::getVmHWM()
{
  auto val = getValueFromFileStatus(proc_status_path, "VmHWM");
  if (val.size() == 0)
    return 0;
  // key: value
  assert(isStrNumber(val[1]));
  return std::stoul(val[1]);
}

uint32_t MemoryPoller::getGpuMemory()
{
  assert(!_process_name.empty());
  auto val = getValueFromFileStatus(gpu_memory_path, _process_name);
  if (val.size() == 0)
    return 0;
  // process_name -> pid -> gpu_mem -> max_gpu_mem
  assert(isStrNumber(val[2]));
  return std::stoul(val[2]);
}

uint32_t MemoryPoller::getPssSum() { return getSumValueFromFileSmaps(proc_smaps_path, "Pss"); }

} // namespace benchmark
