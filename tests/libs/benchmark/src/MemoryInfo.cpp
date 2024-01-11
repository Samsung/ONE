/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "benchmark/MemoryInfo.h"

#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>
#include <sys/time.h>
#include <sys/resource.h>

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

bool prepareVmRSS() { return std::ifstream(proc_status_path).is_open(); }

bool prepareVmHWM() { return std::ifstream(proc_status_path).is_open(); }

bool prepareGpuMemory() { return std::ifstream(gpu_memory_path).is_open(); }

bool preparePssSum() { return std::ifstream(proc_smaps_path).is_open(); }

uint32_t getVmRSS()
{
  auto val = getValueFromFileStatus(proc_status_path, "VmRSS");
  if (val.size() == 0)
    return 0;
  assert(isStrNumber(val[1]));
  return std::stoul(val[1]);
}

uint32_t getVmHWM()
{
  auto val = getValueFromFileStatus(proc_status_path, "VmHWM");
  if (val.size() == 0)
    return 0;
  // key: value
  assert(isStrNumber(val[1]));
  return std::stoul(val[1]);
}

uint32_t getGpuMemory(const std::string &process_name)
{
  assert(!process_name.empty());
  auto val = getValueFromFileStatus(gpu_memory_path, process_name);
  if (val.size() == 0)
    return 0;
  // process_name -> pid -> gpu_mem -> max_gpu_mem
  assert(isStrNumber(val[2]));
  return std::stoul(val[2]);
}

uint32_t getPssSum() { return getSumValueFromFileSmaps(proc_smaps_path, "Pss"); }

std::string getProcessName()
{
  auto val = getValueFromFileStatus(proc_status_path, "Name");
  assert(val.size() >= 2);
  return val[1];
}

} // namespace benchmark
