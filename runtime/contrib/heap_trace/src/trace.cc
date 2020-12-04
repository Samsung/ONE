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

#include "trace.h"

#include <memory>

std::unique_ptr<Trace> GlobalTrace(new Trace);

bool Trace::Guard::_is_trace_not_available = true;
thread_local bool Trace::Guard::_is_recursion_detected = false;

Trace::Trace()
{
  if (!_out.is_open())
  {
    _out.open(getLogFileNameFromEnvVariable("HEAP_TRACE_LOG"));
  }

  Guard{}.markTraceAsReady();
}

const char *Trace::getLogFileNameFromEnvVariable(const char *env_variable_name)
{
  return getenv(env_variable_name);
}

void Trace::logAllocationEvent(void *memory_ptr, size_t size_of_allocated_space_in_bytes)
{
  Guard{}.signalizeAboutPossibleRecursion();
  std::lock_guard<std::mutex> guard(_lock);
  _total_allocated_bytes_on_cpu += size_of_allocated_space_in_bytes;
  if (_peak_heap_usage_on_cpu < _total_allocated_bytes_on_cpu - _total_deallocated_bytes_on_cpu)
  {
    _peak_heap_usage_on_cpu = _total_allocated_bytes_on_cpu - _total_deallocated_bytes_on_cpu;
  }
  _memory_in_use_on_cpu[memory_ptr] = size_of_allocated_space_in_bytes;
  Guard{}.signalizeThatDangerOfRecursionHasPassed();
}

void Trace::logDeallocationEvent(void *memory_ptr)
{
  Guard{}.signalizeAboutPossibleRecursion();
  std::lock_guard<std::mutex> guard(_lock);
  auto found_memory_space_description = _memory_in_use_on_cpu.find(memory_ptr);
  if (found_memory_space_description != _memory_in_use_on_cpu.end())
  {
    _total_deallocated_bytes_on_cpu += found_memory_space_description->second;
    _memory_in_use_on_cpu.erase(found_memory_space_description);
  }
  Guard{}.signalizeThatDangerOfRecursionHasPassed();
}

void Trace::logAllocationEvent(cl_mem memory_ptr, size_t size_of_allocated_space_in_bytes)
{
  Guard{}.signalizeAboutPossibleRecursion();
  std::lock_guard<std::mutex> guard(_lock);
  auto found_memory_space_description = _memory_in_use_on_gpu.find(memory_ptr);
  if (found_memory_space_description == _memory_in_use_on_gpu.end())
  {
    _memory_in_use_on_gpu.insert(
      std::make_pair(memory_ptr, MemoryTraits(1, size_of_allocated_space_in_bytes)));
    _total_allocated_bytes_on_gpu += size_of_allocated_space_in_bytes;
    if (_peak_heap_usage_on_gpu < _total_allocated_bytes_on_gpu - _total_deallocated_bytes_on_gpu)
    {
      _peak_heap_usage_on_gpu = _total_allocated_bytes_on_gpu - _total_deallocated_bytes_on_gpu;
    }
  }
  else
  {
    ++found_memory_space_description->second.ref_counter;
  }
  Guard{}.signalizeThatDangerOfRecursionHasPassed();
}

void Trace::logDeallocationEvent(cl_mem memory_ptr)
{
  Guard{}.signalizeAboutPossibleRecursion();
  std::lock_guard<std::mutex> guard(_lock);
  auto found_memory_space_description = _memory_in_use_on_gpu.find(memory_ptr);
  if (found_memory_space_description != _memory_in_use_on_gpu.end())
  {
    if (--found_memory_space_description->second.ref_counter == 0)
    {
      _total_deallocated_bytes_on_gpu += found_memory_space_description->second.size;
      _memory_in_use_on_gpu.erase(found_memory_space_description);
    }
  }
  Guard{}.signalizeThatDangerOfRecursionHasPassed();
}

Trace::~Trace()
{
  Guard{}.markTraceAsNotReady();

  _out << "On CPU - Peak heap usage: " << _peak_heap_usage_on_cpu
       << " B, Total allocated: " << _total_allocated_bytes_on_cpu
       << " B, Total deallocated: " << _total_deallocated_bytes_on_cpu << " B\n";
  _out << "On GPU - Peak mem usage: " << _peak_heap_usage_on_gpu
       << " B, Total allocated: " << _total_allocated_bytes_on_gpu
       << " B, Total deallocated: " << _total_deallocated_bytes_on_gpu << " B\n";
}
