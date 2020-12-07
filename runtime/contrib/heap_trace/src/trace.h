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

#ifndef TRACE_H_
#define TRACE_H_

#include <CL/cl.h>

#include <unordered_map>
#include <fstream>
#include <mutex>

class Trace
{
  struct MemoryTraits
  {
    size_t ref_counter;
    size_t size;

    MemoryTraits(size_t init_counter_value, size_t size_of_allocated_memory)
      : ref_counter(init_counter_value), size(size_of_allocated_memory)
    {
    }
  };

public:
  class Guard
  {
    friend class Trace;

  public:
    bool isActive() { return _is_trace_not_available || _is_recursion_detected; }

  private:
    void markTraceAsReady() { _is_trace_not_available = false; }
    void markTraceAsNotReady() { _is_trace_not_available = true; }
    void signalizeAboutPossibleRecursion() { _is_recursion_detected = true; }
    void signalizeThatDangerOfRecursionHasPassed() { _is_recursion_detected = false; }

  private:
    static bool _is_trace_not_available;
    static thread_local bool _is_recursion_detected;
  };

public:
  Trace();
  Trace(const Trace &) = delete;
  const Trace &operator=(const Trace &) = delete;

  void logAllocationEvent(void *memory_ptr, size_t size_of_allocated_space_in_bytes);
  void logAllocationEvent(cl_mem memory_ptr, size_t size_of_allocated_space_in_bytes);
  void logDeallocationEvent(void *memory_ptr);
  void logDeallocationEvent(cl_mem memory_ptr);

  ~Trace();

private:
  const char *getLogFileNameFromEnvVariable(const char *env_variable_name);

private:
  std::mutex _lock;
  std::ofstream _out;
  size_t _total_allocated_bytes_on_cpu = 0;
  size_t _total_deallocated_bytes_on_cpu = 0;
  size_t _peak_heap_usage_on_cpu = 0;
  size_t _total_allocated_bytes_on_gpu = 0;
  size_t _total_deallocated_bytes_on_gpu = 0;
  size_t _peak_heap_usage_on_gpu = 0;
  std::unordered_map<void *, size_t> _memory_in_use_on_cpu;
  std::unordered_map<cl_mem, MemoryTraits> _memory_in_use_on_gpu;
};

#endif // !TRACE_H
