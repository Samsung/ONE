/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <iostream>
#include "controller.h"
#include "memory_stats.h"

ParetoScheduler::ParetoScheduler(RunSession *session)
  : _session(session), _available_memory(0), _reference_memory(0), _transient_lock(T_DISABLED),
    _transient_wait_time(0)
{
  _available_memory = get_meminfo(MEM_AVAILABLE);
  _reference_memory = _available_memory;
  opt->set_thresholds(PERCENTAGE_THRESHOLD_LATENCY, PERCENTAGE_THRESHOLD_MEMORY);
  std::cout << "Available memory before session creation: " << _available_memory << std::endl;
}

void ParetoScheduler::latency_monitoring(float exec_time, int inference_cnt)
{
  if (_transient_lock != T_ENABLED_FOR_MEMORY && _session->latency_increased(exec_time))
  {
    // Enable here
    bool reconfig_success = _session->reconfigure_within_exec_time(exec_time);
    if (reconfig_success == true)
    {
      ; // Enable all from here
      _reference_memory = get_meminfo(MEM_AVAILABLE);
      _available_memory = _reference_memory;
      _transient_lock = T_ENABLED_FOR_TIME;
      json->add_instance_record("T lock enabled (available): " + std::to_string(_available_memory));
    }
    else
    {
      _available_memory = get_meminfo(MEM_AVAILABLE);
    }
  }
  else
  {
    _available_memory = get_meminfo(MEM_AVAILABLE);
  }
  // Track any changes in memory for post processing
  if (inference_cnt % TRACE_INTERVAL == 0)
  {
    double rss, vm;
    process_mem_usage(vm, rss);
    json->add_instance_record("Mem (available, free) check: " + std::to_string(_available_memory) +
                              ", " + std::to_string(get_meminfo(MEM_FREE)) + ": " +
                              std::to_string(rss));
  }
  // Enable controller again when the transient phase has ended
  if ((_transient_lock == T_ENABLED_FOR_TIME) || (_transient_lock == T_ENABLED_FOR_MEMORY))
  {
    _transient_wait_time += exec_time;
    if (_transient_wait_time > THRESHOLD_WAIT_TIME)
    {
      _transient_lock = T_DISABLED;
      _transient_wait_time = 0;
      _reference_memory = get_meminfo(MEM_AVAILABLE);
    }
  }
}

void ParetoScheduler::memory_monitoring(void)
{
  if ((_transient_lock == T_DISABLED) && (_available_memory > _reference_memory) &&
      _session->memory_improved(_available_memory - _reference_memory))
  {
    // Enable here
    // _session->reconfigure_within_memory(3 * (_available_memory - _reference_memory));
    _session->reconfigure_for_smallest_exec();
    ; // Enable all from here
    _reference_memory = get_meminfo(MEM_AVAILABLE);
    _transient_lock = T_ENABLED_FOR_MEMORY;
  }
}
