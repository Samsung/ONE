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

#ifndef __ONERT_BACKEND_ACL_CL_CLTIMER_H__
#define __ONERT_BACKEND_ACL_CL_CLTIMER_H__

#include <util/ITimer.h>
#include <arm_compute/core/CL/OpenCL.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <chrono>
#include <list>
#include <sstream>

namespace onert
{
namespace backend
{
namespace acl_cl
{

/**
 * @brief Class to measure CL kernels execution time
 */
class CLTimer : public util::ITimer
{
public:
  /**
   * @brief This function replaces CL function, which enqueues a command to execute a kernel
   *        with a wrapper which remembers enqueued kernels
   */
  void handleBegin() override
  {
    _measured_events.clear();

    _origin_enqueue_function = arm_compute::CLSymbols::get().clEnqueueNDRangeKernel_ptr;

    auto _timer_enqueue_function = [this](cl_command_queue command_queue, cl_kernel kernel,
                                          cl_uint work_dim, const size_t *gwo, const size_t *gws,
                                          const size_t *lws, cl_uint num_events_in_wait_list,
                                          const cl_event *event_wait_list, cl_event *usr_event) {
      cl_event event;
      cl_int enqueue_res =
        this->_origin_enqueue_function(command_queue, kernel, work_dim, gwo, gws, lws,
                                       num_events_in_wait_list, event_wait_list, &event);
      this->_measured_events.emplace_back(event);

      // According to spec, if NULL was provided in usr_event - event shouldn't be returned
      if (usr_event != nullptr)
      {
        clRetainEvent(event);
        *usr_event = event;
      }
      return enqueue_res;
    };
    arm_compute::CLSymbols::get().clEnqueueNDRangeKernel_ptr = _timer_enqueue_function;

    // Set CL_QUEUE_PROFILING_ENABLE flag for the CL command-queue, if it isn't already set
    auto &cl_scheduler = arm_compute::CLScheduler::get();
    auto props = cl_scheduler.queue().getInfo<CL_QUEUE_PROPERTIES>();
    if ((props & CL_QUEUE_PROFILING_ENABLE) == 0)
    {
      cl_scheduler.set_queue(
        cl::CommandQueue(cl_scheduler.context(), props | CL_QUEUE_PROFILING_ENABLE));
    }
  };

  /**
   * @brief Get timer result by addition executed CL kernels durations
   */
  void handleEnd() override
  {
    _timer_res = 0;
    for (auto const &event : _measured_events)
    {
      cl_ulong start;
      cl_ulong end;
      event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
      event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
      _timer_res += (end - start) / 1000.f; // nanoseconds -> microseconds
    }

    // Restore origin CL enqueue function
    arm_compute::CLSymbols::get().clEnqueueNDRangeKernel_ptr = _origin_enqueue_function;
  };

private:
  std::function<decltype(clEnqueueNDRangeKernel)> _origin_enqueue_function;
  std::list<::cl::Event> _measured_events;
};

} // namespace acl_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_CL_CLTIMER_H__
