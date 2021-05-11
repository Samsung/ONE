/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __ONERT_BACKEND_GPU_CL_CL_EVENT_H__
#define __ONERT_BACKEND_GPU_CL_CL_EVENT_H__

#include <cstdint>
#include <string>

#include "OpenclWrapper.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

// A RAII wrapper around opencl event
class CLEvent
{
public:
  CLEvent() {}
  explicit CLEvent(cl_event event);

  // Move only
  CLEvent(CLEvent &&event);
  CLEvent &operator=(CLEvent &&event);
  CLEvent(const CLEvent &) = delete;
  CLEvent &operator=(const CLEvent &) = delete;

  ~CLEvent();

  uint64_t GetStartedTimeNs() const;
  uint64_t GetFinishedTimeNs() const;

  double GetEventTimeMs() const;
  uint64_t GetEventTimeNs() const;

  void Wait() const;

  cl_event event() const { return event_; }

  bool is_valid() const { return event_ != nullptr; }

  void SetName(const std::string &name);
  std::string GetName() const { return name_; }

private:
  void Release();

  cl_event event_ = nullptr;

  std::string name_; // optional, for profiling mostly
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_CL_EVENT_H__
