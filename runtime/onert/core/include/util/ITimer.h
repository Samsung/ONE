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

#ifndef __ONERT_UTIL_ITIMER_H__
#define __ONERT_UTIL_ITIMER_H__

#include <chrono>

namespace onert
{
namespace util
{

class ITimer
{
public:
  virtual void handleBegin() = 0;
  virtual void handleEnd() = 0;
  int getTime() { return _timer_res; };

  virtual ~ITimer() = default;

protected:
  int _timer_res{0};
};

class CPUTimer : public ITimer
{
public:
  void handleBegin() override { _start_time = std::chrono::steady_clock::now(); };

  void handleEnd() override
  {
    const auto end_time = std::chrono::steady_clock::now();
    _timer_res =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - _start_time).count();
  };

private:
  std::chrono::steady_clock::time_point _start_time; // in microseconds
};

} // namespace util
} // namespace onert

#endif // __ONERT_UTIL_ITIMER_H__
