/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_TRAIN_MEASURE_H__
#define __ONERT_TRAIN_MEASURE_H__

#include <algorithm>
#include <ctime>
#include <vector>

namespace
{
uint64_t nowMicros()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_nsec) / 1e3 + static_cast<uint64_t>(ts.tv_sec) * 1e6;
}
} // namespace

namespace onert_train
{

struct Step
{
  uint64_t time; // us
  // TODO Support memory usage
};

class Measure
{
public:
  Measure() = default;

  void set(const int epoch, const int step)
  {
    _results.clear();
    _results.resize(epoch);
    std::for_each(_results.begin(), _results.end(), [step](auto &v) { v.resize(step); });
  }

  void run(const int epoch, const int step, const std::function<void()> &func)
  {
    if (_results.empty() || _results.size() <= epoch || _results[epoch].size() <= step)
    {
      throw std::runtime_error("Please set the number of epochs and steps first");
    }

    _results[epoch][step].time = nowMicros();

    func();

    _results[epoch][step].time = nowMicros() - _results[epoch][step].time;
  }

  double timeMicros(const int epoch)
  {
    if (_results.empty() || _results.size() <= epoch)
    {
      throw std::runtime_error("Invalid epoch");
    }

    double sum = 0u;
    std::for_each(_results[epoch].begin(), _results[epoch].end(),
                  [&sum](auto &v) { sum += v.time; });
    return sum / _results[epoch].size();
  }

  double timeMs(const int epoch) { return timeMicros(epoch) / 1e3; }

private:
  std::vector<std::vector<Step>> _results;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_MEASURE_H__
