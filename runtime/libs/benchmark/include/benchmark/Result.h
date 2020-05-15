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

#ifndef __NNFW_BENCHMARK_RESULT_H__
#define __NNFW_BENCHMARK_RESULT_H__

#include "Phase.h"
#include "MemoryPoller.h"

#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <memory>
#include <cmath>

namespace
{

uint32_t maxMemory(const std::unordered_map<benchmark::Phase, uint32_t> &map)
{
  auto answer = *std::max_element(
      map.begin(), map.end(),
      [](const std::pair<benchmark::Phase, uint32_t> &p1,
         const std::pair<benchmark::Phase, uint32_t> &p2) { return p1.second < p2.second; });
  return answer.second;
}

} // namespace

namespace benchmark
{

// Data class between runner(nnpackage_run and tflite_run) and libbenchmark
class Result
{
public:
  Result(double model_load_time, double prepare_time, const std::vector<double> &execute_times)
      : _model_load_time(model_load_time), _prepare_time(prepare_time), _model_load_rss(0),
        _prepare_rss(0), _execute_rss(0), _peak_rss(0), _model_load_hwm(0), _prepare_hwm(0),
        _execute_hwm(0), _peak_hwm(0)
  {
    // execute
    double sum = 0.0;
    double log_sum = 0.0;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::lowest();
    for (auto t : execute_times)
    {
      sum += t;
      log_sum += std::log(t);
      min = std::min(min, t);
      max = std::max(max, t);
    }
    _execute_time_mean = sum / static_cast<double>(execute_times.size());
    _execute_time_min = min;
    _execute_time_max = max;

    // Calculating geometric mean with logs
    //   "Geometric Mean of (V1, V2, ... Vn)"
    // = (V1*V2*...*Vn)^(1/n)
    // = exp(log((V1*V2*...*Vn)^(1/n)))
    // = exp(log((V1*V2*...*Vn)/n)))
    // = exp((log(V1) + log(V2) + ... + log(Vn))/n)
    // = exp(_log_sum/num)
    _execute_time_geomean = std::exp(log_sum / static_cast<double>(execute_times.size()));
  }

  Result(double model_load_time, double prepare_time, const std::vector<double> &execute_times,
         const std::unique_ptr<MemoryPoller> &memory_poller)
      : Result(model_load_time, prepare_time, execute_times)
  {
    if (!memory_poller)
      return;

    const auto &rss = memory_poller->getRssMap();
    const auto &hwm = memory_poller->getHwmMap();

    // rss
    assert(rss.size() > 0);
    assert(rss.find(Phase::MODEL_LOAD) != rss.end());
    assert(rss.find(Phase::PREPARE) != rss.end());
    assert(rss.find(Phase::EXECUTE) != rss.end());
    _model_load_rss = rss.at(Phase::MODEL_LOAD);
    _prepare_rss = rss.at(Phase::PREPARE);
    _execute_rss = rss.at(Phase::EXECUTE);
    _peak_rss = maxMemory(rss);

    // hwm
    assert(hwm.size() > 0);
    assert(hwm.find(Phase::MODEL_LOAD) != hwm.end());
    assert(hwm.find(Phase::PREPARE) != hwm.end());
    assert(hwm.find(Phase::EXECUTE) != hwm.end());
    _model_load_hwm = hwm.at(Phase::MODEL_LOAD);
    _prepare_hwm = hwm.at(Phase::PREPARE);
    _execute_hwm = hwm.at(Phase::EXECUTE);
    _peak_hwm = maxMemory(hwm);
  }

public:
  double getModelLoadTime() const { return _model_load_time; }
  double getPrepareTime() const { return _prepare_time; }
  double getExecuteTimeMean() const { return _execute_time_mean; }
  double getExecuteTimeGeoMean() const { return _execute_time_geomean; }
  double getExecuteTimeMin() const { return _execute_time_min; }
  double getExecuteTimeMax() const { return _execute_time_max; }

  uint32_t getModelLoadRss() const { return _model_load_rss; }
  uint32_t getPrepareRss() const { return _prepare_rss; }
  uint32_t getExecuteRss() const { return _execute_rss; }
  uint32_t getPeakRss() const { return _peak_rss; }

  uint32_t getModelLoadHwm() const { return _model_load_hwm; }
  uint32_t getPrepareHwm() const { return _prepare_hwm; }
  uint32_t getExecuteHwm() const { return _execute_hwm; }
  uint32_t getPeakHwm() const { return _peak_hwm; }

private:
  double _model_load_time;
  double _prepare_time;
  double _execute_time_mean;
  double _execute_time_geomean;
  double _execute_time_min;
  double _execute_time_max;

  uint32_t _model_load_rss;
  uint32_t _prepare_rss;
  uint32_t _execute_rss;
  uint32_t _peak_rss;

  uint32_t _model_load_hwm;
  uint32_t _prepare_hwm;
  uint32_t _execute_hwm;
  uint32_t _peak_hwm;
};

} // namespace benchmark

#endif // __NNFW_BENCHMARK_RESULT_H__
