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

#include "benchmark/MemoryInfo.h"
#include "benchmark/MemoryPoller.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <numeric>
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

enum PhaseType
{
  MODEL_LOAD,
  PREPARE,
  EXECUTE,
  END_OF_PHASE
};

const std::string getPhaseTypeStr(PhaseType type)
{
  switch (type)
  {
    case MODEL_LOAD:
      return "MODEL_LOAD";
    case PREPARE:
      return "PREPARE";
    case EXECUTE:
      return "EXECUTE";
    default:
      throw std::runtime_error("Invalid phase type");
  }
}

benchmark::PhaseEnum convertToPhaseEnum(PhaseType type)
{
  switch (type)
  {
    case MODEL_LOAD:
      return benchmark::PhaseEnum::MODEL_LOAD;
    case PREPARE:
      return benchmark::PhaseEnum::PREPARE;
    case EXECUTE:
      return benchmark::PhaseEnum::EXECUTE;
    default:
      throw std::runtime_error("Invalid phase type");
  }
}

enum AggregateType
{
  AVERAGE,
  SUM,
  END_OF_AGGREGATE_TYPE
};

enum MemoryType
{
  RSS,
  HWM,
  PSS,
  END_OF_MEM_TYPE
};

const std::string getMemoryTypeStr(MemoryType type)
{
  switch (type)
  {
    case RSS:
      return "RSS";
    case HWM:
      return "HWM";
    case PSS:
      return "PSS";
    default:
      throw std::runtime_error("Invalid memory type");
  }
}

struct Step
{
  uint64_t time; // us
};

struct Phase
{
  uint64_t time;                                // us
  uint32_t memory[MemoryType::END_OF_MEM_TYPE]; // kB
};

class Measure
{
public:
  Measure(bool check_mem_poll) : _check_mem_poll(check_mem_poll)
  {
    if (_check_mem_poll)
    {
      assert(benchmark::prepareVmRSS());
      _mem_poll = std::make_unique<benchmark::MemoryPoller>(std::chrono::milliseconds(100), false);
    }
  }

  void set(const int epoch, const int step)
  {
    _step_results.clear();
    _step_results.resize(epoch);
    std::for_each(_step_results.begin(), _step_results.end(), [step](auto &v) { v.resize(step); });
  }

  void run(const PhaseType phaseType, const std::function<void()> &func)
  {
    auto phaseEnum = convertToPhaseEnum(phaseType);

    if (_check_mem_poll)
    {
      _mem_poll->start(phaseEnum);
    }
    _phase_results[phaseType].time = nowMicros();

    func();

    _phase_results[phaseType].time = nowMicros() - _phase_results[phaseType].time;
    if (_check_mem_poll)
    {
      _mem_poll->end(phaseEnum);

      _phase_results[phaseType].memory[MemoryType::RSS] = _mem_poll->getRssMap().at(phaseEnum);
      _phase_results[phaseType].memory[MemoryType::HWM] = _mem_poll->getHwmMap().at(phaseEnum);
      _phase_results[phaseType].memory[MemoryType::PSS] = _mem_poll->getPssMap().at(phaseEnum);
    }
  }

  void run(const int epoch, const int step, const std::function<void()> &func)
  {
    if (_step_results.empty() || _step_results.size() <= epoch ||
        _step_results[epoch].size() <= step)
    {
      throw std::runtime_error("Please set the number of epochs and steps first");
    }

    _step_results[epoch][step].time = nowMicros();

    func();

    _step_results[epoch][step].time = nowMicros() - _step_results[epoch][step].time;
  }

  double sumTimeMicro(const int epoch)
  {
    double sum = 0u;
    std::for_each(_step_results[epoch].begin(), _step_results[epoch].end(),
                  [&sum](auto &v) { sum += v.time; });
    return sum;
  }

  double timeMicros(const int epoch, const AggregateType aggType)
  {
    if (_step_results.empty() || _step_results.size() <= epoch)
    {
      throw std::runtime_error("Invalid epoch");
    }

    switch (aggType)
    {
      case AVERAGE:
        return sumTimeMicro(epoch) / _step_results[epoch].size();
      case SUM:
        return sumTimeMicro(epoch);
      default:
        throw std::runtime_error("Invalid aggregate type");
    }
  }

  void printTimeMs(const int epoch, const AggregateType aggType)
  {
    std::cout.precision(3);
    std::cout << " - time: " << timeMicros(epoch, aggType) / 1e3 << "ms/step";
  }

  void printResultTime()
  {
    std::cout << "===================================" << std::endl;
    for (int i = 0; i < PhaseType::END_OF_PHASE; ++i)
    {
      auto type = static_cast<PhaseType>(i);
      std::cout << std::setw(12) << std::left << getPhaseTypeStr(type) << " takes "
                << _phase_results[type].time / 1e3 << " ms" << std::endl;
      if (i == PhaseType::EXECUTE)
      {
        for (int j = 0; j < _step_results.size(); ++j)
        {
          std::cout << "- "
                    << "Epoch " << j + 1 << std::setw(12) << std::right << " takes "
                    << timeMicros(j, AggregateType::SUM) / 1e3 << " ms" << std::endl;
        }
      }
    }
    std::cout << "===================================" << std::endl;
  }

  void printResultMemory()
  {
    for (int i = 0; i < MemoryType::END_OF_MEM_TYPE; ++i)
    {
      auto type = static_cast<MemoryType>(i);
      std::cout << getMemoryTypeStr(type) << std::endl;
      for (int j = 0; j < PhaseType::END_OF_PHASE; ++j)
      {
        auto phaseType = static_cast<PhaseType>(j);
        std::cout << "- " << std::setw(12) << std::left << getPhaseTypeStr(phaseType) << " takes "
                  << _phase_results[phaseType].memory[i] << " kb" << std::endl;
      }
      std::cout << "===================================" << std::endl;
    }
  }

  void printResult()
  {
    printResultTime();
    if (_check_mem_poll)
    {
      printResultMemory();
    }
  }

private:
  std::unordered_map<PhaseType, Phase> _phase_results;
  std::vector<std::vector<Step>> _step_results;

  bool _check_mem_poll;
  std::unique_ptr<benchmark::MemoryPoller> _mem_poll;
};

} // namespace onert_train

#endif // __ONERT_TRAIN_MEASURE_H__
