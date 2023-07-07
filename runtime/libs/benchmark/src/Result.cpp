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

#include "benchmark/Result.h"
#include "benchmark/Phases.h"
#include "benchmark/CsvWriter.h"

#include <string>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace
{

template <class T, class R> R average(const std::vector<T> &v)
{
  T sum = std::accumulate(v.begin(), v.end(), 0);
  R avg = sum / static_cast<R>(v.size());
  return avg;
}

double averageTimeMs(const benchmark::Phase &phase)
{
  double avg_us = average<uint64_t, double>(phase.time);
  return avg_us / 1e3;
}

double maxTimeMs(const benchmark::Phase &phase)
{
  auto it = max_element(std::begin(phase.time), std::end(phase.time));
  return *it / 1e3;
}

double minTimeMs(const benchmark::Phase &phase)
{
  auto it = min_element(std::begin(phase.time), std::end(phase.time));
  return *it / 1e3;
}

double geomeanTimeMs(const benchmark::Phase &phase)
{
  double log_sum = 0.0;
  for (auto &&t_us : phase.time)
  {
    log_sum += std::log(t_us / 1e3);
  }

  // Calculating geometric mean with logs
  //   "Geometric Mean of (V1, V2, ... Vn)"
  // = (V1*V2*...*Vn)^(1/n)
  // = exp(log((V1*V2*...*Vn)^(1/n)))
  // = exp(log((V1*V2*...*Vn)/n)))
  // = exp((log(V1) + log(V2) + ... + log(Vn))/n)
  // = exp(_log_sum/num)
  return std::exp(log_sum / static_cast<double>(phase.time.size()));
}

uint32_t averageMemoryKb(const benchmark::Phase &phase, int type)
{
  return average<uint32_t, uint32_t>(phase.memory[type]);
}

uint32_t peakMemory(
  const uint32_t memory[benchmark::PhaseEnum::END_OF_PHASE][benchmark::MemoryType::END_OF_MEM_TYPE],
  int type)
{
  using namespace benchmark;
  // tricky. handle WARMUP as EXECUTE
  return std::max({memory[PhaseEnum::MODEL_LOAD][type], memory[PhaseEnum::PREPARE][type],
                   memory[PhaseEnum::WARMUP][type]});
}

void printResultTime(
  const double time[benchmark::PhaseEnum::END_OF_PHASE][benchmark::FigureType::END_OF_FIG_TYPE])
{
  using namespace benchmark;

  std::cout << "===================================" << std::endl;

  std::streamsize ss_precision = std::cout.precision();
  std::cout << std::setprecision(3);
  std::cout << std::fixed;

  for (int i = PhaseEnum::MODEL_LOAD; i <= PhaseEnum::EXECUTE; ++i)
  {
    // Note. Tricky. Ignore WARMUP
    if (i == PhaseEnum::WARMUP)
      continue;
    std::cout << std::setw(12) << std::left << getPhaseString(i) << " takes "
              << time[i][FigureType::MEAN] << " ms" << std::endl;
  }

  for (int j = FigureType::MEAN; j <= FigureType::GEOMEAN; ++j)
  {
    std::cout << "- " << std::setw(9) << std::left << getFigureTypeString(j) << ":  "
              << time[PhaseEnum::EXECUTE][j] << " ms" << std::endl;
  }

  std::cout << std::setprecision(ss_precision);
  std::cout << std::defaultfloat;

  std::cout << "===================================" << std::endl;
}

void printResultMemory(
  const uint32_t memory[benchmark::PhaseEnum::END_OF_PHASE][benchmark::MemoryType::END_OF_MEM_TYPE])
{
  using namespace benchmark;

  for (int j = MemoryType::RSS; j <= MemoryType::PSS; ++j)
  {
    std::cout << getMemoryTypeString(j) << std::endl;
    for (int i = PhaseEnum::MODEL_LOAD; i <= PhaseEnum::PREPARE; ++i)
    {
      std::cout << "- " << std::setw(12) << std::left << getPhaseString(i) << " takes "
                << memory[i][j] << " kb" << std::endl;
    }
    // Tricky. Handle WARMUP as EXECUTE
    std::cout << "- " << std::setw(12) << std::left << getPhaseString(PhaseEnum::EXECUTE)
              << " takes " << memory[PhaseEnum::WARMUP][j] << " kb" << std::endl;
    std::cout << "- " << std::setw(12) << std::left << "PEAK"
              << " takes " << peakMemory(memory, j) << " kb" << std::endl;
    std::cout << "===================================" << std::endl;
  }
}

void printUsedPeakMemory(uint32_t init_memory, uint32_t peak_memory)
{
  uint32_t used_peak_memory = peak_memory - init_memory;
  std::cout << "Used Peak Memory : " << used_peak_memory << " kb" << std::endl;
  std::cout << "- HWM after run  : " << peak_memory << " kb" << std::endl;
  std::cout << "- HWM before init: " << init_memory << " kb" << std::endl;
  std::cout << "===================================" << std::endl;
}

} // namespace

namespace benchmark
{

Result::Result(const Phases &phases)
{
  const auto option = phases.option();
  {
    for (int i = PhaseEnum::MODEL_LOAD; i <= PhaseEnum::PREPARE; ++i)
    {
      auto phase = phases.at(gPhaseStrings[i]);
      time[i][FigureType::MEAN] = averageTimeMs(phase);
    }

    int i = PhaseEnum::EXECUTE;
    auto exec_phase = phases.at(gPhaseStrings[i]);
    time[i][FigureType::MEAN] = averageTimeMs(exec_phase);
    time[i][FigureType::MAX] = maxTimeMs(exec_phase);
    time[i][FigureType::MIN] = minTimeMs(exec_phase);
    time[i][FigureType::GEOMEAN] = geomeanTimeMs(exec_phase);
  }
  if (option.memory)
  {
    print_memory = true;
    for (int i = PhaseEnum::MODEL_LOAD; i < PhaseEnum::EXECUTE; ++i)
    {
      auto phase = phases.at(gPhaseStrings[i]);
      for (int j = MemoryType::RSS; j <= MemoryType::PSS; ++j)
      {
        memory[i][j] = averageMemoryKb(phase, j);
      }
    }
  }
  init_memory = phases.mem_before_init();
  peak_memory = phases.mem_after_run();
}

void printResult(const Result &result)
{
  printResultTime(result.time);

  if (result.print_memory == false)
    return;

  printResultMemory(result.memory);
  printUsedPeakMemory(result.init_memory, result.peak_memory);
}

// TODO There are necessary for a kind of output data file so that it doesn't have to be csv file
// format.
void writeResult(const Result &result, const std::string &exec, const std::string &model,
                 const std::string &backend)
{
  std::string csv_filename = exec + "-" + model + "-" + backend + ".csv";

  // write to csv
  CsvWriter writer(csv_filename);
  writer << model << backend;

  // TODO Add GEOMEAN
  // time
  auto time = result.time;
  writer << time[PhaseEnum::MODEL_LOAD][FigureType::MEAN]
         << time[PhaseEnum::PREPARE][FigureType::MEAN] << time[PhaseEnum::EXECUTE][FigureType::MIN]
         << time[PhaseEnum::EXECUTE][FigureType::MAX] << time[PhaseEnum::EXECUTE][FigureType::MEAN];

  // memory
  auto memory = result.memory;
  for (int j = MemoryType::RSS; j <= MemoryType::PSS; ++j)
  {
    // Tricky. Handle WARMUP as EXECUTE
    for (int i = PhaseEnum::MODEL_LOAD; i <= PhaseEnum::WARMUP; ++i)
    {
      writer << memory[i][j];
    }
    writer << peakMemory(memory, j);
  }

  bool done = writer.done();

  if (!done)
  {
    std::cerr << "Writing to " << csv_filename << " is failed" << std::endl;
  }
}

} // namespace benchmark
