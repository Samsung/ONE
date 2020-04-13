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

#ifndef __NNFW_BENCHMARK_UTIL_H__
#define __NNFW_BENCHMARK_UTIL_H__

#include "Result.h"
#include "CsvWriter.h"

#include <chrono>
#include <string>
#include <iostream>

namespace benchmark
{

inline uint64_t nowMicros()
{
  auto time_point = std::chrono::high_resolution_clock::now();
  auto since_epoch = time_point.time_since_epoch();
  // default precision of high resolution clock is 10e-9 (nanoseconds)
  return std::chrono::duration_cast<std::chrono::microseconds>(since_epoch).count();
}

// TODO Support not only stdout but also ostream
inline void printResult(const Result &result, bool print_memory)
{
  std::cout << "===================================" << std::endl;
  std::cout << getPhaseString(Phase::MODEL_LOAD) << " takes " << result.getModelLoadTime() / 1e3
            << " ms" << std::endl;
  std::cout << getPhaseString(Phase::PREPARE) << "    takes " << result.getPrepareTime() / 1e3
            << " ms" << std::endl;
  std::cout << getPhaseString(Phase::EXECUTE) << "    takes " << std::endl;
  std::cout << "- Min:  " << result.getExecuteTimeMin() / 1e3 << " ms" << std::endl;
  std::cout << "- Max:  " << result.getExecuteTimeMax() / 1e3 << " ms" << std::endl;
  std::cout << "- Mean: " << result.getExecuteTimeMean() / 1e3 << " ms" << std::endl;
  std::cout << "===================================" << std::endl;

  if (print_memory == false)
    return;

  std::cout << "RSS" << std::endl;
  std::cout << "- " << getPhaseString(Phase::MODEL_LOAD) << " takes " << result.getModelLoadRss()
            << " kb" << std::endl;
  std::cout << "- " << getPhaseString(Phase::PREPARE) << "    takes " << result.getPrepareRss()
            << " kb" << std::endl;
  std::cout << "- " << getPhaseString(Phase::EXECUTE) << "    takes " << result.getExecuteRss()
            << " kb" << std::endl;
  std::cout << "- PEAK "
            << "      takes " << result.getPeakRss() << " kb" << std::endl;
  std::cout << "===================================" << std::endl;
  std::cout << "HWM" << std::endl;
  std::cout << "- " << getPhaseString(Phase::MODEL_LOAD) << " takes " << result.getModelLoadHwm()
            << " kb" << std::endl;
  std::cout << "- " << getPhaseString(Phase::PREPARE) << "    takes " << result.getPrepareHwm()
            << " kb" << std::endl;
  std::cout << "- " << getPhaseString(Phase::EXECUTE) << "    takes " << result.getExecuteHwm()
            << " kb" << std::endl;
  std::cout << "- PEAK "
            << "      takes " << result.getPeakHwm() << " kb" << std::endl;
  std::cout << "===================================" << std::endl;
}

// TODO Support not only csv but also other datafile format such as xml, json, ...
inline void writeResult(const Result &result, const std::string &exec, const std::string &model,
                        const std::string &backend)
{
  std::string csv_filename = exec + "-" + model + "-" + backend + ".csv";

  // write to csv
  CsvWriter writer(csv_filename);
  writer << model << backend << result.getModelLoadTime() / 1e3 << result.getPrepareTime() / 1e3
         << result.getExecuteTimeMin() / 1e3 << result.getExecuteTimeMax() / 1e3
         << result.getExecuteTimeMean() / 1e3 << result.getModelLoadRss() << result.getPrepareRss()
         << result.getExecuteRss() << result.getPeakRss() << result.getModelLoadHwm()
         << result.getPrepareHwm() << result.getExecuteHwm() << result.getPeakHwm();

  bool done = writer.done();

  if (!done)
  {
    std::cerr << "Writing to " << csv_filename << " is failed" << std::endl;
  }
}

} // namespace benchmark

#endif // __NNFW_BENCHMARK_UTIL_H__
