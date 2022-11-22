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

#include "Types.h"
#include "Phases.h"

#include <string>

namespace benchmark
{

// Data class between runner(onert_run and tflite_run) and libbenchmark
class Result
{
public:
  Result(const Phases &phases);

  double time[PhaseEnum::END_OF_PHASE][FigureType::END_OF_FIG_TYPE];
  uint32_t memory[PhaseEnum::END_OF_PHASE][MemoryType::END_OF_MEM_TYPE];
  bool print_memory = false;
  uint32_t init_memory = 0;
  uint32_t peak_memory = 0;
};

// TODO Support not only stdout but also ostream
void printResult(const Result &result);

// TODO Support not only csv but also other datafile format such as xml, json, ...
void writeResult(const Result &result, const std::string &exec, const std::string &model,
                 const std::string &backend);

} // namespace benchmark

#endif // __NNFW_BENCHMARK_RESULT_H__
