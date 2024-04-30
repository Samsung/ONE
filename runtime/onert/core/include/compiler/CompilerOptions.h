/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_COMPILER_OPTIONS_H_
#define __ONERT_COMPILER_COMPILER_OPTIONS_H_

#include "ir/OpCode.h"
#include "ir/Index.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <unordered_set>

namespace onert
{
namespace compiler
{

struct ManualSchedulerOptions
{
public:
  void setBackendMap(const std::string &str);

public:
  std::string backend_for_all;
  std::unordered_map<ir::OpCode, std::string> opcode_to_backend;
  std::unordered_map<ir::OperationIndex, std::string> index_to_backend;
};

class CompilerOptions
{
public:
  /**
   * @brief   Set default values for CompilerOptions
   * @return  Generated CompileOption
   *
   * @note    All these default values should not be fetched from Env
   *          when we stop supporting Android NNAPI.
   */
  static std::unique_ptr<CompilerOptions> fromGlobalConfig();

  /**
   * @brief Allow to compute float32 using float16 data type
   */
  void enableToFp16() { fp16_enable = true; }

  /**
   * @brief Force default values of CompilerOptions for correct compilations
   *
   * @note  This should be called after CompilerOptions setting is finished
   *        to prevent value overwriting
   */
  void forceInternalOptions();

  /**
   * @brief Print option value
   */
  void verboseOptions();

public:
  // GENERAL OPTIONS
  std::vector<std::string> backend_list;
  std::string minmax_filepath; //< File path to save minmax

  // OPTIONS ONLY FOR DEBUGGING/PROFILING
  std::string trace_filepath; //< File path to save trace records
  int graph_dump_level;       //< Graph dump level, values between 0 and 2 are valid
  std::string executor;       //< Executor name to use
  ManualSchedulerOptions manual_scheduler_options; //< Options for ManualScheduler
  bool he_scheduler;      //< HEScheduler if true, ManualScheduler otherwise
  bool he_profiling_mode; //< Whether HEScheduler profiling mode ON/OFF
  bool fp16_enable;       //< Whether fp16 mode ON/OFF
  std::unordered_set<onert::ir::OperationIndex> frozen_train_ops;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_COMPILER_OPTIONS_H_
