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

namespace onert
{
namespace compiler
{

struct ManualSchedulerOptions
{
  void setBackendMap(const std::string &str);

  std::string backend_for_all;
  std::unordered_map<ir::OpCode, std::string> opcode_to_backend;
  std::unordered_map<ir::OperationIndex, std::string> index_to_backend;
};

struct CompilerOptions
{
  /**
   * @brief   Set default values for CompilerOptions
   * @return  Generated CompileOption
   *
   * @note    All these default values should not be fetched from Env
   *          when we stop supporting Android NNAPI.
   */
  static std::unique_ptr<CompilerOptions> fromGlobalConfig();

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

  // GENERAL OPTIONS
  std::vector<std::string> backend_list;
  bool minmax_dump; //< Whether minmax dump is enabled or not

  // OPTIONS ONLY FOR DEBUGGING/PROFILING
  bool tracing_mode;    //< Whether tracing mode ON/OFF
  int graph_dump_level; //< Graph dump level, values between 0 and 2 are valid
  std::string executor; //< Executor name to use
  ManualSchedulerOptions manual_scheduler_options; //< Options for ManualScheduler
  bool he_scheduler;         //< HEScheduler if true, ManualScheduler otherwise
  bool he_profiling_mode;    //< Whether HEScheduler profiling mode ON/OFF
  bool fp16_enable;          //< Whether fp16 mode ON/OFF
  std::string workspace_dir; //< Workspace directory path
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_COMPILER_OPTIONS_H_
