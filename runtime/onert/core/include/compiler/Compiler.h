/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file  Compiler.h
 * @brief This file contains Compiler class to define and run compilation phase
 */

#ifndef __ONERT_COMPILER_COMPILE_H_
#define __ONERT_COMPILER_COMPILE_H_

#include "ir/Graph.h"
#include "exec/IExecutor.h"
#include "util/TracingCtx.h"

namespace onert
{

namespace compiler
{

enum class State
{
  CREATED, // Before compilation
  COMPILED // Success compilation
};

struct ManualSchedulerOptions
{
  std::string backend_for_all;
  std::unordered_map<ir::OpCode, std::string> opcode_to_backend;
  std::unordered_map<ir::OperationIndex, std::string> index_to_backend;
};

struct CompilerOptions
{
  // GENERAL OPTIONS
  std::vector<std::string> backend_list;

  // OPTIONS ONLY FOR DEBUGGING/PROFILING
  std::string trace_filepath; //< File path to save trace records
  int graph_dump_level;       //< Graph dump level, values between 0 and 2 are valid
  std::string executor;       //< Executor name to use
  ManualSchedulerOptions manual_scheduler_options; //< Options for ManualScheduler
  bool he_scheduler;      //< HEScheduler if true, ManualScheduler otherwise
  bool he_profiling_mode; //< Whether HEScheduler profiling mode ON/OFF
  bool disable_compile;   //< Run with Interpreter if true, try compilation otherwise
  bool fp16_enable;       //< Whether fp16 mode ON/OFF

  util::TracingCtx *tracing_ctx; //< Profiling information
};

CompilerOptions fetchCompilerOptionsFromGlobalConfig(const ir::Subgraphs &subgs);

/**
 * @brief Class to compile graph model
 */
class Compiler
{
public:
  /**
   * @brief     Construct a new Compiler object
   * @param[in] subgs All subgraphs of a model
   * @param[in] tracing_ctx Profiling information
   */
  Compiler(const std::shared_ptr<ir::Subgraphs> &subgs, util::TracingCtx *tracing_ctx);

public:
  /**
   * @brief   Do compilation with the options
   *
   * @return std::shared_ptr<exec::ExecutorMap> Executors as a result of compilation
   */
  std::shared_ptr<exec::ExecutorMap> compile(void);

  State state(void) const { return _state; }

  CompilerOptions &options() { return _options; }

  /**
   * @brief   Allow to compute float32 using float16 data type
   */
  void enableToFp16();

  /**
   * @brief   Set backends from string-encoded mappings from operation index to backend type (cpu,
   * acl_cl)
   */
  void set_backend_from_str(const char *backend_settings);

private:
  void checkProfilerConditions();
  std::shared_ptr<ir::Graph> &primary_subgraph() { return _subgraphs->at(ir::SubgraphIndex{0}); }

private:
  std::shared_ptr<ir::Subgraphs> _subgraphs;
  // NOTE These executors does not have duplicated subgraph. This mean they do not allow support
  // subgraphs being called recursively because data of non-constant tensor of parent executor will
  // be updated by child executor. If you want to support subgraphs being called recursively, you
  // have to add allocate non-constant tensor memory of executors in execution time when each
  // subgraph is called.
  State _state;
  CompilerOptions _options;
};

} // namespace compiler

} // namespace onert

#endif // __ONERT_COMPILER_COMPILE_H_
