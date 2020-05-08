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
  int op_seq_max_node;        //< Number of nodes that can be
  std::string executor;       //< Executor name to use
  ManualSchedulerOptions manual_scheduler_options; //< Options for ManualScheduler
  bool he_scheduler;       //< HEScheduler if true, ManualScheduler otherwise
  bool he_profiling_mode;  //< Whether HEScheduler profiling mode ON/OFF
  bool delete_cached_data; //< Whether CachedData deletion ON/OFF
  bool disable_compile;    //< Run with Interpreter if true, try compilation otherwise
  bool fp16_enable;        //< Whether fp16 mode ON/OFF
};

CompilerOptions fetchCompilerOptionsFromGlobalConfig(const ir::Graph &graph);

/**
 * @brief Class to compile graph model
 */
class Compiler
{
public:
  /**
   * @brief     Construct a new Compiler object
   * @param[in] subgs All subgraphs of a model
   */
  Compiler(const std::shared_ptr<ir::Subgraphs> &subgs);

public:
  /**
   * @brief   Run compilation. Compilation result will be saved in _plan
   */
  void compile(void);
  /**
   * @brief       Pass plan reference
   * @param[out]  plan  Plan reference to return\n
   *                    Set nullptr if compile is not run yet
   */
  void release(std::shared_ptr<exec::ExecutorMap> &executors) { executors = _executors; }

  void state(State state) { _state = state; }
  State state(void) const { return _state; }

  /**
   * @brief   Check if model can compile
   * @return  @c true if model can compile, otherwise @c false
   * @note    This method don't check model correctness,\n
   *          so model verification should be done before calling this method
   */
  bool checkCompilable();
  CompilerOptions &options() { return _options; }

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
  std::shared_ptr<exec::ExecutorMap> _executors;
  State _state;
  CompilerOptions _options;
};

} // namespace compiler

} // namespace onert

#endif // __ONERT_COMPILER_COMPILE_H_
