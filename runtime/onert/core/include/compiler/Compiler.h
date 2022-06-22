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
public:
  void setBackendMap(const ir::Model &model, const std::string &str);

public:
  std::string backend_for_all;
  std::unordered_map<ir::OpCode, std::string> opcode_to_backend;
  std::unordered_map<ir::OperationIndex, std::string> index_to_backend;
};

struct PartialGraphOptions
{
  std::unordered_map<ir::OperationIndex, ir::SubgraphIndex> index_to_graph;
};

class CompilerOptions
{
public:
  CompilerOptions(const ir::Model &model) { fetchCompilerOptionsFromGlobalConfig(model); }

private:
  // Set default values for CompilerOptions
  // All these default values should not be fetched from Env, when we stop supporting Android NNAPI.
  void fetchCompilerOptionsFromGlobalConfig(const ir::Model &model);

public:
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
  PartialGraphOptions partial_graph_options;
};

struct CompilerArtifact
{
  CompilerArtifact(void) = delete;
  CompilerArtifact(std::shared_ptr<exec::ExecutorMap> executors,
                   std::unique_ptr<const util::TracingCtx> tracing_ctx)
    : _executors{executors}, _tracing_ctx{std::move(tracing_ctx)} {};

  std::shared_ptr<exec::ExecutorMap> _executors;
  std::unique_ptr<const util::TracingCtx> _tracing_ctx;
};

/**
 * @brief Class to compile graph model
 */
class Compiler
{
public:
  /**
   * @brief     Construct a new Compiler object
   * @param[in] model model to compile
   * @param[in] coptions Compiler Options
   */
  Compiler(const std::shared_ptr<ir::Model> &model, CompilerOptions &copt);

public:
  /**
   * @brief   Do compilation with the options
   *
   * @return std::shared_ptr<CompilerArtifact> Executors as a result of compilation
   */
  std::shared_ptr<CompilerArtifact> compile(void);

  /**
   * @brief   Do compilation with the options
   *
   * @return std::vector<std::shared_ptr<CompilerArtifact>> Executors as a result of compilation
   * for pipeline
   */
  std::vector<std::shared_ptr<CompilerArtifact>> compile(const char *package_file_path,
                                                         const char *map_file_path);

  State state(void) const { return _state; }

  CompilerOptions &options() { return _options; }

  /**
   * @brief   Allow to compute float32 using float16 data type
   */
  void enableToFp16();

  /**
   * @brief   Build the partial graphs to compile with original graph
   */
  bool buildPartialGraph(uint32_t num_graphs);

private:
  void checkProfilerConditions();
  std::shared_ptr<ir::Graph> &primary_subgraph() { return _model->at(ir::SubgraphIndex{0}); }

private:
  std::shared_ptr<ir::Model> _model;
  // NOTE These executors does not have duplicated subgraph. This mean they do not allow support
  // subgraphs being called recursively because data of non-constant tensor of parent executor will
  // be updated by child executor. If you want to support subgraphs being called recursively, you
  // have to add allocate non-constant tensor memory of executors in execution time when each
  // subgraph is called.
  State _state;
  CompilerOptions &_options;
};

} // namespace compiler

} // namespace onert

#endif // __ONERT_COMPILER_COMPILE_H_
