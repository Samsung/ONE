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

#include "compiler/Compiler.h"

#include "ParamChecker.h"
#include "ExecutorFactory.h"
#include "OperationValidator.h"

#include "compiler/BackendManager.h"
#include "compiler/IScheduler.h"
#include "compiler/ManualScheduler.h"
#include "compiler/HEScheduler.h"
#include "exec/ExecTime.h"
#include "ir/operation/LowerInfo.h"
#include "dumper/dot/DotDumper.h"
#include "compiler/Linear.h"
#include "interp/InterpExecutor.h"
#include "util/ConfigSource.h"
#include "ir/OperationDumper.h"
#include "compiler/CachedDataDeleter.h"
#include "misc/string_helpers.h"

namespace onert
{

namespace compiler
{

CompilerOptions fetchCompilerOptionsFromGlobalConfig(const ir::Graph &graph)
{
  CompilerOptions options;

  options.backend_list = nnfw::misc::split(util::getConfigString(util::config::BACKENDS), ';');

  options.trace_filepath = util::getConfigString(util::config::TRACE_FILEPATH);
  options.graph_dump_level = util::getConfigInt(util::config::GRAPH_DOT_DUMP);
  options.op_seq_max_node = util::getConfigInt(util::config::OP_SEQ_MAX_NODE);
  options.executor = util::getConfigString(util::config::EXECUTOR);
  options.he_scheduler = util::getConfigBool(util::config::USE_SCHEDULER);
  options.he_profiling_mode = util::getConfigBool(util::config::PROFILING_MODE);
  options.delete_cached_data = util::getConfigBool(util::config::DELETE_CACHED_DATA);
  options.disable_compile = util::getConfigBool(util::config::DISABLE_COMPILE);

  {
    // Backend for all
    auto &ms_options = options.manual_scheduler_options;

    // Default value for op_backend_all is first element in the backend list
    ms_options.backend_for_all = util::getConfigString(util::config::OP_BACKEND_ALLOPS);

// Opcode to Backend
#define OP(OpName)                                                                      \
  {                                                                                     \
    const auto &backend_str = util::getConfigString(util::config::OP_BACKEND_##OpName); \
    if (!backend_str.empty())                                                           \
    {                                                                                   \
      ms_options.opcode_to_backend[ir::OpCode::OpName] = backend_str;                   \
    }                                                                                   \
  }
#include "ir/Operations.lst"
#undef OP

    // Index to Backend
    auto map_str = util::getConfigString(util::config::OP_BACKEND_MAP);
    auto key_val_list = nnfw::misc::split(map_str, ';');
    for (const auto &key_val_str : key_val_list)
    {
      if (key_val_str.empty())
      {
        continue;
      }

      auto key_val = nnfw::misc::split(key_val_str, '=');
      const auto &key_str = key_val.at(0);
      const auto &val = key_val.at(1);
      auto key = static_cast<uint32_t>(std::stoi(key_str));

      graph.operations().at(ir::OperationIndex{key}); // Check if exist, or this wil throw
      ms_options.index_to_backend.emplace(ir::OperationIndex{key}, val);
    }
  }
  return options;
}

Compiler::Compiler(const std::shared_ptr<ir::Graph> &graph)
    : _graph{graph}, _executors{nullptr}, _state{State::CREATED}
{

  // Set default values for CompilerOptions
  // All these default values should not be fetched from Env, when we stop supporting Android NN
  // API.
  _options = fetchCompilerOptionsFromGlobalConfig(*_graph);
}

void Compiler::checkProfilerConditions()
{
  if (!_options.he_scheduler)
    throw std::runtime_error("Heterogeneous scheduler must be enabled during profiling.");

  if (_options.executor != "Dataflow")
    throw std::runtime_error("Profiling mode works only with 'Dataflow' executor");
}

void Compiler::compile(void)
{
  _state = State::STARTED;

  /***************************************************
   * Prepare compilation phase
   ***************************************************/

  // The subgraphs being nullptr means that _graph has one subgraph
  if (_graph->subgraphs() == nullptr)
  {
    auto subgs = std::make_shared<onert::ir::Subgraphs>();
    subgs->push(onert::ir::SubgraphIndex{0}, _graph);
    _graph->setSubgraphs(subgs);
  }

  // Operation validation check
  OperationValidator{*_graph}();

  // Compilable check
  if (!checkCompilable())
  {
    // TODO Support multiple executors for interpreter
    _executors = std::make_shared<exec::ExecutorMap>();
    _executors->insert(
        std::make_pair(ir::SubgraphIndex{0}, std::make_unique<interp::InterpExecutor>(*_graph)));
    return;
  }

  // Mode check
  if (_options.he_profiling_mode)
    checkProfilerConditions();

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options.graph_dump_level);

  onert::dumper::dot::DotDumper dot_dumper(*_graph, dump_level);
  dot_dumper.dump("before_lower");

  // Lower: Assign backend
  auto lowered_graph = std::make_unique<ir::LoweredGraph>(*_graph, _options);

  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<ir::LoweredGraph>> lowered_subgs;
  _graph->subgraphs()->iterate([&](const ir::SubgraphIndex &index, ir::Graph &graph) {
    // Lower: Assign backend
    lowered_subgs[index] = std::make_unique<ir::LoweredGraph>(graph, _options);
    if (_options.fp16_enable)
    {
      // TODO Fill this
      // DO NOTHING
    }

    // NOTE. Current datas' reference of constant operands is 2 because of
    // original graph and lowered graph.
    // To delete cached data, this doing should be done for the original graph
    // at this line and then once again for the lowered graph in ExecutorFactory
    // TODO. Delete this code as code for disconnecting btw Graph and nnfw session lands
    if (_options.delete_cached_data)
    {
      CachedDataDeleter(graph.operands()).run();
    }
  });

  /*************************************************************
   *  Backend independent analysis & optimization phase finished
   *************************************************************/

  _state = State::LOWERED;

  _executors = std::make_shared<exec::ExecutorMap>();
  for (auto &pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;
    auto indexed_ranks = lowered_subg->indexed_ranks();

    onert::dumper::dot::DotDumper dot_dumper_lowered(lowered_subg.get(), dump_level);
    dot_dumper_lowered.dump("after_lower_subg-" + std::to_string(subg_index.value()));

    ir::OperationDumper dumper("START SUBGRAPH " + std::to_string(subg_index.value()));
    lowered_subg->graph().operations().iterate(
        [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });

    auto executor = std::unique_ptr<exec::IExecutor>{
        ExecutorFactory::get().create(std::move(lowered_subg), _options)};
    executor->setIndexedRanks(indexed_ranks);
    _executors->insert(std::make_pair(subg_index, std::move(executor)));
  }

  /********************************
   * Code generation phase finished
   ********************************/
  _state = State::COMPILED;
}

bool Compiler::checkCompilable()
{
  // Disable compile phase
  // When ready to use interpreter backend, remove this config and use backend setting
  if (_options.disable_compile)
  {
    return false;
  }

  // TODO check unspecified operand shape

  // Check compilable parameter
  ParamChecker paramChecker{_graph};
  paramChecker();
  if (paramChecker.haveNoneConstParam())
  {
    return false;
  }

  return true;
}

} // namespace compiler

} // namespace onert
