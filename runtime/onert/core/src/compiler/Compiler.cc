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
#include "Fp32ToFp16Converter.h"

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
  options.fp16_enable = util::getConfigBool(util::config::FP16_ENABLE);

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

Compiler::Compiler(const std::shared_ptr<ir::Subgraphs> &subgs)
    : _subgraphs{subgs}, _executors{nullptr}, _state{State::CREATED}
{

  // Set default values for CompilerOptions
  // All these default values should not be fetched from Env, when we stop supporting Android NN
  // API.
  // TODO Support multiple subgraphs
  _options = fetchCompilerOptionsFromGlobalConfig(*primary_subgraph());
}

void Compiler::enableToFp16() { _options.fp16_enable = true; }

void Compiler::checkProfilerConditions()
{
  if (!_options.he_scheduler)
    throw std::runtime_error("Heterogeneous scheduler must be enabled during profiling.");

  if (_options.executor != "Dataflow")
    throw std::runtime_error("Profiling mode works only with 'Dataflow' executor");
}

void Compiler::compile(void)
{
  {
    VERBOSE(Compiler) << std::boolalpha;
    VERBOSE(Compiler) << "==== Compiler Options ====" << std::endl;
    VERBOSE(Compiler) << "backend_list             : "
                      << nnfw::misc::join(_options.backend_list.begin(),
                                          _options.backend_list.end(), "/")
                      << std::endl;
    VERBOSE(Compiler) << "trace_filepath           : " << _options.trace_filepath << std::endl;
    VERBOSE(Compiler) << "graph_dump_level         : " << _options.graph_dump_level << std::endl;
    VERBOSE(Compiler) << "op_seq_max_node          : " << _options.op_seq_max_node << std::endl;
    VERBOSE(Compiler) << "executor                 : " << _options.executor << std::endl;
    VERBOSE(Compiler) << "manual_scheduler_options : (Too many things to print)" << std::endl;
    VERBOSE(Compiler) << "he_scheduler             : " << _options.he_scheduler << std::endl;
    VERBOSE(Compiler) << "he_profiling_mode        : " << _options.he_profiling_mode << std::endl;
    VERBOSE(Compiler) << "delete_cached_data       : " << _options.delete_cached_data << std::endl;
    VERBOSE(Compiler) << "disable_compile          : " << _options.disable_compile << std::endl;
    VERBOSE(Compiler) << "fp16_enable              : " << _options.fp16_enable << std::endl;
    VERBOSE(Compiler) << std::noboolalpha;
  }

  /***************************************************
   * Prepare compilation phase
   ***************************************************/

  // Operation validation check
  assert(_subgraphs);
  assert(_subgraphs->at(ir::SubgraphIndex{0}));
  _subgraphs->iterate([](const onert::ir::SubgraphIndex &, const onert::ir::Graph &graph) {
    OperationValidator{graph}();
  });

  // Compilable check
  if (!checkCompilable())
  {
    // TODO Support multiple executors for interpreter
    _executors = std::make_shared<exec::ExecutorMap>();
    _executors->insert(std::make_pair(
        ir::SubgraphIndex{0}, std::make_unique<interp::InterpExecutor>(*primary_subgraph())));
    _state = State::COMPILED;
    return;
  }

  // Mode check
  if (_options.he_profiling_mode)
    checkProfilerConditions();

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options.graph_dump_level);

  // Lower: Assign backend
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<ir::LoweredGraph>> lowered_subgs;
  _subgraphs->iterate([&](const ir::SubgraphIndex &index, ir::Graph &graph) {
    onert::dumper::dot::DotDumper dot_dumper(graph, dump_level);
    dot_dumper.dump(nnfw::misc::str("before_lower_subg-", index.value()));

    // Lower: Assign backend
    lowered_subgs[index] = std::make_unique<ir::LoweredGraph>(graph, _options);

    if (_options.fp16_enable)
    {
      // NOTE: the only acl_cl backend enables fp16 mode
      Fp32ToFp16Converter(*lowered_subgs[index]).run();
    }

    graph.setSubgraphs(nullptr);
  });

  _subgraphs.reset();

  /*************************************************************
   *  Backend independent analysis & optimization phase finished
   *************************************************************/

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
  for (uint32_t i = 0; i < _subgraphs->count(); ++i)
  {
    auto graph = _subgraphs->at(ir::SubgraphIndex{i});
    ParamChecker paramChecker{graph};
    paramChecker();
    if (paramChecker.haveNoneConstParam())
    {
      return false;
    }
  }

  return true;
}

} // namespace compiler

} // namespace onert
