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
#include "ShapeValidator.h"
#include "Fp32ToFp16Converter.h"

#include <backend/builtin/Config.h>
#include "compiler/BackendManager.h"
#include "compiler/IScheduler.h"
#include "compiler/ManualScheduler.h"
#include "compiler/HEScheduler.h"
#include "compiler/StaticShapeInferer.h"
#include "compiler/OpSequenceLowerInfo.h"
#include "compiler/pass/ConstantOutputPass.h"
#include "compiler/pass/OddOutputPass.h"
#include "compiler/pass/PassRunner.h"
#include "exec/ExecTime.h"
#include "ir/verifier/Verifier.h"
#include "dumper/dot/DotDumper.h"
#include "compiler/Linear.h"
#include "interp/InterpExecutor.h"
#include "util/ConfigSource.h"
#include "util/logging.h"
#include "ir/OperationDumper.h"
#include "misc/string_helpers.h"

namespace
{

using namespace onert;

std::string getOpBackends(std::unordered_map<ir::OpCode, std::string> &opcode_to_backend)
{
  std::unordered_map<ir::OpCode, std::string>::iterator it;
  std::string opbackends;

  for (it = opcode_to_backend.begin(); it != opcode_to_backend.end(); ++it)
  {
    if (!opbackends.empty())
      opbackends = opbackends + ", ";

    auto opcode = it->first;
    const std::string opname = ir::toString(opcode);
    opbackends += opname + "=" + it->second;
  }
  return opbackends;
}

} // namespace

namespace onert
{

namespace compiler
{

CompilerOptions fetchCompilerOptionsFromGlobalConfig(const ir::Subgraphs &subgs)
{
  CompilerOptions options;
  options.backend_list = nnfw::misc::split(util::getConfigString(util::config::BACKENDS), ';');
  options.trace_filepath = util::getConfigString(util::config::TRACE_FILEPATH);
  options.graph_dump_level = util::getConfigInt(util::config::GRAPH_DOT_DUMP);
  options.executor = util::getConfigString(util::config::EXECUTOR);
  options.he_scheduler = util::getConfigBool(util::config::USE_SCHEDULER);
  options.he_profiling_mode = util::getConfigBool(util::config::PROFILING_MODE);
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
    // TODO Support multiple subgraphs for manual scheduling
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

      subgs.at(ir::SubgraphIndex{0})
        ->operations()
        .at(ir::OperationIndex{key}); // Check if exist, or this wil throw
      ms_options.index_to_backend.emplace(ir::OperationIndex{key}, val);
    }
  }
  return options;
}

Compiler::Compiler(const std::shared_ptr<ir::Subgraphs> &subgs, util::TracingCtx *tracing_ctx)
  : _subgraphs{subgs}, _state{State::CREATED}
{
  // Set default values for CompilerOptions
  // All these default values should not be fetched from Env, when we stop supporting Android NN
  // API.
  _options = fetchCompilerOptionsFromGlobalConfig(*subgs);

  _options.tracing_ctx = tracing_ctx;
}

void Compiler::enableToFp16() { _options.fp16_enable = true; }

void Compiler::checkProfilerConditions()
{
  if (!_options.he_scheduler)
    throw std::runtime_error("Heterogeneous scheduler must be enabled during profiling.");

  if (_options.executor != "Dataflow")
    throw std::runtime_error("Profiling mode works only with 'Dataflow' executor");
}

std::shared_ptr<exec::ExecutorMap> Compiler::compile(void)
{
  // Set control flow backend for control flow operators
  {
    auto &builtin_id = backend::builtin::Config::ID;
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::If] = builtin_id;
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::While] = builtin_id;
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::Permute] = builtin_id;
  }

  // FIXME This is a workaround for bcq operations, should remove it
  {
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQFullyConnected] = "bcq";
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQGather] = "bcq";
  }

  {
    VERBOSE(Compiler) << std::boolalpha << "==== Compiler Options ====" << std::endl;
    VERBOSE(Compiler) << "backend_list             : "
                      << nnfw::misc::join(_options.backend_list.begin(),
                                          _options.backend_list.end(), "/")
                      << std::endl;
    VERBOSE(Compiler) << "trace_filepath           : " << _options.trace_filepath << std::endl;
    VERBOSE(Compiler) << "graph_dump_level         : " << _options.graph_dump_level << std::endl;
    VERBOSE(Compiler) << "executor                 : " << _options.executor << std::endl;
    VERBOSE(Compiler) << "manual backend_for_all   : "
                      << _options.manual_scheduler_options.backend_for_all << std::endl;
    VERBOSE(Compiler) << "manual_scheduler_options : "
                      << getOpBackends(_options.manual_scheduler_options.opcode_to_backend)
                      << std::endl;
    VERBOSE(Compiler) << "he_scheduler             : " << _options.he_scheduler << std::endl;
    VERBOSE(Compiler) << "he_profiling_mode        : " << _options.he_profiling_mode << std::endl;
    VERBOSE(Compiler) << "disable_compile          : " << _options.disable_compile << std::endl;
    VERBOSE(Compiler) << "fp16_enable              : " << _options.fp16_enable << std::endl
                      << std::noboolalpha;
  }

  _subgraphs->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
    // Mandatory passes
    pass::PassRunner{}
      .append(std::make_unique<pass::ConstantOutputPass>(subg))
      .append(std::make_unique<pass::OddOutputPass>(subg))
      .run();
  });

  /***************************************************
   * Prepare compilation phase
   ***************************************************/
  auto executors = std::make_shared<exec::ExecutorMap>();

  // Compilable check
  // TODO: Support hybrid execution -
  //       execution between interpreter and compiled executor (including control flow)
  if (!checkCompilable())
  {
    _subgraphs->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      executors->emplace(index, std::make_unique<interp::InterpExecutor>(subg));
    });
    _state = State::COMPILED;
    return executors;
  }

  // Mode check
  if (_options.he_profiling_mode)
    checkProfilerConditions();

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options.graph_dump_level);

  // Lower: Assign backend
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>> lowered_subgs;
  _subgraphs->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
    onert::dumper::dot::DotDumper dot_dumper(subg, dump_level);
    dot_dumper.dump(nnfw::misc::str("before_lower_subg-", index.value()));

    // Lower: Assign backend
    lowered_subgs[index] = std::make_unique<compiler::LoweredGraph>(subg, _options);

    // Check backend(s) for subgraph support FP16
    bool backends_support_fp16 = true;
    auto all_backends = BackendManager::get().getAll();
    for (auto backend : all_backends)
    {
      // Builtin backend is not for actual computaion of operations so it is an exception
      if (backend->config()->id() != backend::builtin::Config::ID)
        backends_support_fp16 &= backend->config()->supportFP16();
    }

    if (_options.fp16_enable && backends_support_fp16)
    {
      // NOTE: the only acl_cl backend enables fp16 mode
      Fp32ToFp16Converter(*lowered_subgs[index]).run();
    }

    subg.setSubgraphs(nullptr);
  });

  _subgraphs.reset();

  for (auto &pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;
    onert::dumper::dot::DotDumper dot_dumper_lowered(lowered_subg.get(), dump_level);
    dot_dumper_lowered.dump("after_lower_subg-" + std::to_string(subg_index.value()));
  }

  // Shape inference.
  {
    const auto primary_subg_idx = ir::SubgraphIndex{0};
    StaticShapeInferer inferer(primary_subg_idx, lowered_subgs);
    lowered_subgs.at(primary_subg_idx)
      ->iterateTopolOpSeqs([&](const ir::OpSequenceIndex &, ir::OpSequence &op_seq) {
        auto has_dynamic_tensor = inferer.infer(op_seq);
        op_seq.has_dynamic_tensor(has_dynamic_tensor);
      });
    inferer.dump();
  }

  // Shape validation
  // TODO Move shape independent feature check from ShapeValidator to OperationValidator
  // TODO Move ShapeValidator into shape inference
  //      - Check input tensor shape validation
  //      - Check parameter value validation which valid value is depend on input tensor shape
  //      - Output tensor shape validation check is needless because
  //        static/dynamic shape inferer will make valid output shape
  for (auto &pair : lowered_subgs)
  {
    auto &lowered_subg = pair.second;
    compiler::ShapeValidator{lowered_subg->graph()}();
  }

  /*************************************************************
   *  Backend independent analysis & optimization phase finished
   *************************************************************/

  executors = std::make_shared<exec::ExecutorMap>();
  for (auto &pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;
    auto indexed_ranks = lowered_subg->indexed_ranks();

    ir::OperationDumper dumper("Executor generation of Subgraph " +
                               std::to_string(subg_index.value()));
    lowered_subg->graph().operations().iterate(
      [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });
    auto executor = std::unique_ptr<exec::IExecutor>{
      ExecutorFactory::get().create(std::move(lowered_subg), _options, executors)};
    executor->setIndexedRanks(indexed_ranks);
    executors->insert(std::make_pair(subg_index, std::move(executor)));
  }

  /********************************
   * Code generation phase finished
   ********************************/
  _state = State::COMPILED;
  return executors;
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
