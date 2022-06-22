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

#include "ExecutorFactory.h"
#include "ShapeValidator.h"
#include "pass/ConstantOutputPass.h"
#include "pass/OddOutputPass.h"
#include "pass/PassRunner.h"
#include "pass/UnusedOperandEliminationPass.h"
#include "../backend/builtin/Config.h"
#include "../dumper/dot/DotDumper.h"
#include "../interp/InterpExecutor.h"
#include "../ir/OperationCloner.h"
#include "../ir/OperationDumper.h"
#include "../ir/verifier/Verifier.h"

#include "compiler/StaticShapeInferer.h"
#include "util/ConfigSource.h"
#include "util/logging.h"

#include <misc/string_helpers.h>
#include <json/json.h>

// TODO Remove using fstream header
#include <fstream>

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

void verboseOptions(compiler::CompilerOptions &options)
{
  VERBOSE(Compiler) << std::boolalpha << "==== Compiler Options ====" << std::endl;
  VERBOSE(Compiler) << "backend_list             : "
                    << nnfw::misc::join(options.backend_list.begin(), options.backend_list.end(),
                                        "/")
                    << std::endl;
  VERBOSE(Compiler) << "trace_filepath           : " << options.trace_filepath << std::endl;
  VERBOSE(Compiler) << "graph_dump_level         : " << options.graph_dump_level << std::endl;
  VERBOSE(Compiler) << "executor                 : " << options.executor << std::endl;
  VERBOSE(Compiler) << "manual backend_for_all   : "
                    << options.manual_scheduler_options.backend_for_all << std::endl;
  VERBOSE(Compiler) << "manual_scheduler_options : "
                    << getOpBackends(options.manual_scheduler_options.opcode_to_backend)
                    << std::endl;
  VERBOSE(Compiler) << "he_scheduler             : " << options.he_scheduler << std::endl;
  VERBOSE(Compiler) << "he_profiling_mode        : " << options.he_profiling_mode << std::endl;
  VERBOSE(Compiler) << "disable_compile          : " << options.disable_compile << std::endl;
  VERBOSE(Compiler) << "fp16_enable              : " << options.fp16_enable << std::endl
                    << std::noboolalpha;
}

} // namespace

namespace onert
{

namespace compiler
{
void ManualSchedulerOptions::setBackendMap(const ir::Model &model, const std::string &str)
{
  // TODO Support multiple subgraphs for manual scheduling
  auto key_val_list = nnfw::misc::split(str, ';');
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

    model.at(ir::SubgraphIndex{0})
      ->operations()
      .at(ir::OperationIndex{key}); // Check if exist, or this wil throw
    this->index_to_backend.emplace(ir::OperationIndex{key}, val);
  }
}

void CompilerOptions::fetchCompilerOptionsFromGlobalConfig(const ir::Model &model)
{
  backend_list = nnfw::misc::split(util::getConfigString(util::config::BACKENDS), ';');
  trace_filepath = util::getConfigString(util::config::TRACE_FILEPATH);
  graph_dump_level = util::getConfigInt(util::config::GRAPH_DOT_DUMP);
  executor = util::getConfigString(util::config::EXECUTOR);
  he_scheduler = util::getConfigBool(util::config::USE_SCHEDULER);
  he_profiling_mode = util::getConfigBool(util::config::PROFILING_MODE);
  disable_compile = util::getConfigBool(util::config::DISABLE_COMPILE);
  fp16_enable = util::getConfigBool(util::config::FP16_ENABLE);
  {
    // Backend for all
    auto &ms_options = manual_scheduler_options;

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
    ms_options.setBackendMap(model, map_str);
  }
}

Compiler::Compiler(const std::shared_ptr<ir::Model> &model, CompilerOptions &copt)
  : _model{model}, _state{State::CREATED}, _options(copt)
{
  // DO NOTHING
}

void Compiler::enableToFp16() { _options.fp16_enable = true; }

void Compiler::checkProfilerConditions()
{
  if (!_options.he_scheduler)
    throw std::runtime_error("Heterogeneous scheduler must be enabled during profiling.");

  if (_options.executor != "Dataflow")
    throw std::runtime_error("Profiling mode works only with 'Dataflow' executor");
}

bool Compiler::buildPartialGraph(uint32_t num_graphs)
{
  if (_model->subgraphs_count() > 1)
    return false;

  auto partialgraphs = std::make_shared<ir::Model>();

  for (uint32_t idx = 0; idx < num_graphs; idx++)
  {
    auto partialgraph = std::make_unique<ir::Graph>();
    partialgraphs->push(ir::SubgraphIndex{idx}, std::move(partialgraph));
  }
  _model->primary_subgraph()->setPartialModel(partialgraphs);

  auto partial_graph = primary_subgraph()->partialgraphs();

  primary_subgraph()->operands().iterate(
    [&](const ir::OperandIndex &operand_index, const ir::Operand &operand) {
      auto use_operations = operand.getUses();

      for (auto use_operation : use_operations)
      {
        auto graph_index = _options.partial_graph_options.index_to_graph.find(use_operation);
        if (graph_index == _options.partial_graph_options.index_to_graph.end())
        {
          throw std::runtime_error("Invalid Partition Map");
        }
        auto partition = partial_graph->at(graph_index->second);

        if (partition->operands().exist(operand_index))
        {
          continue;
        }

        auto new_operand = std::make_unique<ir::Operand>(operand);
        new_operand->clearDefUse();
        auto new_operand_ind = partition->addOperand(operand_index, std::move(new_operand));
        UNUSED_RELEASE(new_operand_ind);
        assert(new_operand_ind == operand_index);
      }
    });

  primary_subgraph()->operations().iterate(
    [&](const ir::OperationIndex &operation_index, const ir::Operation &operation) {
      auto graph_index = _options.partial_graph_options.index_to_graph.find(operation_index);
      if (graph_index == _options.partial_graph_options.index_to_graph.end())
      {
        throw std::runtime_error("Invalid Partition Map");
      }
      auto partition = partial_graph->at(graph_index->second);

      auto operand_io = (operation.getInputs() + operation.getOutputs()) | ir::Remove::DUPLICATED |
                        ir::Remove::UNDEFINED;
      for (auto operand_index : operand_io)
      {
        if (partition->operands().exist(operand_index))
          continue;

        const auto &operand = primary_subgraph()->operands().at(operand_index);

        auto new_operand = std::make_unique<ir::Operand>(operand);
        new_operand->clearDefUse();

        auto new_operand_index = partition->addOperand(operand_index, std::move(new_operand));
        UNUSED_RELEASE(new_operand_index);
        assert(new_operand_index == operand_index);
      }

      auto new_operation_index = partition->addOperation(operation_index, clone(operation));
      UNUSED_RELEASE(new_operation_index);
      assert(new_operation_index == operation_index);
    });

  for (uint32_t idx = 0; idx < partial_graph->subgraphs_count(); idx++)
  {
    auto partition = partial_graph->at(ir::SubgraphIndex{idx});

    partition->operands().iterate([&](const ir::OperandIndex &operand_index,
                                      const ir::Operand &operand) {
      if (primary_subgraph()->getInputs().contains(operand_index) ||
          (!operand.getDef().valid() && !operand.isConstant()))
      {
        partition->addInput(operand_index, primary_subgraph()->tensor_names()->at(operand_index));
      }
      if (primary_subgraph()->getOutputs().contains(operand_index) || operand.getUses().size() == 0)
      {
        partition->addOutput(operand_index, primary_subgraph()->tensor_names()->at(operand_index));
      }

      if (primary_subgraph()->operands().at(operand_index).getUses().size() > 1 &&
          !primary_subgraph()->operands().at(operand_index).isConstant() &&
          !partition->getInputs().contains(operand_index))
      {
        auto use_operations = primary_subgraph()->operands().at(operand_index).getUses();
        auto iter = use_operations.begin();
        ir::SubgraphIndex graph_index =
          _options.partial_graph_options.index_to_graph.find(*iter++)->second;
        while (iter != use_operations.end())
        {
          if (graph_index != _options.partial_graph_options.index_to_graph.find(*iter)->second &&
              !partition->getOutputs().contains(operand_index))
          {
            partition->addOutput(operand_index,
                                 primary_subgraph()->tensor_names()->at(operand_index));
          }
          iter++;
        }
      }
    });

    partition->verify();

    bool same = true;
    if (partition->getInputs().size() == primary_subgraph()->getInputs().size())
    {
      for (auto iter = partition->getInputs().begin(); iter != partition->getInputs().end(); ++iter)
      {
        if (!primary_subgraph()->getInputs().contains(*iter))
        {
          same = false;
          break;
        }
      }
      if (same == true)
      {
        partition->getInputs() = primary_subgraph()->getInputs();
      }
      else
      {
        partition->input_sort();
      }
    }

    same = true;
    if (partition->getOutputs().size() == primary_subgraph()->getOutputs().size())
    {
      for (auto iter = partition->getOutputs().begin(); iter != partition->getOutputs().end();
           ++iter)
      {
        if (!primary_subgraph()->getOutputs().contains(*iter))
        {
          same = false;
          break;
        }
      }
      if (same == true)
      {
        partition->getOutputs() = primary_subgraph()->getOutputs();
      }
      else
      {
        partition->output_sort();
      }
    }
  }
  return true;
}

std::shared_ptr<CompilerArtifact> Compiler::compile(void)
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

  // FIXME This is a workaround for bulk operations, should remove it
  {
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::Bulk] = "trix";
  }

  verboseOptions(_options);

  _model->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
    // Mandatory passes
    pass::PassRunner{}
      .append(std::make_unique<pass::ConstantOutputPass>(subg))
      .append(std::make_unique<pass::OddOutputPass>(subg))
      .run();

    // Optimizations
    pass::PassRunner{}.append(std::make_unique<pass::UnusedOperandEliminationPass>(subg)).run();
  });

  /***************************************************
   * Prepare compilation phase
   ***************************************************/
  // Compilable check
  // TODO: Support hybrid execution -
  //       execution between interpreter and compiled executor (including control flow)
  if (_options.disable_compile)
  {
    auto executors = std::make_shared<exec::ExecutorMap>();

    _model->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      executors->emplace(index, std::make_unique<interp::InterpExecutor>(subg));
      subg.setModel(_model);
    });
    _state = State::COMPILED;
    return std::make_shared<CompilerArtifact>(executors, nullptr);
  }

  // Mode check
  if (_options.he_profiling_mode)
    checkProfilerConditions();

  // Tracing context
  auto tracing_ctx = std::make_unique<util::TracingCtx>(_model.get());

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options.graph_dump_level);

  // Lower: Assign backend
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>> lowered_subgs;
  _model->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
    onert::dumper::dot::DotDumper dot_dumper(subg, dump_level);
    dot_dumper.dump(nnfw::misc::str("before_lower_subg-", index.value()));

    // Lower: Assign backend
    lowered_subgs[index] = std::make_unique<compiler::LoweredGraph>(subg, _options);

    // Set tracing_ctx for copied graph
    tracing_ctx->setSubgraphIndex(&(lowered_subgs[index]->graph()), index.value());

    subg.setModel(nullptr);
  });

  _model.reset();

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
    auto &lowered_subg = lowered_subgs.at(primary_subg_idx);
    auto ordered_ops = lowered_subg->graph().topolSortOperations();
    for (auto op_ind : ordered_ops)
    {
      const auto &op = lowered_subg->graph().operations().at(op_ind);
      bool has_dynamic_tensor = inferer.infer(op);
      lowered_subg->setHasDynamicTensor(op_ind, has_dynamic_tensor);
    }
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

  auto executors = std::make_shared<exec::ExecutorMap>();
  for (auto &pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;
    auto indexed_ranks = lowered_subg->indexed_ranks();

    ir::OperationDumper dumper("Executor generation of Subgraph " +
                               std::to_string(subg_index.value()));
    lowered_subg->graph().operations().iterate(
      [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });
    auto executor = std::unique_ptr<exec::IExecutor>{ExecutorFactory::get().create(
      std::move(lowered_subg), tracing_ctx.get(), _options, executors)};
    executor->setIndexedRanks(indexed_ranks);
    executors->insert(std::make_pair(subg_index, std::move(executor)));
  }

  /********************************
   * Code generation phase finished
   ********************************/
  _state = State::COMPILED;
  return std::make_shared<CompilerArtifact>(executors, std::move(tracing_ctx));
}

std::vector<std::shared_ptr<CompilerArtifact>> Compiler::compile(const char *package_file_path,
                                                                 const char *map_file_path)
{
  std::string package_path(package_file_path);
  std::string partition_map_file;

  if (map_file_path)
  {
    partition_map_file = map_file_path;
  }
  else
  {
    partition_map_file = package_path + "/partition_map.json";
  }

  std::ifstream pmfs(partition_map_file);
  Json::Value root;
  pmfs >> root;
  const Json::Value &map = root["partition_map"];
  const Json::Value &np = root["num_partitions"];

  uint32_t num_graphs = 1;

  if (pmfs.is_open())
  {
    num_graphs = np.asUInt();
    for (uint32_t i = 0; i < (uint32_t)map.size(); ++i)
    {
      _options.partial_graph_options.index_to_graph[ir::OperationIndex{i}] =
        ir::SubgraphIndex{map[i].asUInt()};
    }
  }
  else
  {
    throw std::runtime_error("There is no partition map file");
  }

  if (!buildPartialGraph(num_graphs))
  {
    throw std::runtime_error("It doesn't support in case there are subgraphs");
  }

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

  // FIXME This is a workaround for bulk operations, should remove it
  {
    _options.manual_scheduler_options.opcode_to_backend[ir::OpCode::Bulk] = "trix";
  }

  verboseOptions(_options);

  _model->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
    // Mandatory passes
    auto part = subg.partialgraphs();
    part->iterate([&](const ir::SubgraphIndex &, ir::Graph &partialgraph) {
      pass::PassRunner{}
        .append(std::make_unique<pass::ConstantOutputPass>(partialgraph))
        .append(std::make_unique<pass::OddOutputPass>(partialgraph))
        .run();

      // Optimizations
      pass::PassRunner{}
        .append(std::make_unique<pass::UnusedOperandEliminationPass>(partialgraph))
        .run();
    });
  });

  /***************************************************
   * Prepare compilation phase
   ***************************************************/

  // Compilable check
  // TODO: Support hybrid execution -
  //       execution between interpreter and compiled executor (including control flow)
  if (_options.disable_compile)
  {
    std::vector<std::shared_ptr<CompilerArtifact>> results;
    auto executors = std::make_shared<exec::ExecutorMap>();

    _model->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      executors->emplace(index, std::make_unique<interp::InterpExecutor>(subg));
    });
    results.push_back(std::make_shared<CompilerArtifact>(executors, nullptr));
    _state = State::COMPILED;
    return results;
  }

  // Mode check
  if (_options.he_profiling_mode)
    checkProfilerConditions();

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options.graph_dump_level);

  // Lower: Assign backend
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>>
    lowered_partialgraphs;
  _model->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
    auto part = subg.partialgraphs();
    part->iterate([&](const ir::SubgraphIndex &pindex, ir::Graph &partialgraph) {
      onert::dumper::dot::DotDumper dot_dumper_part(partialgraph, dump_level);
      dot_dumper_part.dump(nnfw::misc::str("before_lower_subg_partialgraph-", pindex.value()));

      // // Lower: Assign backend
      lowered_partialgraphs[pindex] =
        std::make_unique<compiler::LoweredGraph>(subg, partialgraph, _options);
      partialgraph.setModel(nullptr);
    });
  });

  for (auto &pair : lowered_partialgraphs)
  {

    const auto &partialgraph_index = pair.first;
    auto &lowered_partialgraph = pair.second;
    onert::dumper::dot::DotDumper dot_dumper_lowered_part(lowered_partialgraph.get(), dump_level);
    dot_dumper_lowered_part.dump("after_lower_subg_partialgraph-" +
                                 std::to_string(partialgraph_index.value()));
  }

  // Partial Graph shape inference
  for (auto &pair : lowered_partialgraphs)
  {
    const auto &partialgraph_index = pair.first;
    auto &lowered_partialgraph = pair.second;
    StaticShapeInferer partial_inferer(partialgraph_index, lowered_partialgraphs);
    auto ordered_ops = lowered_partialgraph->graph().topolSortOperations();
    for (auto op_ind : ordered_ops)
    {
      const auto &op = lowered_partialgraph->graph().operations().at(op_ind);
      bool has_dynamic_tensor = partial_inferer.infer(op);
      lowered_partialgraph->setHasDynamicTensor(op_ind, has_dynamic_tensor);
    }
    partial_inferer.dump();
  }

  // Shape validation
  // TODO Move shape independent feature check from ShapeValidator to OperationValidator
  // TODO Move ShapeValidator into shape inference
  //      - Check input tensor shape validation
  //      - Check parameter value validation which valid value is depend on input tensor shape
  //      - Output tensor shape validation check is needless because
  //        static/dynamic shape inferer will make valid output shape
  for (auto &pair : lowered_partialgraphs)
  {
    auto &lowered_partialgraph = pair.second;
    compiler::ShapeValidator{lowered_partialgraph->graph()}();
  }

  /*************************************************************
   *  Backend independent analysis & optimization phase finished
   *************************************************************/
  std::map<uint32_t, std::unique_ptr<compiler::LoweredGraph>> ordered;
  for (auto &pair : lowered_partialgraphs)
  {
    // const auto &partialgraph_index = pair.first;
    auto &lowered_partialgraph = pair.second;

    ordered.insert(make_pair(pair.first.value(), std::move(lowered_partialgraph)));
  }

  std::vector<std::shared_ptr<CompilerArtifact>> results;
  for (auto &pair : ordered)
  {
    auto executors = std::make_shared<exec::ExecutorMap>();

    const auto &partialgraph_index = ir::SubgraphIndex(pair.first);
    auto &lowered_partialgraph = pair.second;
    auto indexed_ranks = lowered_partialgraph->indexed_ranks();
    ir::OperationDumper dumper("Executor generation of Subgraph " +
                               std::to_string(partialgraph_index.value()));
    lowered_partialgraph->graph().operations().iterate(
      [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });
    auto executor = std::unique_ptr<exec::IExecutor>{
      ExecutorFactory::get().create(std::move(lowered_partialgraph), nullptr, _options, executors)};
    executor->setIndexedRanks(indexed_ranks);
    executors->insert(std::make_pair(ir::SubgraphIndex{0}, std::move(executor)));

    // It doesn't support tracing in case of partial graph
    results.push_back(std::make_shared<CompilerArtifact>(executors, nullptr));
  }

  _model.reset();
  /********************************
   * Code generation phase finished
   ********************************/
  _state = State::COMPILED;

  return results;
}

} // namespace compiler

} // namespace onert
