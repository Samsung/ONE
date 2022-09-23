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

#include <misc/polymorphic_downcast.h>
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

std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::StaticShapeInferer>>
createStaticShapeInferers(
  const std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>>
    &lowered_subgs)
{
  // Allocate StaticShapeInferer per each subgraph
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::StaticShapeInferer>> inferers;
  for (auto &pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;
    inferers[subg_index] = std::make_unique<compiler::StaticShapeInferer>(lowered_subg.get());
  }

  // Append observers in all StaticShapeInferers
  for (auto &pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;

    // TODO: Change this iteration for all to controlflow iteration
    lowered_subg->graph().operations().iterate([&](const ir::OperationIndex &,
                                                   const ir::Operation &op) {
      // A Function to append child inferers. These make it possible for a StaticShapeInferer to
      // call StaticShapeInferes of child subgraphs recursively
      auto appendChildInferer = [&](const ir::SubgraphIndex &child_subg_idx) {
        auto *child_inferer = inferers.at(child_subg_idx).get();
        inferers.at(subg_index)->appendChildInferer(child_subg_idx, child_inferer);
      };

      // A Function to appaend subg input observers. This makes it possible for a StaticShapeInferer
      // to update inputs of child subgraphs
      auto appendSubgraphInputObserver = [&](const ir::SubgraphIndex &child_subg_idx) {
        std::vector<ir::Operand *> child_subg_inputs;
        auto &child_subg = lowered_subgs.at(child_subg_idx)->graph();
        for (const auto &input_idx : child_subg.getInputs())
        {
          auto operand_ptr = child_subg.operands().getRawPtr(input_idx);
          child_subg_inputs.emplace_back(operand_ptr);
        }
        inferers.at(subg_index)
          ->appendSubgInputObserver(child_subg_idx,
                                    std::make_unique<compiler::OperandObserver>(child_subg_inputs));
      };

      // A Function to set controlflow output observers. This makes it possible for a
      // StaticShapeInferer to update outputs of parent controlflow opeerations
      auto setControlFlowOutputObserver = [&](const ir::SubgraphIndex &child_subg_idx) {
        std::vector<ir::Operand *> cf_outputs;
        auto &subg = lowered_subg->graph();
        for (const auto &output_idx : op.getOutputs())
        {
          auto operand_ptr = subg.operands().getRawPtr(output_idx);
          cf_outputs.emplace_back(operand_ptr);
        }
        inferers.at(child_subg_idx)
          ->setControlflowOutputObserver(std::make_unique<compiler::OperandObserver>(cf_outputs));
      };

      // Append Observers in a StaticShapeInferer
      if (op.opcode() == ir::OpCode::If)
      {
        const auto &if_op = nnfw::misc::polymorphic_downcast<const ir::operation::If &>(op);

        appendChildInferer(if_op.param().then_subg_index);
        appendChildInferer(if_op.param().else_subg_index);

        appendSubgraphInputObserver(if_op.param().then_subg_index);
        appendSubgraphInputObserver(if_op.param().else_subg_index);

        setControlFlowOutputObserver(if_op.param().then_subg_index);
      }
      else if (op.opcode() == ir::OpCode::While)
      {
        const auto &while_op = nnfw::misc::polymorphic_downcast<const ir::operation::While &>(op);

        appendChildInferer(while_op.param().cond_subg_index);
        appendChildInferer(while_op.param().body_subg_index);

        appendSubgraphInputObserver(while_op.param().cond_subg_index);
        appendSubgraphInputObserver(while_op.param().body_subg_index);

        setControlFlowOutputObserver(while_op.param().body_subg_index);
      }
    });
  }

  return inferers;
}

} // namespace

namespace onert
{

namespace compiler
{
void ManualSchedulerOptions::setBackendMap(const std::string &str)
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
    this->index_to_backend.emplace(ir::OperationIndex{key}, val);
  }
}

std::unique_ptr<CompilerOptions> CompilerOptions::fromGlobalConfig()
{
  auto o = std::make_unique<CompilerOptions>();
  o->backend_list = nnfw::misc::split(util::getConfigString(util::config::BACKENDS), ';');
  o->trace_filepath = util::getConfigString(util::config::TRACE_FILEPATH);
  o->graph_dump_level = util::getConfigInt(util::config::GRAPH_DOT_DUMP);
  o->executor = util::getConfigString(util::config::EXECUTOR);
  o->he_scheduler = util::getConfigBool(util::config::USE_SCHEDULER);
  o->he_profiling_mode = util::getConfigBool(util::config::PROFILING_MODE);
  o->disable_compile = util::getConfigBool(util::config::DISABLE_COMPILE);
  o->fp16_enable = util::getConfigBool(util::config::FP16_ENABLE);
  {
    // Backend for all
    auto &ms_options = o->manual_scheduler_options;

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
    ms_options.setBackendMap(map_str);
  }
  return o;
}

Compiler::Compiler(const std::shared_ptr<ir::Model> &model, CompilerOptions &copt)
  : _nnpkg{std::make_shared<ir::NNPkg>(model)}, _state{State::CREATED}, _voptions{&copt}
{
  // DO NOTHING
}

Compiler::Compiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
                   std::vector<std::unique_ptr<CompilerOptions>> &copts)
  : _nnpkg{nnpkg}, _state{State::CREATED}, _voptions{}
{
  for (uint32_t i = 0; i < copts.size(); i++)
  {
    _voptions.push_back(copts[i].get());
  }
}

void Compiler::enableToFp16()
{
  for (auto options : _voptions)
    options->fp16_enable = true;
}

void Compiler::checkProfilerConditions()
{
  if (_nnpkg->model_count() != 1)
    throw std::runtime_error("NYI: Profiling mode for multiple model is not supported yet");

  auto &options = *_voptions[0];

  if (options.he_scheduler)
    throw std::runtime_error("Heterogeneous scheduler must be enabled during profiling.");

  if (options.executor != "Dataflow")
    throw std::runtime_error("Profiling mode works only with 'Dataflow' executor");
}

bool Compiler::buildPartialGraph(uint32_t num_graphs)
{
  // Use 1st model and options only on partial graph (pipeline) compile
  assert(_nnpkg->model_count() == 1);
  assert(_voptions.size() == 1);

  auto model = _nnpkg->primary_model();
  auto &options = *_voptions[0];

  if (model->subgraphs_count() > 1)
    return false;

  auto partialgraphs = std::make_shared<ir::Model>();

  for (uint32_t idx = 0; idx < num_graphs; idx++)
  {
    auto partialgraph = std::make_unique<ir::Graph>();
    partialgraphs->push(ir::SubgraphIndex{idx}, std::move(partialgraph));
  }
  model->primary_subgraph()->setPartialModel(partialgraphs);

  auto partial_graph = primary_subgraph()->partialgraphs();

  primary_subgraph()->operands().iterate(
    [&](const ir::OperandIndex &operand_index, const ir::Operand &operand) {
      auto use_operations = operand.getUses();

      for (auto use_operation : use_operations)
      {
        auto graph_index = options.partial_graph_options.index_to_graph.find(use_operation);
        if (graph_index == options.partial_graph_options.index_to_graph.end())
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
      auto graph_index = options.partial_graph_options.index_to_graph.find(operation_index);
      if (graph_index == options.partial_graph_options.index_to_graph.end())
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
          options.partial_graph_options.index_to_graph.find(*iter++)->second;
        while (iter != use_operations.end())
        {
          if (graph_index != options.partial_graph_options.index_to_graph.find(*iter)->second &&
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
  for (auto options : _voptions)
  {
    // Set control flow backend for control flow operators
    auto &builtin_id = backend::builtin::Config::ID;
    options->manual_scheduler_options.opcode_to_backend[ir::OpCode::If] = builtin_id;
    options->manual_scheduler_options.opcode_to_backend[ir::OpCode::While] = builtin_id;
    options->manual_scheduler_options.opcode_to_backend[ir::OpCode::Permute] = builtin_id;

    // FIXME This is a workaround for bcq operations, should remove it
    options->manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQFullyConnected] = "bcq";
    options->manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQGather] = "bcq";

    // FIXME This is a workaround for bulk operations, should remove it
    options->manual_scheduler_options.opcode_to_backend[ir::OpCode::Bulk] = "trix";

    verboseOptions(*options);
  }

  // NYI: allow one model compilation
  auto const model_count = _nnpkg->model_count();
  if (model_count != _voptions.size())
    throw std::runtime_error{"Model count and option vector size mismatch"};

  for (uint32_t i = 0; i < model_count; i++)
  {
    _nnpkg->model(ir::ModelIndex{i})->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
      // Mandatory passes
      pass::PassRunner{}
        .append(std::make_unique<pass::ConstantOutputPass>(subg))
        .append(std::make_unique<pass::OddOutputPass>(subg))
        .run();

      // Optimizations
      pass::PassRunner{}.append(std::make_unique<pass::UnusedOperandEliminationPass>(subg)).run();
    });
  }

  /***************************************************
   * Prepare compilation phase
   ***************************************************/
  // Compilable check
  // TODO: Support hybrid execution -
  //       execution between interpreter and compiled executor (including control flow)
  if (_voptions[0]->disable_compile)
  {
    if (model_count > 1)
      throw std::runtime_error{"NYI: Disable compilation for multi model is not supported yet"};

    auto executors = std::make_shared<exec::Executors>();

    _nnpkg->primary_model()->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      executors->emplace(index, std::make_unique<interp::InterpExecutor>(subg));
    });
    _state = State::COMPILED;
    return std::make_shared<CompilerArtifact>(executors, nullptr);
  }

  // Mode check
  // TODO handle option for each model
  if (_voptions[0]->he_profiling_mode)
    checkProfilerConditions();

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  // TODO Handle dump level for each model
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_voptions[0]->graph_dump_level);
  onert::dumper::dot::DotDumper dot_dumper(dump_level);

  // Tracing context
  auto tracing_ctx = std::make_unique<util::TracingCtx>();

  // Model edge context
  std::unique_ptr<ir::ModelEdges> model_edges = nullptr;

  // Lower: Assign backend
  std::unordered_map<ir::ModelIndex,
                     std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>>>
    lowered_subgs;

  if (model_count != 1)
  {
    // TODO Support tracing_ctx for multiple model
    tracing_ctx = nullptr;

    // Copy model edge context
    model_edges = std::make_unique<ir::ModelEdges>(_nnpkg->model_edges());
  }

  for (uint32_t i = 0; i < model_count; i++)
  {
    auto const index_m = ir::ModelIndex{i};
    auto model = _nnpkg->model(ir::ModelIndex{i});

    model->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      dot_dumper.dump(subg, nnfw::misc::str("before_lower_model-", i, "-subg-", index.value()));
      // Lower: Assign backend
      lowered_subgs[index_m][index] = std::make_unique<compiler::LoweredGraph>(subg, *_voptions[0]);
      // Set tracing_ctx for copied graph
      if (tracing_ctx != nullptr)
        tracing_ctx->setSubgraphIndex(&(lowered_subgs[index_m][index]->graph()), index.value());
    });
  }

  _nnpkg.reset();

  for (auto &pair : lowered_subgs)
  {
    const auto &index_m = pair.first;
    auto &model_lsubg = pair.second;

    for (auto &pair_inner : model_lsubg)
    {
      const auto &subg_index = pair_inner.first;
      auto &lowered_subg = pair_inner.second;
      dot_dumper.dump(*lowered_subg, nnfw::misc::str("after_lower_model-", index_m.value(),
                                                     "-subg-", subg_index.value()));
    }
  }

  // Shape inference.
  for (auto &pair : lowered_subgs)
  {
    auto &model_lsubgs = pair.second;
    // Run the StaticShapeInfer of primary subg. All child StaticShapeInferers are called
    // recursively
    std::unordered_map<ir::SubgraphIndex, std::unique_ptr<StaticShapeInferer>> inferers =
      createStaticShapeInferers(model_lsubgs);

    const auto primary_subg_idx = ir::SubgraphIndex{0};
    inferers.at(primary_subg_idx)->infer();

    for (const auto &pair_inferer : inferers)
    {
      const auto inferer = pair_inferer.second.get();
      inferer->dump();
    }
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
    auto &model_lsubgs = pair.second;

    for (auto &pair_inner : model_lsubgs)
    {
      auto &lowered_subg = pair_inner.second;
      compiler::ShapeValidator{lowered_subg->graph()}();
    }
  }

  /*************************************************************
   *  Backend independent analysis & optimization phase finished
   *************************************************************/
  auto executors = std::make_shared<exec::Executors>(std::move(model_edges));
  for (auto &pair : lowered_subgs)
  {
    auto const &model_index = pair.first;
    auto &model_lsubgs = pair.second;

    if (model_count != 1 && model_lsubgs.size() != 1)
      throw std::runtime_error{
        "NYI: Executor generation of multiple model including controlflow is not supported yet"};

    for (auto &pair_inner : model_lsubgs)
    {
      auto const subg_index = pair_inner.first;
      auto &lowered_subg = pair_inner.second;
      auto const indexed_ranks = lowered_subg->indexed_ranks();

      ir::OperationDumper dumper("Executor generation of Subgraph " +
                                 std::to_string(subg_index.value()));
      lowered_subg->graph().operations().iterate(
        [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });

      auto &options = (model_count > 1) ? *_voptions[subg_index.value()] : *_voptions[0];
      auto executor = std::unique_ptr<exec::IExecutor>{ExecutorFactory::get().create(
        std::move(lowered_subg), tracing_ctx.get(), options, executors)};
      executor->setIndexedRanks(indexed_ranks);

      // executors can handle multiple model by passing model index as subgraph index
      // So executors cannot handle multiple model with controlflow
      // TODO Pass model index and subg index to executors
      (model_count == 1)
        ? executors->emplace(subg_index, std::move(executor))
        : executors->emplace(ir::SubgraphIndex{model_index.value()}, std::move(executor));
    }
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
  // Allow one model compilation for pipeline
  if (_nnpkg->model_count() != 1)
    throw std::runtime_error{"Multiple models compilation for pipeline is not supported yet."};
  assert(_voptions.size() == 1);

  auto model = _nnpkg->primary_model();
  auto &options = *_voptions[0];

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
      options.partial_graph_options.index_to_graph[ir::OperationIndex{i}] =
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
    options.manual_scheduler_options.opcode_to_backend[ir::OpCode::If] = builtin_id;
    options.manual_scheduler_options.opcode_to_backend[ir::OpCode::While] = builtin_id;
    options.manual_scheduler_options.opcode_to_backend[ir::OpCode::Permute] = builtin_id;
  }

  // FIXME This is a workaround for bcq operations, should remove it
  {
    options.manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQFullyConnected] = "bcq";
    options.manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQGather] = "bcq";
  }

  // FIXME This is a workaround for bulk operations, should remove it
  {
    options.manual_scheduler_options.opcode_to_backend[ir::OpCode::Bulk] = "trix";
  }

  verboseOptions(options);

  model->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
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
  if (options.disable_compile)
  {
    std::vector<std::shared_ptr<CompilerArtifact>> results;
    auto executors = std::make_shared<exec::Executors>();

    model->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      executors->emplace(index, std::make_unique<interp::InterpExecutor>(subg));
    });
    results.push_back(std::make_shared<CompilerArtifact>(executors, nullptr));
    _state = State::COMPILED;
    return results;
  }

  // Mode check
  if (options.he_profiling_mode)
    checkProfilerConditions();

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(options.graph_dump_level);
  onert::dumper::dot::DotDumper dot_dumper_part(dump_level);

  // Lower: Assign backend
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>>
    lowered_partialgraphs;
  model->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
    auto part = subg.partialgraphs();
    part->iterate([&](const ir::SubgraphIndex &pindex, ir::Graph &partialgraph) {
      dot_dumper_part.dump(partialgraph,
                           nnfw::misc::str("before_lower_subg_partialgraph-", pindex.value()));

      // // Lower: Assign backend
      lowered_partialgraphs[pindex] =
        std::make_unique<compiler::LoweredGraph>(subg, partialgraph, options);
    });
  });

  for (auto &pair : lowered_partialgraphs)
  {

    const auto &partialgraph_index = pair.first;
    auto &lowered_partialgraph = pair.second;
    dot_dumper_part.dump(*lowered_partialgraph, "after_lower_subg_partialgraph-" +
                                                  std::to_string(partialgraph_index.value()));
  }

  // Partial Graph shape inference
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<StaticShapeInferer>> inferers =
    createStaticShapeInferers(lowered_partialgraphs);
  // NOTE If partialgraph has subgraphs StaticShapeInferer may be called multiple times
  for (auto &pair : lowered_partialgraphs)
  {
    const auto &partialgraph_index = pair.first;
    const auto partial_inferer = inferers.at(partialgraph_index).get();
    partial_inferer->infer();
    partial_inferer->dump();
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
    auto executors = std::make_shared<exec::Executors>();

    const auto &partialgraph_index = ir::SubgraphIndex(pair.first);
    auto &lowered_partialgraph = pair.second;
    auto indexed_ranks = lowered_partialgraph->indexed_ranks();
    ir::OperationDumper dumper("Executor generation of Subgraph " +
                               std::to_string(partialgraph_index.value()));
    lowered_partialgraph->graph().operations().iterate(
      [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });
    auto executor = std::unique_ptr<exec::IExecutor>{
      ExecutorFactory::get().create(std::move(lowered_partialgraph), nullptr, options, executors)};
    executor->setIndexedRanks(indexed_ranks);
    executors->emplace(ir::SubgraphIndex{0}, std::move(executor));

    // It doesn't support tracing in case of partial graph
    results.push_back(std::make_shared<CompilerArtifact>(executors, nullptr));
  }

  _nnpkg.reset();
  /********************************
   * Code generation phase finished
   ********************************/
  _state = State::COMPILED;

  return results;
}

} // namespace compiler

} // namespace onert
