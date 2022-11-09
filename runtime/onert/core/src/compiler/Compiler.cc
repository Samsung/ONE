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
#include "../exec/SingleModelExecutors.h"
#include "../interp/InterpExecutor.h"
#include "../ir/OperationCloner.h"
#include "../ir/OperationDumper.h"

#include "compiler/StaticShapeInferer.h"

#include <misc/string_helpers.h>
#include <json/json.h>

// TODO Remove using fstream header
#include <fstream>

namespace onert
{

namespace compiler
{

Compiler::Compiler(const std::shared_ptr<ir::Model> &model, CompilerOptions &copt)
  : _model{model}, _options{&copt}
{
  // DO NOTHING
}

Compiler::Compiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
                   std::vector<std::unique_ptr<CompilerOptions>> &copts)
  : _model{nnpkg->primary_model()}, _options{copts[0].get()}
{
  // Use for single model only
  assert(nnpkg->model_count() == 1);
}

void Compiler::enableToFp16() { _options->fp16_enable = true; }

void Compiler::checkProfilerConditions()
{
  if (_options->he_scheduler)
    throw std::runtime_error("Heterogeneous scheduler must be enabled during profiling.");

  if (_options->executor != "Dataflow")
    throw std::runtime_error("Profiling mode works only with 'Dataflow' executor");
}

std::shared_ptr<CompilerArtifact> Compiler::compile(void)
{
  {
    // Set control flow backend for control flow operators
    auto &builtin_id = backend::builtin::Config::ID;
    _options->manual_scheduler_options.opcode_to_backend[ir::OpCode::If] = builtin_id;
    _options->manual_scheduler_options.opcode_to_backend[ir::OpCode::While] = builtin_id;
    _options->manual_scheduler_options.opcode_to_backend[ir::OpCode::Permute] = builtin_id;

    // FIXME This is a workaround for bcq operations, should remove it
    _options->manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQFullyConnected] = "bcq";
    _options->manual_scheduler_options.opcode_to_backend[ir::OpCode::BCQGather] = "bcq";

    // FIXME This is a workaround for bulk operations, should remove it
    _options->manual_scheduler_options.opcode_to_backend[ir::OpCode::Bulk] = "trix";

    _options->verboseOptions();
  }

  {
    _model->iterate([&](const ir::SubgraphIndex &, ir::Graph &subg) {
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
  if (_options->disable_compile)
  {
    auto executors = std::make_shared<exec::SingleModelExecutors>();

    _model->iterate([&](const ir::SubgraphIndex &index, ir::Graph &subg) {
      executors->emplace(ir::ModelIndex{0}, index, std::make_unique<interp::InterpExecutor>(subg));
    });
    return std::make_shared<CompilerArtifact>(executors, nullptr);
  }

  // Mode check
  // TODO handle option for each model
  if (_options->he_profiling_mode)
    checkProfilerConditions();

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  // TODO Handle dump level for each model
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options->graph_dump_level);
  onert::dumper::dot::DotDumper dot_dumper(dump_level);

  // Tracing context
  auto tracing_ctx = std::make_unique<util::TracingCtx>();

  // Lower: Assign backend
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>> lowered_subgs;

  {
    _model->iterate([&](const ir::SubgraphIndex &subg_index, ir::Graph &subg) {
      dot_dumper.dump(subg, nnfw::misc::str("before_lower_subg-", subg_index.value()));
      // Lower: Assign backend
      lowered_subgs[subg_index] = std::make_unique<compiler::LoweredGraph>(subg, *_options);
      // Set tracing_ctx for copied graph
      if (tracing_ctx != nullptr)
        tracing_ctx->setSubgraphIndex(&(lowered_subgs[subg_index]->graph()), subg_index.value());
    });
  }

  _model.reset();

  for (auto &pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    auto &lowered_subg = pair.second;
    dot_dumper.dump(*lowered_subg, nnfw::misc::str("after_lower_subg-", subg_index.value()));
  }

  // Shape inference.
  {
    // Run the StaticShapeInfer of primary subg. All child StaticShapeInferers are called
    // recursively
    std::unordered_map<ir::SubgraphIndex, std::unique_ptr<StaticShapeInferer>> inferers =
      StaticShapeInferer::createStaticShapeInferers(lowered_subgs);

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
    auto &lowered_subg = pair.second;
    compiler::ShapeValidator{lowered_subg->graph()}();
  }

  /*************************************************************
   *  Backend independent analysis & optimization phase finished
   *************************************************************/
  auto executors = std::make_shared<exec::SingleModelExecutors>();

  for (auto &pair : lowered_subgs)
  {
    auto const model_index = ir::ModelIndex{0};
    auto const subg_index = pair.first;
    auto &lowered_subg = pair.second;
    auto const indexed_ranks = lowered_subg->indexed_ranks();

    ir::OperationDumper dumper("Executor generation of Subgraph " +
                               std::to_string(subg_index.value()));
    lowered_subg->graph().operations().iterate(
      [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });

    auto executor = std::unique_ptr<exec::IExecutor>{ExecutorFactory::get().create(
      std::move(lowered_subg), tracing_ctx.get(), *_options, executors, model_index)};
    executor->setIndexedRanks(indexed_ranks);
    executors->emplace(model_index, subg_index, std::move(executor));
  }

  /********************************
   * Code generation phase finished
   ********************************/
  return std::make_shared<CompilerArtifact>(executors, std::move(tracing_ctx));
}

} // namespace compiler
} // namespace onert
