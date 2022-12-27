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
#include "../dumper/dot/DotDumper.h"
#include "../interp/InterpExecutor.h"
#include "../ir/OperationCloner.h"
#include "../ir/OperationDumper.h"
#include "../ir/verifier/Verifier.h"

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
  : _nnpkg{std::make_shared<ir::NNPkg>(model)}, _voptions{&copt}
{
  // DO NOTHING
}

Compiler::Compiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
                   std::vector<std::unique_ptr<CompilerOptions>> &copts)
  : _nnpkg{nnpkg}, _voptions{}
{
  for (uint32_t i = 0; i < copts.size(); i++)
  {
    _voptions.push_back(copts[i].get());
  }
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

std::shared_ptr<CompilerArtifact> Compiler::compile(void)
{
  for (auto options : _voptions)
  {
    options->forceInternalOptions();
    options->verboseOptions();
  }

  // NYI: allow one model compilation
  auto const model_count = _nnpkg->model_count();
  if (model_count != _voptions.size())
    throw std::runtime_error{"Model count and option vector size mismatch"};

  for (uint16_t i = 0; i < model_count; i++)
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
      executors->emplace(ir::ModelIndex{0}, index, std::make_unique<interp::InterpExecutor>(subg));
    });
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

  for (uint16_t i = 0; i < model_count; i++)
  {
    auto const model_index = ir::ModelIndex{i};
    auto model = _nnpkg->model(model_index);

    model->iterate([&](const ir::SubgraphIndex &subg_index, ir::Graph &subg) {
      dot_dumper.dump(subg,
                      nnfw::misc::str("before_lower_model-", i, "-subg-", subg_index.value()));
      // Lower: Assign backend
      lowered_subgs[model_index][subg_index] =
        std::make_unique<compiler::LoweredGraph>(subg, *_voptions[i]);
      // Set tracing_ctx for copied graph
      if (tracing_ctx != nullptr)
        tracing_ctx->setSubgraphIndex(&(lowered_subgs[model_index][subg_index]->graph()),
                                      subg_index.value());
    });
  }

  _nnpkg.reset();

  for (auto &pair : lowered_subgs)
  {
    const auto &model_index = pair.first;
    auto &model_lsubg = pair.second;

    for (auto &pair_inner : model_lsubg)
    {
      const auto &subg_index = pair_inner.first;
      auto &lowered_subg = pair_inner.second;
      dot_dumper.dump(*lowered_subg, nnfw::misc::str("after_lower_model-", model_index.value(),
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
      StaticShapeInferer::createStaticShapeInferers(model_lsubgs);

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

    for (auto &pair_inner : model_lsubgs)
    {
      auto const subg_index = pair_inner.first;
      auto &lowered_subg = pair_inner.second;
      auto const indexed_ranks = lowered_subg->indexed_ranks();

      ir::OperationDumper dumper("Executor generation of Subgraph " +
                                 std::to_string(subg_index.value()));
      lowered_subg->graph().operations().iterate(
        [&](const ir::OperationIndex &, const ir::Operation &op) { op.accept(dumper); });

      auto &options = *_voptions[model_index.value()];
      auto executor = std::unique_ptr<exec::IExecutor>{ExecutorFactory::get().create(
        std::move(lowered_subg), tracing_ctx.get(), options, executors, model_index)};
      executor->setIndexedRanks(indexed_ranks);
      executors->emplace(model_index, subg_index, std::move(executor));
    }
  }

  /********************************
   * Code generation phase finished
   ********************************/
  return std::make_shared<CompilerArtifact>(executors, std::move(tracing_ctx));
}

} // namespace compiler

} // namespace onert
