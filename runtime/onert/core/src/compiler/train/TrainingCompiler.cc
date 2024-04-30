/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TrainingCompiler.h"

#include "StaticBackwardShapeInferer.h"
#include "TrainableOperationConverter.h"
#include "pass/LossInsertionPass.h"
#include "../CompilerHelpers.h"
#include "../ExecutorFactory.h"
#include "../pass/ConstantOutputPass.h"
#include "../pass/OddOutputPass.h"
#include "../pass/PassRunner.h"
#include "../pass/UnusedOperandEliminationPass.h"
#include "../ShapeValidator.h"
#include "../../dumper/dot/DotDumper.h"
#include "../../exec/train/TrainableExecutors.h"
#include "../../ir/OperationDumper.h"
#include "../../ir/verifier/Verifier.h"

#include <compiler/StaticShapeInferer.h>
#include <compiler/train/LoweredTrainableGraph.h>
#include <ir/train/TrainableGraph.h>

#include <misc/polymorphic_downcast.h>
#include <misc/string_helpers.h>

namespace onert
{
namespace compiler
{
namespace train
{

TrainingCompiler::TrainingCompiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
                                   std::vector<std::unique_ptr<CompilerOptions>> &copts,
                                   const ir::train::TrainingInfo &training_info)
  : _model{nnpkg->primary_model()}, _options{copts[0].get()}, _training_info{training_info}
{
  if (nnpkg->model_count() > 1)
    throw std::runtime_error("TrainingCompiler does not support multiple models yet");

  if (nnpkg->primary_model()->subgraphs_count() > 1)
    throw std::runtime_error("TrainingCompiler does not support multiple subgraphs yet");
}

std::shared_ptr<CompilerArtifact> TrainingCompiler::compile(void)
{
  /***************************************************
   * Prepare compilation phase
   ***************************************************/
  if (!_options)
    throw std::runtime_error{"Empty compile option"};

  // Mode check
  // TODO handle option for each model
  if (_options->he_profiling_mode)
  {
    if (!_options->he_scheduler)
      throw std::runtime_error("Heterogeneous scheduler must be enabled during profiling.");

    if (_options->executor != "Dataflow")
      throw std::runtime_error("Profiling mode works only with 'Dataflow' executor");
  }

  if (!_options->minmax_filepath.empty())
  {
    if (_options->executor != "Linear")
      throw std::runtime_error("Recording minmax works only with Linear executor");
  }

  _options->forceInternalOptions();
  _options->verboseOptions();

  auto custom_kernel_builder = _model->getKernelBuilder();

  _model->iterate([&](const ir::SubgraphIndex &, ir::IGraph &graph) {
    auto &subg = nnfw::misc::polymorphic_downcast<ir::Graph &>(graph);
    // Mandatory passes
    compiler::pass::PassRunner{}
      .append(std::make_unique<compiler::pass::ConstantOutputPass>(subg))
      .append(std::make_unique<compiler::pass::OddOutputPass>(subg))
      .run();

    // Optimizations
    compiler::pass::PassRunner{}
      .append(std::make_unique<compiler::pass::UnusedOperandEliminationPass>(subg))
      .run();
  });

  std::unordered_map<ir::SubgraphIndex, std::shared_ptr<ir::train::TrainableGraph>>
    trainable_subgraphs;

  if (_model->hasOnly<ir::Graph>())
  {
    // Create trainable subgraphs by copy and converting inference model
    _model->iterate([&](const ir::SubgraphIndex &subg_index, const ir::IGraph &graph) {
      const auto &subg = nnfw::misc::polymorphic_downcast<const ir::Graph &>(graph);
      // Create TrainableGraph by copying Graph
      auto trainable_subg = std::make_shared<ir::train::TrainableGraph>(subg);

      // Convert operations to trainable operations
      auto converter = TrainableOperationConverter{*trainable_subg, &_training_info};
      subg.operations().iterate(
        [&](const onert::ir::OperationIndex &op_index, const onert::ir::IOperation &op) {
          auto trainable_op = converter(op);
          if (_options->frozen_train_ops.find(op_index) != std::end(_options->frozen_train_ops))
          {
            trainable_op->disableWeightsUpdate();
          }
          auto gen_index = trainable_subg->replaceOperation(op_index, std::move(trainable_op));
          UNUSED_RELEASE(gen_index);
          assert(gen_index == op_index);
        });

      trainable_subgraphs[subg_index] = std::move(trainable_subg);
    });
  }
  else
  {
    // TODO Support models that have TrainableGraphs
    throw std::runtime_error("TrainingCompiler: Invalid model");
  }

  // operation
  _model.reset();

  // TODO Handle dump level for each model
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options->graph_dump_level);
  onert::dumper::dot::DotDumper dot_dumper(dump_level);

  for (const auto &pair : trainable_subgraphs)
  {
    const auto &subg_index = pair.first;
    const auto &subg = pair.second;
    dot_dumper.dump(*subg, nnfw::misc::str("before_loss_insertion-", subg_index.value()));
  }

  // Apply pass for trainable subgraphs
  for (auto &&pair : trainable_subgraphs)
  {
    auto trainable_subg = pair.second;
    auto subg_index = pair.first;

    compiler::pass::PassRunner{}
      .append(std::make_unique<train::pass::LossInsertionPass>(*trainable_subg, &_training_info,
                                                               subg_index))
      .run();
  }

  for (const auto &pair : trainable_subgraphs)
  {
    const auto &subg_index = pair.first;
    const auto &subg = pair.second;
    dot_dumper.dump(*subg, nnfw::misc::str("after_loss_insertion-", subg_index.value()));
  }

  // Change input shape according to batch_size
  for (auto &&pair : trainable_subgraphs)
  {
    auto trainable_subg = pair.second;

    for (const auto &ind : trainable_subg->getInputs())
    {
      auto &input = trainable_subg->operands().at(ind);
      auto new_shape = input.info().shape();
      // TODO Consider batch size index
      if (new_shape.dim(0) != 1)
        throw std::runtime_error("the first dim is not 1. It is not supported yet.");
      new_shape.dim(0) = _training_info.batchSize();
      input.info().shape(new_shape);
    }
  }

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  // Tracing context
  auto tracing_ctx = std::make_unique<util::TracingCtx>();

  // Lower: Assign backend
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::train::LoweredTrainableGraph>>
    lowered_subgs;
  {
    for (auto &&pair : trainable_subgraphs)
    {
      auto &subg_index = pair.first;
      auto trainable_subg = pair.second;

      // Lower: Assign backend
      lowered_subgs[subg_index] =
        std::make_unique<compiler::train::LoweredTrainableGraph>(*trainable_subg, *_options);
      // Set tracing_ctx for copied graph
      if (tracing_ctx != nullptr)
        tracing_ctx->setSubgraphIndex(&(lowered_subgs[subg_index]->graph()), subg_index.value());
    }
  }

  for (const auto &pair : lowered_subgs)
  {
    const auto &subg_index = pair.first;
    const auto &lowered_subg = pair.second;
    dot_dumper.dump(*lowered_subg, nnfw::misc::str("after_lower_subg-", subg_index.value()));
  }

  // Set operands' info for back propagation as default tensor info
  for (const auto &pair : lowered_subgs)
  {
    auto lowered_subg = pair.second.get();
    auto &tgraph = lowered_subg->trainable_graph();
    tgraph.operands().iterate([&](const ir::OperandIndex &index, const ir::Operand &obj) {
      if (!obj.isConstant())
      {
        auto bwd_operand = std::make_unique<ir::Operand>(obj);
        const auto gen_index = tgraph.addBackwardOperand(index, std::move(bwd_operand));
        assert(gen_index == index);
        UNUSED_RELEASE(gen_index);
      }
    });
  }

  // Shape inference.
  {
    // Run the StaticShapeInfer of primary subg. All child StaticShapeInferers are called
    // recursively
    std::unordered_map<ir::SubgraphIndex, std::unique_ptr<StaticShapeInferer>> inferers =
      createStaticShapeInferers(lowered_subgs);

    const auto primary_subg_idx = ir::SubgraphIndex{0};
    inferers.at(primary_subg_idx)->infer();

    for (const auto &pair_inferer : inferers)
    {
      const auto inferer = pair_inferer.second.get();
      inferer->dump();
    }

    // NOTE StaticBackwardShapeInferer is allocated for each subgraph,
    //      so it does not support models that have controlflow operations yet.
    for (auto &&pair : lowered_subgs)
    {
      auto &lowered_subg = pair.second;
      auto inferer = std::make_unique<StaticBackwardShapeInferer>(lowered_subg.get());
      inferer->infer();
      inferer->dump();
    }
  }

  // Shape validation
  for (const auto &pair : lowered_subgs)
  {
    auto &lowered_subg = pair.second;
    compiler::ShapeValidator{lowered_subg->graph()}();
  }

  // TODO Validate shapes of the tensors for back propagation

  /*************************************************************
   *  Backend independent analysis & optimization phase finished
   *************************************************************/
  auto executors = std::make_shared<exec::train::TrainableExecutors>();
  for (auto &&pair : lowered_subgs)
  {
    auto const model_index = ir::ModelIndex{0};
    auto const subg_index = pair.first;
    auto &lowered_subg = pair.second;
    auto const indexed_ranks = lowered_subg->indexed_ranks();

    ir::OperationDumper dumper("Executor generation of Subgraph " +
                               std::to_string(subg_index.value()));
    lowered_subg->graph().operations().iterate(
      [&](const ir::OperationIndex &, const ir::IOperation &op) { op.accept(dumper); });

    ExecutorFactoryArgs args;
    args.tracing_ctx = tracing_ctx.get();
    args.options = _options;
    args.model_index = model_index;
    args.custom_kernel_builder = custom_kernel_builder;
    auto executor = std::unique_ptr<exec::IExecutor>{
      ExecutorFactory::get().create(std::move(lowered_subg), executors, args, _training_info)};
    executor->setIndexedRanks(indexed_ranks);
    executors->emplace(model_index, subg_index, std::move(executor));
  }

  /********************************
   * Code generation phase finished
   ********************************/
  return std::make_shared<CompilerArtifact>(executors, std::move(tracing_ctx));
}

} // namespace train
} // namespace compiler
} // namespace onert
