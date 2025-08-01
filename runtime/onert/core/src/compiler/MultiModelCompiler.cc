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

#include "MultiModelCompiler.h"

#include "CompilerHelpers.h"
#include "ExecutorFactory.h"
#include "ShapeValidator.h"
#include "pass/ConstantOutputPass.h"
#include "pass/OddOutputPass.h"
#include "pass/PassRunner.h"
#include "pass/PermutationIOPass.h"
#include "pass/UnusedOperandEliminationPass.h"
#include "../dumper/dot/DotDumper.h"
#include "../exec/MultiModelExecutors.h"
#include "../ir/OperationDumper.h"
#include "../ir/verifier/Verifier.h"

#include "compiler/StaticShapeInferer.h"

#include <misc/string_helpers.h>
#include <misc/polymorphic_downcast.h>

namespace onert::compiler
{

MultiModelCompiler::MultiModelCompiler(const std::shared_ptr<ir::NNPkg> &nnpkg,
                                       CompilerOptions *copts)
  : _nnpkg{nnpkg}, _options{copts}
{
  // DO NOTHING
}

CompilerOptions MultiModelCompiler::optionForSingleModel(const ir::ModelIndex &model_index)
{
  CompilerOptions model_opts = CompilerOptions(*_options); // Copy options
  model_opts.input_layout.clear();
  model_opts.output_layout.clear();
  model_opts.input_type.clear();
  model_opts.output_type.clear();

  // Set option for selected model
  auto option_for_model = [](const auto &src_opt, auto &dst_opt, const ir::ModelIndex &model_index,
                             auto io_desc_getter) {
    for (const auto &[index, val] : src_opt)
    {
      const auto &io_desc = io_desc_getter(index.value());
      if (std::get<ir::ModelIndex>(io_desc) == model_index)
        dst_opt.insert_or_assign(std::get<ir::IOIndex>(io_desc), val);
    }
  };

  option_for_model(_options->input_layout, model_opts.input_layout, model_index,
                   [this](uint32_t idx) { return _nnpkg->input(idx); });
  option_for_model(_options->output_layout, model_opts.output_layout, model_index,
                   [this](uint32_t idx) { return _nnpkg->output(idx); });
  option_for_model(_options->input_type, model_opts.input_type, model_index,
                   [this](uint32_t idx) { return _nnpkg->input(idx); });
  option_for_model(_options->output_type, model_opts.output_type, model_index,
                   [this](uint32_t idx) { return _nnpkg->output(idx); });

  return model_opts;
}

std::shared_ptr<CompilerArtifact> MultiModelCompiler::compile(void)
{
  /***************************************************
   * Prepare compilation phase
   ***************************************************/
  {
    if (!_options)
      throw std::runtime_error{"Empty compile option"};

    // Mode check
    // TODO handle option for each model
    if (_options->he_profiling_mode)
      throw std::runtime_error("NYI: Profiling mode for multiple model is not supported yet");

    _options->forceInternalOptions();
    _options->verboseOptions();
  }

  // NYI: allow one model compilation
  auto const model_count = _nnpkg->model_count();
  for (uint16_t i = 0; i < model_count; i++)
  {
    if (!_nnpkg->model(ir::ModelIndex{i})->hasOnly<ir::Graph>())
      throw std::runtime_error("MultiModelCompiler can only compile models for inference.");
  }

  std::unordered_map<ir::ModelIndex, CompilerOptions> model_options;
  for (uint16_t i = 0; i < model_count; i++)
  {
    auto model_index = ir::ModelIndex{i};
    model_options[model_index] = optionForSingleModel(model_index);
    _nnpkg->model(ir::ModelIndex{i})->iterate([&](const ir::SubgraphIndex &, ir::IGraph &graph) {
      auto &subg = nnfw::misc::polymorphic_downcast<ir::Graph &>(graph);

      // Mandatory passes
      pass::PassRunner{}
        .append(std::make_unique<pass::ConstantOutputPass>(subg))
        .append(std::make_unique<pass::OddOutputPass>(subg))
        .append(std::make_unique<pass::PermutationIOPass>(subg, model_options[model_index]))
        .run();

      // Optimizations
      pass::PassRunner{}.append(std::make_unique<pass::UnusedOperandEliminationPass>(subg)).run();
    });
  }

  /***************************************************
   * Backend independent analysis & optimization phase
   ***************************************************/
  // TODO Handle dump level for each model
  auto dump_level = static_cast<dumper::dot::DotDumper::Level>(_options->graph_dump_level);
  onert::dumper::dot::DotDumper dot_dumper(dump_level);

  // Tracing context
  // TODO Support tracing_ctx for multiple model
  std::unique_ptr<util::TracingCtx> tracing_ctx = nullptr;

  // Model edge context: copy model edge context
  auto model_edges = std::make_unique<ir::ModelEdges>(_nnpkg->model_edges());

  // Custom kernels
  std::unordered_map<ir::ModelIndex, std::shared_ptr<backend::custom::IKernelBuilder>>
    custom_kernel_builders;
  for (uint16_t i = 0; i < model_count; i++)
  {
    auto const model_index = ir::ModelIndex{i};
    custom_kernel_builders[model_index] = _nnpkg->model(model_index)->getKernelBuilder();
  }

  // Lower: Assign backend
  std::unordered_map<ir::ModelIndex,
                     std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>>>
    lowered_subgs;

  for (uint16_t i = 0; i < model_count; i++)
  {
    auto const model_index = ir::ModelIndex{i};
    auto model = _nnpkg->model(model_index);

    model->iterate([&](const ir::SubgraphIndex &subg_index, ir::IGraph &graph) {
      auto &subg = nnfw::misc::polymorphic_downcast<ir::Graph &>(graph);

      dot_dumper.dump(subg,
                      nnfw::misc::str("before_lower_model-", i, "-subg-", subg_index.value()));
      // Lower: Assign backend
      lowered_subgs[model_index][subg_index] =
        std::make_unique<compiler::LoweredGraph>(subg, model_options[model_index]);
      // Set tracing_ctx for copied graph
      if (tracing_ctx != nullptr)
        tracing_ctx->setSubgraphIndex(&(lowered_subgs[model_index][subg_index]->graph()),
                                      subg_index.value());
    });
  }

  _nnpkg.reset();

  for (const auto &[model_index, model_lsubg] : lowered_subgs)
  {
    for (const auto &[subg_index, lowered_subg] : model_lsubg)
    {
      dot_dumper.dump(*lowered_subg, nnfw::misc::str("after_lower_model-", model_index.value(),
                                                     "-subg-", subg_index.value()));
    }
  }

  // Shape inference.
  for (auto &&pair : lowered_subgs)
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
  for (const auto &pair : lowered_subgs)
  {
    const auto &model_lsubgs = pair.second;

    for (const auto &pair_inner : model_lsubgs)
    {
      const auto &lowered_subg = pair_inner.second;
      compiler::ShapeValidator{lowered_subg->graph()}();
    }
  }

  /*************************************************************
   *  Backend independent analysis & optimization phase finished
   *************************************************************/
  auto executors = std::make_shared<exec::MultiModelExecutors>(std::move(model_edges));
  for (auto &&pair : lowered_subgs)
  {
    auto const &model_index = pair.first;
    auto &model_lsubgs = pair.second;

    for (auto &&pair_inner : model_lsubgs)
    {
      auto const subg_index = pair_inner.first;
      auto &lowered_subg = pair_inner.second;
      auto const indexed_ranks = lowered_subg->indexed_ranks();

      ir::OperationDumper dumper("Executor generation of Subgraph " +
                                 std::to_string(subg_index.value()));
      lowered_subg->graph().operations().iterate(
        [&](const ir::OperationIndex &, const ir::IOperation &op) { op.accept(dumper); });

      ExecutorFactoryArgs args;
      args.tracing_ctx = tracing_ctx.get();
      args.options = &model_options[model_index];
      args.model_index = model_index;
      args.custom_kernel_builder = custom_kernel_builders[model_index];
      if (_options->internal_output_alloc)
      {
        const auto &pkg_outputs = _nnpkg->model_edges().pkg_outputs;
        for (const auto &desc : pkg_outputs)
        {
          // Only outputs of this entry
          if (const auto &[m, s, io] = desc; m == model_index && s == subg_index)
          {
            // Map IOIndex to OperandIndex
            auto idx = lowered_subg->graph().getOutputs().at(io);
            args.internal_io_indexes.add(idx);
          }
        }
      }
      auto executor = std::unique_ptr<exec::IExecutor>{
        ExecutorFactory::get().create(std::move(lowered_subg), executors, args)};
      executor->setIndexedRanks(indexed_ranks);
      executors->emplace(model_index, subg_index, std::move(executor));
    }
  }

  /********************************
   * Code generation phase finished
   ********************************/
  return std::make_shared<CompilerArtifact>(executors, std::move(tracing_ctx));
}

} // namespace onert::compiler
