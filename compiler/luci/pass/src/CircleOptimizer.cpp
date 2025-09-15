/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci/CircleOptimizer.h"

#include "luci/Pass/CanonicalizePass.h"
#include "luci/Pass/ConvertNCHWToNHWCPass.h"
#include "luci/Pass/CommonSubExpressionEliminationPass.h"
#include "luci/Pass/ExpandBroadcastConstPass.h"
#include "luci/Pass/FoldAddV2Pass.h"
#include "luci/Pass/FoldCastPass.h"
#include "luci/Pass/FoldDensifyPass.h"
#include "luci/Pass/FoldDepthwiseConv2DPass.h"
#include "luci/Pass/FoldDequantizePass.h"
#include "luci/Pass/FoldFullyConnectedPass.h"
#include "luci/Pass/FoldGatherPass.h"
#include "luci/Pass/FoldMulPass.h"
#include "luci/Pass/FoldReshapePass.h"
#include "luci/Pass/FoldShapePass.h"
#include "luci/Pass/FoldSparseToDensePass.h"
#include "luci/Pass/FoldSqueezePass.h"
#include "luci/Pass/ForwardReshapeToUnaryOpPass.h"
#include "luci/Pass/ForwardTransposeOpPass.h"
#include "luci/Pass/FuseActivationFunctionPass.h"
#include "luci/Pass/FuseAddToFullyConnectedBiasPass.h"
#include "luci/Pass/FuseAddWithConvPass.h"
#include "luci/Pass/FuseAddWithFullyConnectedPass.h"
#include "luci/Pass/FuseAddWithTConvPass.h"
#include "luci/Pass/FuseBatchNormWithConvPass.h"
#include "luci/Pass/FuseBatchNormWithDwConvPass.h"
#include "luci/Pass/FuseBatchNormWithTConvPass.h"
#include "luci/Pass/FuseBCQPass.h"
#include "luci/Pass/FuseMulToFullyConnectedWeightsPass.h"
#include "luci/Pass/FuseInstanceNormPass.h"
#include "luci/Pass/FuseMeanWithMeanPass.h"
#include "luci/Pass/FuseMulWithConvPass.h"
#include "luci/Pass/FuseMulWithDivPass.h"
#include "luci/Pass/FuseMulWithFullyConnectedPass.h"
#include "luci/Pass/FuseMulWithRmsNormPass.h"
#include "luci/Pass/FusePreActivationBatchNormPass.h"
#include "luci/Pass/FusePReluPass.h"
#include "luci/Pass/FuseGeluPass.h"
#include "luci/Pass/FuseRsqrtPass.h"
#include "luci/Pass/FuseSliceWithTConvPass.h"
#include "luci/Pass/FuseHorizontalFullyConnectedPass.h"
#include "luci/Pass/FuseTransposeWithMeanPass.h"
#include "luci/Pass/FuseRmsNormPass.h"
#include "luci/Pass/FuseRoPEPass.h"
#include "luci/Pass/MakeBatchNormGammaPositivePass.h"
#include "luci/Pass/RemoveDuplicateConstPass.h"
#include "luci/Pass/RemoveFakeQuantPass.h"
#include "luci/Pass/RemoveGatherGuardPass.h"
#include "luci/Pass/RemoveQDQForMixedPrecisionOpPass.h"
#include "luci/Pass/RemoveQuantDequantSeqPass.h"
#include "luci/Pass/RemoveRedundantReshapePass.h"
#include "luci/Pass/RemoveRedundantTransposePass.h"
#include "luci/Pass/RemoveRedundantQuantizePass.h"
#include "luci/Pass/RemoveUnnecessaryAddPass.h"
#include "luci/Pass/RemoveUnnecessaryCastPass.h"
#include "luci/Pass/RemoveUnnecessaryMulDivPass.h"
#include "luci/Pass/RemoveUnnecessaryReshapePass.h"
#include "luci/Pass/RemoveUnnecessaryReshapeNetPass.h"
#include "luci/Pass/RemoveUnnecessarySlicePass.h"
#include "luci/Pass/RemoveUnnecessaryStridedSlicePass.h"
#include "luci/Pass/RemoveUnnecessarySplitPass.h"
#include "luci/Pass/RemoveUnnecessaryTransposeNetPass.h"
#include "luci/Pass/ReplaceNonConstFCWithBatchMatMulPass.h"
#include "luci/Pass/ReplaceMulAddWithDepthwiseConvPass.h"
#include "luci/Pass/ReplaceSubWithAddPass.h"
#include "luci/Pass/ReplaceWithFCGeluFCPass.h"
#include "luci/Pass/ResolveCustomOpAddPass.h"
#include "luci/Pass/ResolveCustomOpBatchMatMulPass.h"
#include "luci/Pass/ResolveCustomOpMatMulPass.h"
#include "luci/Pass/ResolveCustomOpMaxPoolWithArgmaxPass.h"
#include "luci/Pass/ResolveCustomOpSplitVPass.h"
#include "luci/Pass/ResolveFormerCustomOpPass.h"
#include "luci/Pass/SparsifyTensorPass.h"
#include "luci/Pass/ShuffleWeightTo16x1Float32Pass.h"
#include "luci/Pass/SubstituteExpandDimsToReshapePass.h"
#include "luci/Pass/SubstitutePackToReshapePass.h"
#include "luci/Pass/SubstitutePadV2ToPadPass.h"
#include "luci/Pass/SubstituteSplitVToSplitPass.h"
#include "luci/Pass/SubstituteSqueezeToReshapePass.h"
#include "luci/Pass/SubstituteStridedSliceToReshapePass.h"
#include "luci/Pass/SubstituteTransposeToReshapePass.h"
#include "luci/Pass/TransformMinMaxToRelu6Pass.h"
#include "luci/Pass/TransformMinReluToRelu6Pass.h"
#include "luci/Pass/TransformSqrtDivToRsqrtMulPass.h"
#include "luci/Pass/DecomposeHardSwishPass.h"
#include "luci/Pass/DecomposeSoftmaxPass.h"
#include "luci/Pass/UnrollUnidirectionalSequenceLSTMPass.h"
#include "luci/Pass/XpSepActFromTransposeConvPass.h"
// TODO add more passes

#include "luci/Pass/CircleShapeInferencePass.h"
#include "luci/Pass/CircleTypeInferencePass.h"

// logo passes
#include <logo/RemoveDeadNodeWithQueryPass.h>

#include "ModulePhase.h"
#include "ProgressReporter.h"

#include <luci/IR/CircleNodes.h>
#include <logo/Phase.h>
#include <pepper/csv2vec.h>

#include <memory>
#include <sstream>

namespace
{

using namespace luci;

class OptimizeOptionsImpl final : public luci::CircleOptimizer::Options
{
public:
  void enable(Algorithm) final;
  void param(AlgorithmParameters, const std::string &) final;
  const std::string param(AlgorithmParameters) const final;
  bool query(Algorithm) final;

private:
  std::vector<Algorithm> _algorithms;
  std::map<AlgorithmParameters, const std::string> _algorithm_params;
};

void OptimizeOptionsImpl::enable(Algorithm algo) { _algorithms.push_back(algo); }

void OptimizeOptionsImpl::param(AlgorithmParameters param, const std::string &str)
{
  _algorithm_params.insert(std::pair<AlgorithmParameters, const std::string>(param, str));
}

const std::string OptimizeOptionsImpl::param(AlgorithmParameters param) const
{
  auto param_str = _algorithm_params.find(param);
  if (param_str != _algorithm_params.end())
  {
    return param_str->second;
  }
  else
  {
    return std::string();
  }
}

bool OptimizeOptionsImpl::query(Algorithm algo)
{
  std::vector<Algorithm>::iterator it = std::find(_algorithms.begin(), _algorithms.end(), algo);
  if (it == _algorithms.end())
    return false;

  return true;
}

// TODO Make a struct for args
void convert_nchw_to_nhwc(loco::Graph *g, bool preserve_input, bool preserve_output, bool fuse_fc,
                          bool fuse_gelu)
{
  logo::Phase phase;

  phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  // Resolve custom Ops
  phase.emplace_back(std::make_unique<luci::ResolveCustomOpAddPass>());
  phase.emplace_back(std::make_unique<luci::ResolveCustomOpBatchMatMulPass>());
  phase.emplace_back(std::make_unique<luci::ResolveCustomOpMatMulPass>());
  phase.emplace_back(std::make_unique<luci::ResolveCustomOpMaxPoolWithArgmaxPass>());
  phase.emplace_back(std::make_unique<luci::ResolveCustomOpSplitVPass>());
  phase.emplace_back(std::make_unique<luci::ResolveFormerCustomOpPass>());

  // Fuse FullyConnected with Add
  // Why we perform FuseAddWithFullyConnectedPass before ConvertNCHWToNHWCPass?
  // FullyConnected Op's layout is not changed in ConvertNCHWToNHWCPass, while
  // Add Op's layer is changed from NCHW to NHWC.
  // This disables fusion of Add and FullyConnected after ConvertNCHWToNHWC.
  if (fuse_fc)
    phase.emplace_back(std::make_unique<luci::FuseAddWithFullyConnectedPass>());

  // Fuse decomposed ops to Gelu Op
  // Why here? ConverNCHWToNHWCPass inserts additional Ops, so it is better to fuse
  // Gelu in advance.
  if (fuse_gelu)
    phase.emplace_back(std::make_unique<luci::FuseGeluPass>());

  phase.emplace_back(
    std::make_unique<luci::ConvertNCHWToNHWCPass>(preserve_input, preserve_output));

  ProgressReporter prog(g, logo::PhaseStrategy::Restart);
  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

} // namespace

namespace luci
{

CircleOptimizer::Options *CircleOptimizer::options(void)
{
  if (_options == nullptr)
  {
    _options = std::make_unique<OptimizeOptionsImpl>();
  }

  return _options.get();
}

void CircleOptimizer::canonicalize(loco::Graph *g) const
{
  logo::Phase phase;

  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CanonicalizePass>());
  phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());

  ProgressReporter prog(g, logo::PhaseStrategy::Restart);
  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

void CircleOptimizer::optimize(luci::Module *m) const
{
  luci::Phase phase;

  // Following passes are needed everytime when other passes create new node or modify some nodes.
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  if (_options->query(Options::Algorithm::FuseBCQ))
  {
    phase.emplace_back(std::make_unique<FuseBCQPass>());
  }

  ModuleProgressReporter prog(m, logo::PhaseStrategy::Restart);
  PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{m};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

template <typename T> std::unique_ptr<logo::Pass> createPassInstance(void)
{
  return std::make_unique<T>();
}

void CircleOptimizer::optimize(loco::Graph *g) const
{
  canonicalize(g);

  logo::Phase phase;

  // Conversion from NCHW to NHWC is done first to avoid interference with other optimizations.
  if (_options->query(Options::Algorithm::ConvertNCHWToNHWC))
  {
    bool preserve_input =
      _options->param(Options::AlgorithmParameters::NCHW_to_NHWC_input_shape) != "true";
    bool preserve_output =
      _options->param(Options::AlgorithmParameters::NCHW_to_NHWC_output_shape) != "true";

    bool fuse_fc = _options->query(Options::Algorithm::FuseAddWithFullyConnected);
    bool fuse_gelu = _options->query(Options::Algorithm::FuseGelu);

    convert_nchw_to_nhwc(g, preserve_input, preserve_output, fuse_fc, fuse_gelu);
  }

  /* TRANSFORM DECLARATION BEGIN */
  phase.emplace_back(std::make_unique<logo::RemoveDeadNodeWithQueryPass>());

  // Following passes are needed everytime when other passes create new node or modify some nodes.
  phase.emplace_back(std::make_unique<luci::CircleShapeInferencePass>());
  phase.emplace_back(std::make_unique<luci::CircleTypeInferencePass>());

  // Forward Reshape/Transpose is done after
  // 1. SubstituteXXXToReshape
  // 2. RemoveRedundantReshape/Transpose
  // See https://github.com/Samsung/ONE/pull/10596 for more details
  if (_options->query(Options::Algorithm::SubstituteExpandDimsToReshape))
  {
    phase.emplace_back(std::make_unique<luci::SubstituteExpandDimsToReshapePass>());
  }
  if (_options->query(Options::Algorithm::SubstitutePackToReshape))
  {
    phase.emplace_back(std::make_unique<luci::SubstitutePackToReshapePass>());
  }
  if (_options->query(Options::Algorithm::SubstituteSqueezeToReshape))
  {
    phase.emplace_back(std::make_unique<luci::SubstituteSqueezeToReshapePass>());
  }
  if (_options->query(Options::Algorithm::SubstituteStridedSliceToReshape))
  {
    phase.emplace_back(std::make_unique<luci::SubstituteStridedSliceToReshapePass>());
  }
  if (_options->query(Options::Algorithm::SubstituteTransposeToReshape))
  {
    phase.emplace_back(std::make_unique<luci::SubstituteTransposeToReshapePass>());
  }
  if (_options->query(Options::Algorithm::RemoveRedundantReshape))
  {
    phase.emplace_back(std::make_unique<luci::RemoveRedundantReshapePass>());
  }
  if (_options->query(Options::Algorithm::RemoveRedundantTranspose))
  {
    phase.emplace_back(std::make_unique<luci::RemoveRedundantTransposePass>());
  }

  // clang-format off
  std::map<Options::Algorithm, std::unique_ptr<logo::Pass> (*)(void)> option_to_pass;

  option_to_pass[Options::Algorithm::CommonSubExpressionElimination] = &createPassInstance<luci::CommonSubExpressionEliminationPass>;
  option_to_pass[Options::Algorithm::ResolveCustomOpAdd] = &createPassInstance<luci::ResolveCustomOpAddPass>;
  option_to_pass[Options::Algorithm::ResolveCustomOpBatchMatMul] = &createPassInstance<luci::ResolveCustomOpBatchMatMulPass>;
  option_to_pass[Options::Algorithm::ResolveCustomOpMatMul] = &createPassInstance<luci::ResolveCustomOpMatMulPass>;
  option_to_pass[Options::Algorithm::ResolveFormerCustomOp] = &createPassInstance<luci::ResolveFormerCustomOpPass>;
  option_to_pass[Options::Algorithm::FuseMeanWithMean] = &createPassInstance<luci::FuseMeanWithMeanPass>;
  option_to_pass[Options::Algorithm::FuseMulWithConv] = &createPassInstance<luci::FuseMulWithConvPass>;
  option_to_pass[Options::Algorithm::FuseMulWithDiv] = &createPassInstance<luci::FuseMulWithDivPass>;
  option_to_pass[Options::Algorithm::FuseMulWithFullyConnected] = &createPassInstance<luci::FuseMulWithFullyConnectedPass>;
  option_to_pass[Options::Algorithm::FuseMulWithRmsNorm] = &createPassInstance<luci::FuseMulWithRmsNormPass>;
  option_to_pass[Options::Algorithm::ResolveCustomOpMaxPoolWithArgmax] = &createPassInstance<luci::ResolveCustomOpMaxPoolWithArgmaxPass>;
  option_to_pass[Options::Algorithm::ResolveCustomOpSplitV] = &createPassInstance<luci::ResolveCustomOpSplitVPass>;
  option_to_pass[Options::Algorithm::FuseInstanceNorm] = &createPassInstance<luci::FuseInstanceNormPass>;
  option_to_pass[Options::Algorithm::FuseBatchNormWithConv] = &createPassInstance<luci::FuseBatchNormWithConvPass>;
  option_to_pass[Options::Algorithm::FuseBatchNormWithDwConv] = &createPassInstance<luci::FuseBatchNormWithDwConvPass>;
  option_to_pass[Options::Algorithm::FuseBatchNormWithTConv] = &createPassInstance<luci::FuseBatchNormWithTConvPass>;
  option_to_pass[Options::Algorithm::FuseSliceWithTConv] = &createPassInstance<luci::FuseSliceWithTConvPass>;
  option_to_pass[Options::Algorithm::FuseAddToFullyConnectedBias] = &createPassInstance<luci::FuseAddToFullyConnectedBiasPass>;
  option_to_pass[Options::Algorithm::FuseAddWithConv] = &createPassInstance<luci::FuseAddWithConvPass>;
  option_to_pass[Options::Algorithm::FuseAddWithFullyConnected] = &createPassInstance<luci::FuseAddWithFullyConnectedPass>;
  option_to_pass[Options::Algorithm::FuseAddWithTConv] = &createPassInstance<luci::FuseAddWithTConvPass>;
  option_to_pass[Options::Algorithm::FuseActivationFunction] = &createPassInstance<luci::FuseActivationFunctionPass>;
  option_to_pass[Options::Algorithm::FuseMulToFullyConnectedWeights] = &createPassInstance<luci::FuseMulToFullyConnectedWeightsPass>;
  option_to_pass[Options::Algorithm::FusePRelu] = &createPassInstance<luci::FusePReluPass>;
  option_to_pass[Options::Algorithm::FuseGelu] = &createPassInstance<luci::FuseGeluPass>;
  option_to_pass[Options::Algorithm::FuseRsqrt] = &createPassInstance<luci::FuseRsqrtPass>;
  option_to_pass[Options::Algorithm::FuseHorizontalFullyConnected] = &createPassInstance<luci::FuseHorizontalFullyConnectedPass>;
  option_to_pass[Options::Algorithm::FuseTransposeWithMean] = &createPassInstance<luci::FuseTransposeWithMeanPass>;
  option_to_pass[Options::Algorithm::FuseRmsNorm] = &createPassInstance<luci::FuseRmsNormPass>;
  option_to_pass[Options::Algorithm::FuseRoPE] = &createPassInstance<luci::FuseRoPEPass>;
  option_to_pass[Options::Algorithm::FoldAddV2] = &createPassInstance<luci::FoldAddV2Pass>;
  option_to_pass[Options::Algorithm::FoldCast] = &createPassInstance<luci::FoldCastPass>;
  option_to_pass[Options::Algorithm::FoldDensify] = &createPassInstance<luci::FoldDensifyPass>;
  option_to_pass[Options::Algorithm::FoldDepthwiseConv2D] = &createPassInstance<luci::FoldDepthwiseConv2DPass>;
  option_to_pass[Options::Algorithm::FoldDequantize] = &createPassInstance<luci::FoldDequantizePass>;
  option_to_pass[Options::Algorithm::FoldFullyConnected] = &createPassInstance<luci::FoldFullyConnectedPass>;
  option_to_pass[Options::Algorithm::FoldGather] = &createPassInstance<luci::FoldGatherPass>;
  option_to_pass[Options::Algorithm::FoldMul] = &createPassInstance<luci::FoldMulPass>;
  option_to_pass[Options::Algorithm::FoldReshape] = &createPassInstance<luci::FoldReshapePass>;
  option_to_pass[Options::Algorithm::FoldShape] = &createPassInstance<luci::FoldShapePass>;
  option_to_pass[Options::Algorithm::FoldSparseToDense] = &createPassInstance<luci::FoldSparseToDensePass>;
  option_to_pass[Options::Algorithm::FoldSqueeze] = &createPassInstance<luci::FoldSqueezePass>;
  option_to_pass[Options::Algorithm::FusePreActivationBatchNorm] = &createPassInstance<luci::FusePreActivationBatchNormPass>;
  option_to_pass[Options::Algorithm::MakeBatchNormGammaPositive] = &createPassInstance<luci::MakeBatchNormGammaPositivePass>;
  option_to_pass[Options::Algorithm::ShuffleWeightTo16x1Float32] = &createPassInstance<luci::ShuffleWeightTo16x1Float32Pass>;
  option_to_pass[Options::Algorithm::ExpandBroadcastConst] = &createPassInstance<luci::ExpandBroadcastConstPass>;
  option_to_pass[Options::Algorithm::RemoveDuplicateConst] = &createPassInstance<luci::RemoveDuplicateConstPass>;
  option_to_pass[Options::Algorithm::RemoveFakeQuant] = &createPassInstance<luci::RemoveFakeQuantPass>;
  option_to_pass[Options::Algorithm::RemoveGatherGuard] = &createPassInstance<luci::RemoveGatherGuardPass>;
  option_to_pass[Options::Algorithm::RemoveQDQForMixedPrecisionOp] = &createPassInstance<luci::RemoveQDQForMixedPrecisionOpPass>;
  option_to_pass[Options::Algorithm::RemoveQuantDequantSeq] = &createPassInstance<luci::RemoveQuantDequantSeqPass>;
  option_to_pass[Options::Algorithm::RemoveUnnecessaryAdd] = &createPassInstance<luci::RemoveUnnecessaryAddPass>;
  option_to_pass[Options::Algorithm::RemoveUnnecessaryCast] = &createPassInstance<luci::RemoveUnnecessaryCastPass>;
  option_to_pass[Options::Algorithm::RemoveUnnecessaryDiv] = &createPassInstance<luci::RemoveUnnecessaryDivPass>;
  option_to_pass[Options::Algorithm::RemoveUnnecessaryMul] = &createPassInstance<luci::RemoveUnnecessaryMulPass>;
  option_to_pass[Options::Algorithm::RemoveUnnecessarySlice] = &createPassInstance<luci::RemoveUnnecessarySlicePass>;
  option_to_pass[Options::Algorithm::RemoveUnnecessaryStridedSlice] = &createPassInstance<luci::RemoveUnnecessaryStridedSlicePass>;
  option_to_pass[Options::Algorithm::RemoveUnnecessarySplit] = &createPassInstance<luci::RemoveUnnecessarySplitPass>;
  option_to_pass[Options::Algorithm::RemoveUnnecessaryTranspose] = &createPassInstance<luci::RemoveUnnecessaryTransposeNetPass>;
  option_to_pass[Options::Algorithm::RemoveRedundantQuantize] = &createPassInstance<luci::RemoveRedundantQuantizePass>;
  option_to_pass[Options::Algorithm::ReplaceNonConstFCWithBatchMatMul] = &createPassInstance<luci::ReplaceNonConstFCWithBatchMatMulPass>;
  option_to_pass[Options::Algorithm::ReplaceMulAddWithDepthwiseConv] = &createPassInstance<luci::ReplaceMulAddWithDepthwiseConvPass>;
  option_to_pass[Options::Algorithm::ReplaceSubWithAdd] = &createPassInstance<luci::ReplaceSubWithAddPass>;
  option_to_pass[Options::Algorithm::ReplaceWithFCGeluFC] = &createPassInstance<luci::ReplaceWithFCGeluFCPass>;
  option_to_pass[Options::Algorithm::SubstitutePadV2ToPad] = &createPassInstance<luci::SubstitutePadV2ToPadPass>;
  option_to_pass[Options::Algorithm::SubstituteSplitVToSplit] = &createPassInstance<luci::SubstituteSplitVToSplitPass>;
  option_to_pass[Options::Algorithm::TransformMinMaxToRelu6Pass] = &createPassInstance<luci::TransformMinMaxToRelu6Pass>;
  option_to_pass[Options::Algorithm::TransformMinReluToRelu6Pass] = &createPassInstance<luci::TransformMinReluToRelu6Pass>;
  option_to_pass[Options::Algorithm::TransformSqrtDivToRsqrtMul] = &createPassInstance<luci::TransformSqrtDivToRsqrtMulPass>;
  option_to_pass[Options::Algorithm::DecomposeHardSwishPass] = &createPassInstance<luci::DecomposeHardSwishPass>;
  option_to_pass[Options::Algorithm::DecomposeSoftmaxPass] = &createPassInstance<luci::DecomposeSoftmaxPass>;
  option_to_pass[Options::Algorithm::UnrollUnidirSeqLSTM] = &createPassInstance<luci::UnrollUnidirectionalSequenceLSTMPass>;
  // NOTE Experimental options; these will be removed someday
  //      Add experimental options here
  option_to_pass[Options::Algorithm::XpSepActFromTransposeConv] = &createPassInstance<luci::XpSepActFromTransposeConvPass>;
  option_to_pass[Options::Algorithm::ForwardReshapeToUnaryOp] = &createPassInstance<luci::ForwardReshapeToUnaryOpPass>;
  option_to_pass[Options::Algorithm::ForwardTransposeOp] = &createPassInstance<luci::ForwardTransposeOpPass>;
  // clang-format on 

  for (auto const &m : option_to_pass)
  {
    if (_options->query(m.first))
    {
      phase.emplace_back(m.second());
    }
  }

  // TODO Extend `option_to_pass` to be able to instantiate two or more pass objects.
  if (_options->query(Options::Algorithm::RemoveUnnecessaryReshape))
  {
    phase.emplace_back(std::make_unique<luci::RemoveUnnecessaryReshapePass>());
    phase.emplace_back(std::make_unique<luci::RemoveUnnecessaryReshapeNetPass>());
  }

  /* TRANSFORM DECLARATION END */

  ProgressReporter prog(g, logo::PhaseStrategy::Restart);
  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{g};
  phase_runner.attach(&prog);
  phase_runner.run(phase);
}

void CircleOptimizer::sparsify(loco::Graph *g) const
{
  if (_options->query(Options::Algorithm::SparsifyTensorPass))
  {
    std::string tensor_name = _options->param(Options::AlgorithmParameters::Sparsify_tensor_name);
    std::string str_tarversal_order =
      _options->param(Options::AlgorithmParameters::Sparsify_traversal_order);
    std::string str_format = _options->param(Options::AlgorithmParameters::Sparsify_format);
    std::string str_block_size = _options->param(Options::AlgorithmParameters::Sparsify_block_size);
    std::string str_block_map = _options->param(Options::AlgorithmParameters::Sparsify_block_map);

    // traversal order
    std::vector<int32_t> traversal_order = pepper::csv_to_vector<int32_t>(str_tarversal_order);
    // format
    std::vector<DimensionType> format;
    std::istringstream is(str_format);
    for (char c; is >> c;)
    {
      assert(c != ',');
      if (c == 'd')
        format.push_back(DimensionType::DENSE);
      else if (c == 's')
        format.push_back(DimensionType::SPARSE_CSR);
      if (is.peek() == ',')
        is.ignore();
    }
    // block size
    std::vector<int32_t> block_size = pepper::csv_to_vector<int32_t>(str_block_size);
    // block map
    std::vector<int32_t> block_map = pepper::csv_to_vector<int32_t>(str_block_map);

    luci::SparsifyTensorPass sparsifier{tensor_name, traversal_order, format, block_size,
                                        block_map};
    sparsifier.run(g);
  }
}

} // namespace luci
