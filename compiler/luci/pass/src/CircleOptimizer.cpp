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
#include "luci/Pass/FoldSparseToDensePass.h"
#include "luci/Pass/ForwardReshapeToUnaryOpPass.h"
#include "luci/Pass/ForwardTransposeOpPass.h"
#include "luci/Pass/FuseActivationFunctionPass.h"
#include "luci/Pass/FuseAddWithFullyConnectedPass.h"
#include "luci/Pass/FuseAddWithTConvPass.h"
#include "luci/Pass/FuseBatchNormWithConvPass.h"
#include "luci/Pass/FuseBatchNormWithDwConvPass.h"
#include "luci/Pass/FuseBatchNormWithTConvPass.h"
#include "luci/Pass/FuseBCQPass.h"
#include "luci/Pass/FuseInstanceNormPass.h"
#include "luci/Pass/FuseMeanWithMeanPass.h"
#include "luci/Pass/FusePreActivationBatchNormPass.h"
#include "luci/Pass/FusePReluPass.h"
#include "luci/Pass/FuseGeluPass.h"
#include "luci/Pass/FuseSliceWithTConvPass.h"
#include "luci/Pass/FuseHorizontalFullyConnectedPass.h"
#include "luci/Pass/FuseTransposeWithMeanPass.h"
#include "luci/Pass/MakeBatchNormGammaPositivePass.h"
#include "luci/Pass/RemoveDuplicateConstPass.h"
#include "luci/Pass/RemoveFakeQuantPass.h"
#include "luci/Pass/RemoveQuantDequantSeqPass.h"
#include "luci/Pass/RemoveRedundantReshapePass.h"
#include "luci/Pass/RemoveRedundantTransposePass.h"
#include "luci/Pass/RemoveRedundantQuantizePass.h"
#include "luci/Pass/RemoveUnnecessaryAddPass.h"
#include "luci/Pass/RemoveUnnecessaryReshapePass.h"
#include "luci/Pass/RemoveUnnecessaryReshapeNetPass.h"
#include "luci/Pass/RemoveUnnecessarySlicePass.h"
#include "luci/Pass/RemoveUnnecessaryStridedSlicePass.h"
#include "luci/Pass/RemoveUnnecessarySplitPass.h"
#include "luci/Pass/ReplaceNonConstFCWithBatchMatMulPass.h"
#include "luci/Pass/ReplaceMulAddWithDepthwiseConvPass.h"
#include "luci/Pass/ReplaceSubWithAddPass.h"
#include "luci/Pass/ReplaceWithFCGeluFCPass.h"
#include "luci/Pass/ResolveCustomOpAddPass.h"
#include "luci/Pass/ResolveCustomOpBatchMatMulPass.h"
#include "luci/Pass/ResolveCustomOpMatMulPass.h"
#include "luci/Pass/ResolveCustomOpMaxPoolWithArgmaxPass.h"
#include "luci/Pass/ResolveCustomOpSplitVPass.h"
#include "luci/Pass/SparsifyTensorPass.h"
#include "luci/Pass/ShuffleWeightTo16x1Float32Pass.h"
#include "luci/Pass/SubstitutePackToReshapePass.h"
#include "luci/Pass/SubstitutePadV2ToPadPass.h"
#include "luci/Pass/SubstituteSplitVToSplitPass.h"
#include "luci/Pass/SubstituteSqueezeToReshapePass.h"
#include "luci/Pass/SubstituteStridedSliceToReshapePass.h"
#include "luci/Pass/SubstituteTransposeToReshapePass.h"
#include "luci/Pass/TransformMinMaxToRelu6Pass.h"
#include "luci/Pass/TransformMinReluToRelu6Pass.h"
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

void CircleOptimizer::optimize(loco::Graph *g) const
{
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

  if (_options->query(Options::Algorithm::CommonSubExpressionElimination))
  {
    phase.emplace_back(std::make_unique<luci::CommonSubExpressionEliminationPass>());
  }
  if (_options->query(Options::Algorithm::ResolveCustomOpAdd))
  {
    phase.emplace_back(std::make_unique<luci::ResolveCustomOpAddPass>());
  }
  if (_options->query(Options::Algorithm::ResolveCustomOpBatchMatMul))
  {
    phase.emplace_back(std::make_unique<luci::ResolveCustomOpBatchMatMulPass>());
  }
  if (_options->query(Options::Algorithm::ResolveCustomOpMatMul))
  {
    phase.emplace_back(std::make_unique<luci::ResolveCustomOpMatMulPass>());
  }
  if (_options->query(Options::Algorithm::FuseMeanWithMean))
  {
    phase.emplace_back(std::make_unique<FuseMeanWithMeanPass>());
  }
  if (_options->query(Options::Algorithm::ResolveCustomOpMaxPoolWithArgmax))
  {
    phase.emplace_back(std::make_unique<luci::ResolveCustomOpMaxPoolWithArgmaxPass>());
  }
  if (_options->query(Options::Algorithm::ResolveCustomOpSplitV))
  {
    phase.emplace_back(std::make_unique<luci::ResolveCustomOpSplitVPass>());
  }
  if (_options->query(Options::Algorithm::FuseInstanceNorm))
  {
    phase.emplace_back(std::make_unique<FuseInstanceNormPass>());
  }
  if (_options->query(Options::Algorithm::FuseBatchNormWithConv))
  {
    phase.emplace_back(std::make_unique<FuseBatchNormWithConvPass>());
  }
  if (_options->query(Options::Algorithm::FuseBatchNormWithDwConv))
  {
    phase.emplace_back(std::make_unique<FuseBatchNormWithDwConvPass>());
  }
  if (_options->query(Options::Algorithm::FuseBatchNormWithTConv))
  {
    phase.emplace_back(std::make_unique<FuseBatchNormWithTConvPass>());
  }
  if (_options->query(Options::Algorithm::FuseSliceWithTConv))
  {
    phase.emplace_back(std::make_unique<FuseSliceWithTConvPass>());
  }
  if (_options->query(Options::Algorithm::FuseAddWithFullyConnected))
  {
    phase.emplace_back(std::make_unique<FuseAddWithFullyConnectedPass>());
  }
  if (_options->query(Options::Algorithm::FuseAddWithTConv))
  {
    phase.emplace_back(std::make_unique<FuseAddWithTConvPass>());
  }
  if (_options->query(Options::Algorithm::FuseActivationFunction))
  {
    phase.emplace_back(std::make_unique<FuseActivationFunctionPass>());
  }
  if (_options->query(Options::Algorithm::FusePRelu))
  {
    phase.emplace_back(std::make_unique<FusePReluPass>());
  }
  if (_options->query(Options::Algorithm::FuseGelu))
  {
    phase.emplace_back(std::make_unique<FuseGeluPass>());
  }
  if (_options->query(Options::Algorithm::FuseHorizontalFullyConnected))
  {
    phase.emplace_back(std::make_unique<FuseHorizontalFullyConnectedPass>());
  }
  if (_options->query(Options::Algorithm::FuseTransposeWithMean))
  {
    phase.emplace_back(std::make_unique<FuseTransposeWithMeanPass>());
  }
  if (_options->query(Options::Algorithm::FoldAddV2))
  {
    phase.emplace_back(std::make_unique<luci::FoldAddV2Pass>());
  }
  if (_options->query(Options::Algorithm::FoldCast))
  {
    phase.emplace_back(std::make_unique<luci::FoldCastPass>());
  }
  if (_options->query(Options::Algorithm::FoldDensify))
  {
    phase.emplace_back(std::make_unique<luci::FoldDensifyPass>());
  }
  if (_options->query(Options::Algorithm::FoldDepthwiseConv2D))
  {
    phase.emplace_back(std::make_unique<luci::FoldDepthwiseConv2DPass>());
  }
  if (_options->query(Options::Algorithm::FoldDequantize))
  {
    phase.emplace_back(std::make_unique<luci::FoldDequantizePass>());
  }
  if (_options->query(Options::Algorithm::FoldFullyConnected))
  {
    phase.emplace_back(std::make_unique<luci::FoldFullyConnectedPass>());
  }
  if (_options->query(Options::Algorithm::FoldGather))
  {
    phase.emplace_back(std::make_unique<luci::FoldGatherPass>());
  }
  if (_options->query(Options::Algorithm::FoldSparseToDense))
  {
    phase.emplace_back(std::make_unique<luci::FoldSparseToDensePass>());
  }
  if (_options->query(Options::Algorithm::FusePreActivationBatchNorm))
  {
    phase.emplace_back(std::make_unique<luci::FusePreActivationBatchNormPass>());
  }
  if (_options->query(Options::Algorithm::MakeBatchNormGammaPositive))
  {
    phase.emplace_back(std::make_unique<luci::MakeBatchNormGammaPositivePass>());
  }
  if (_options->query(Options::Algorithm::ShuffleWeightTo16x1Float32))
  {
    phase.emplace_back(std::make_unique<luci::ShuffleWeightTo16x1Float32Pass>());
  }
  if (_options->query(Options::Algorithm::ExpandBroadcastConst))
  {
    phase.emplace_back(std::make_unique<luci::ExpandBroadcastConstPass>());
  }
  if (_options->query(Options::Algorithm::RemoveDuplicateConst))
  {
    phase.emplace_back(std::make_unique<luci::RemoveDuplicateConstPass>());
  }
  if (_options->query(Options::Algorithm::RemoveFakeQuant))
  {
    phase.emplace_back(std::make_unique<luci::RemoveFakeQuantPass>());
  }
  if (_options->query(Options::Algorithm::RemoveQuantDequantSeq))
  {
    phase.emplace_back(std::make_unique<luci::RemoveQuantDequantSeqPass>());
  }
  if (_options->query(Options::Algorithm::RemoveUnnecessaryAdd))
  {
    phase.emplace_back(std::make_unique<luci::RemoveUnnecessaryAddPass>());
  }
  if (_options->query(Options::Algorithm::RemoveUnnecessaryReshape))
  {
    phase.emplace_back(std::make_unique<luci::RemoveUnnecessaryReshapePass>());
    phase.emplace_back(std::make_unique<luci::RemoveUnnecessaryReshapeNetPass>());
  }
  if (_options->query(Options::Algorithm::RemoveUnnecessarySlice))
  {
    phase.emplace_back(std::make_unique<luci::RemoveUnnecessarySlicePass>());
  }
  if (_options->query(Options::Algorithm::RemoveUnnecessaryStridedSlice))
  {
    phase.emplace_back(std::make_unique<luci::RemoveUnnecessaryStridedSlicePass>());
  }
  if (_options->query(Options::Algorithm::RemoveUnnecessarySplit))
  {
    phase.emplace_back(std::make_unique<luci::RemoveUnnecessarySplitPass>());
  }
  if (_options->query(Options::Algorithm::RemoveRedundantReshape))
  {
    phase.emplace_back(std::make_unique<luci::RemoveRedundantReshapePass>());
  }
  if (_options->query(Options::Algorithm::RemoveRedundantTranspose))
  {
    phase.emplace_back(std::make_unique<luci::RemoveRedundantTransposePass>());
  }
  if (_options->query(Options::Algorithm::RemoveRedundantQuantize))
  {
    phase.emplace_back(std::make_unique<luci::RemoveRedundantQuantizePass>());
  }
  if (_options->query(Options::Algorithm::ReplaceNonConstFCWithBatchMatMul))
  {
    phase.emplace_back(std::make_unique<luci::ReplaceNonConstFCWithBatchMatMulPass>());
  }
  if (_options->query(Options::Algorithm::ReplaceMulAddWithDepthwiseConv))
  {
    phase.emplace_back(std::make_unique<luci::ReplaceMulAddWithDepthwiseConvPass>());
  }
  if (_options->query(Options::Algorithm::ReplaceSubWithAdd))
  {
    phase.emplace_back(std::make_unique<luci::ReplaceSubWithAddPass>());
  }
  if (_options->query(Options::Algorithm::ReplaceWithFCGeluFC))
  {
    phase.emplace_back(std::make_unique<luci::ReplaceWithFCGeluFCPass>());
  }
  if (_options->query(Options::Algorithm::SubstitutePackToReshape))
  {
    phase.emplace_back(std::make_unique<luci::SubstitutePackToReshapePass>());
  }
  if (_options->query(Options::Algorithm::SubstitutePadV2ToPad))
  {
    phase.emplace_back(std::make_unique<luci::SubstitutePadV2ToPadPass>());
  }
  if (_options->query(Options::Algorithm::SubstituteSplitVToSplit))
  {
    phase.emplace_back(std::make_unique<luci::SubstituteSplitVToSplitPass>());
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
  if (_options->query(Options::Algorithm::TransformMinMaxToRelu6Pass))
  {
    phase.emplace_back(std::make_unique<luci::TransformMinMaxToRelu6Pass>());
  }
  if (_options->query(Options::Algorithm::TransformMinReluToRelu6Pass))
  {
    phase.emplace_back(std::make_unique<luci::TransformMinReluToRelu6Pass>());
  }
  if (_options->query(Options::Algorithm::DecomposeHardSwishPass))
  {
    phase.emplace_back(std::make_unique<luci::DecomposeHardSwishPass>());
  }
  if (_options->query(Options::Algorithm::DecomposeSoftmaxPass))
  {
    phase.emplace_back(std::make_unique<luci::DecomposeSoftmaxPass>());
  }
  if (_options->query(Options::Algorithm::UnrollUnidirSeqLSTM))
  {
    phase.emplace_back(std::make_unique<luci::UnrollUnidirectionalSequenceLSTMPass>());
  }

  // NOTE Experimental options; these will be removed someday
  //      Add experimental options here
  if (_options->query(Options::Algorithm::XpSepActFromTransposeConv))
  {
    phase.emplace_back(std::make_unique<luci::XpSepActFromTransposeConvPass>());
  }

  // Forward Reshape/Transpose is done after
  // 1. SubstituteXXXToReshape
  // 2. RemoveRedundantReshape/Transpose
  // See https://github.com/Samsung/ONE/pull/10596 for more details
  if (_options->query(Options::Algorithm::ForwardReshapeToUnaryOp))
  {
    phase.emplace_back(std::make_unique<luci::ForwardReshapeToUnaryOpPass>());
  }
  if (_options->query(Options::Algorithm::ForwardTransposeOp))
  {
    phase.emplace_back(std::make_unique<luci::ForwardTransposeOpPass>());
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
