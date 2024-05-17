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

#ifndef __LUCI_CIRCLE_OPTIMIZER_H__
#define __LUCI_CIRCLE_OPTIMIZER_H__

#include <loco.h>

#include <luci/IR/Module.h>

#include <string>
#include <vector>

namespace luci
{

class CircleOptimizer final
{
public:
  struct Options
  {
    enum Algorithm
    {
      FuseAddWithConv,
      FuseAddWithFullyConnected,
      FuseAddWithTConv,
      FuseBatchNormWithConv,
      FuseBatchNormWithDwConv,
      FuseBatchNormWithTConv,
      FuseSliceWithTConv,
      FuseBCQ,
      FuseHorizontalFullyConnected,
      FuseInstanceNorm,
      FuseMeanWithMean,
      FuseMulWithConv,
      FuseMulWithDiv,
      FuseTransposeWithMean,
      ResolveCustomOpAdd,
      ResolveCustomOpBatchMatMul,
      ResolveCustomOpMatMul,
      ResolveCustomOpMaxPoolWithArgmax,
      ResolveCustomOpSplitV,
      ResolveFormerCustomOp,
      FoldAddV2,
      FoldCast,
      FoldDensify,
      FoldDepthwiseConv2D,
      FoldFullyConnected,
      FoldDequantize,
      FoldGather,
      FoldReshape,
      FoldShape,
      FoldSparseToDense,
      FoldSqueeze,
      ForwardReshapeToUnaryOp,
      ForwardTransposeOp,
      SparsifyTensorPass,
      FusePreActivationBatchNorm,
      MakeBatchNormGammaPositive,
      FuseActivationFunction,
      FusePRelu,
      FuseGelu,
      FuseRsqrt,
      ShuffleWeightTo16x1Float32,
      RemoveRedundantTranspose,
      ReplaceMulAddWithDepthwiseConv,
      ReplaceNonConstFCWithBatchMatMul,
      ReplaceSubWithAdd,
      ReplaceWithFCGeluFC,
      SubstitutePackToReshape,
      SubstitutePadV2ToPad,
      SubstituteSplitVToSplit,
      SubstituteSqueezeToReshape,
      ExpandBroadcastConst,
      ConvertNCHWToNHWC,
      CommonSubExpressionElimination,
      RemoveUnnecessaryAdd,
      RemoveUnnecessarySlice,
      RemoveUnnecessaryStridedSlice,
      RemoveUnnecessarySplit,
      RemoveUnnecessaryReshape,
      RemoveUnnecessaryTranspose,
      TransformMinMaxToRelu6Pass,
      TransformMinReluToRelu6Pass,
      DecomposeHardSwishPass,
      DecomposeSoftmaxPass,
      SubstituteStridedSliceToReshape,
      SubstituteTransposeToReshape,
      RemoveRedundantQuantize,
      RemoveRedundantReshape,
      RemoveFakeQuant,
      RemoveQDQForMixedPrecisionOp,
      RemoveQuantDequantSeq,
      RemoveDuplicateConst,
      UnrollUnidirSeqLSTM,
      XpSepActFromTransposeConv,
      RemoveGatherGuard,
    };

    enum AlgorithmParameters
    {
      // sparsify
      Sparsify_tensor_name,
      Sparsify_traversal_order,
      Sparsify_format,
      Sparsify_block_size,
      Sparsify_block_map,

      // convert NCHW to NHWC
      NCHW_to_NHWC_input_shape,
      NCHW_to_NHWC_output_shape,
    };

    virtual ~Options() = default;

    virtual void enable(Algorithm) = 0;
    virtual bool query(Algorithm) = 0;
    virtual void param(AlgorithmParameters, const std::string &) = 0;
    virtual const std::string param(AlgorithmParameters) const = 0;
  };

public:
  // TODO maybe caller can provide Options as ctor parameters
  Options *options(void);

public:
  void optimize(luci::Module *) const;

  void optimize(loco::Graph *) const;

  void sparsify(loco::Graph *) const;

private:
  std::unique_ptr<Options> _options;
};

} // namespace luci

#endif // __LUCI_CIRCLE_OPTIMIZER_H__
