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

#include "luci/Service/CircleTypeInferenceRule.h"
#include "CircleTypeInferenceHelper.h"

#include <luci/IR/CircleDialect.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/IR/CircleNodes.h>

#include <cassert>

namespace
{

struct TypeInferenceAlgorithm final : public luci::CircleNodeVisitor<loco::DataType>
{
  // TODO Given a tensor x of complex numbers, Abs operation returns a tensor of type float32 or
  // float64.
  loco::DataType visit(const luci::CircleAbs *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleAdd *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleAddN *node) final
  {
    auto dtype = luci::dtype_get(node->inputs(0));

    for (uint32_t idx = 1; idx < node->arity(); ++idx)
    {
      auto dtype_idx = luci::dtype_get(node->inputs(idx));
      if (dtype != dtype_idx)
      {
        INTERNAL_EXN_V("ADD_N dtype not same as the first input: ", idx);
      }
    }

    return luci::dtype_get(node->inputs(0));
  }

  loco::DataType visit(const luci::CircleArgMax *node) final { return node->output_type(); }

  loco::DataType visit(const luci::CircleArgMin *node) final { return node->output_type(); }

  loco::DataType visit(const luci::CircleAveragePool2D *node) final
  {
    return luci::dtype_get(node->value());
  }

  loco::DataType visit(const luci::CircleBatchMatMul *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleBatchToSpaceND *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleBroadcastTo *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleCast *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleCeil *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleConcatenation *node) final
  {
    // TODO Support when CircleConcatenation has 0 input
    assert(node->numValues() > 0);

    for (uint32_t i = 1; i < node->numValues(); ++i)
      assert(luci::dtype_get(node->values(i - 1)) == luci::dtype_get(node->values(i)));

    return luci::dtype_get(node->values(0));
  }

  loco::DataType visit(const luci::CircleConst *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleConv2D *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleCos *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleCumSum *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleCustom *node) final
  {
    if (node->custom_code() == "BatchMatMulV2")
    {
      return luci::dtype_get(node->inputs(0));
    }
    return node->dtype();
  }

  loco::DataType visit(const luci::CircleDensify *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleDepthToSpace *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleDepthwiseConv2D *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleDequantize *) final { return loco::DataType::FLOAT32; }

  loco::DataType visit(const luci::CircleDiv *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleElu *node) final
  {
    return luci::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleEqual *) final { return loco::DataType::BOOL; }

  loco::DataType visit(const luci::CircleExp *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleExpandDims *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleFakeQuant *node) final
  {
    return luci::dtype_get(node->inputs());
  }

  loco::DataType visit(const luci::CircleFill *node) final
  {
    return luci::dtype_get(node->value());
  }

  loco::DataType visit(const luci::CircleFloor *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleFloorDiv *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleFloorMod *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleFullyConnected *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleGather *node) final
  {
    return luci::dtype_get(node->params());
  }

  loco::DataType visit(const luci::CircleGatherNd *node) final
  {
    return luci::dtype_get(node->params());
  }

  loco::DataType visit(const luci::CircleGelu *node) final
  {
    return luci::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleGreater *) final { return loco::DataType::BOOL; }

  loco::DataType visit(const luci::CircleGreaterEqual *) final { return loco::DataType::BOOL; }

  loco::DataType visit(const luci::CircleHardSwish *node) final
  {
    return luci::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleCirGru *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleIf *node) final
  {
    // Type of If is not used. Just use input 0
    assert(node->input_count() > 0);
    return luci::dtype_get(node->input(0));
  }

  loco::DataType visit(const luci::CircleL2Normalize *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleL2Pool2D *node) final
  {
    return luci::dtype_get(node->value());
  }

  loco::DataType visit(const luci::CircleLeakyRelu *node) final
  {
    return luci::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleLess *) final { return loco::DataType::BOOL; }

  loco::DataType visit(const luci::CircleLessEqual *) final { return loco::DataType::BOOL; }

  loco::DataType visit(const luci::CircleLocalResponseNormalization *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleLog *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleLogicalAnd *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleLogicalNot *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleLogicalOr *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleLogistic *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleLogSoftmax *node) final
  {
    return luci::dtype_get(node->logits());
  }

  loco::DataType visit(const luci::CircleMatrixDiag *node) final
  {
    return luci::dtype_get(node->diagonal());
  }

  loco::DataType visit(const luci::CircleMatrixSetDiag *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleMaximum *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleMaxPool2D *node) final
  {
    return luci::dtype_get(node->value());
  }

  loco::DataType visit(const luci::CircleMean *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleMinimum *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleMirrorPad *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleNeg *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleNonMaxSuppressionV4 *node) final
  {
    return luci::dtype_get(node->boxes());
  }

  loco::DataType visit(const luci::CircleNonMaxSuppressionV5 *node) final
  {
    return luci::dtype_get(node->boxes());
  }

  loco::DataType visit(const luci::CircleNotEqual *) final { return loco::DataType::BOOL; }

  loco::DataType visit(const luci::CirclePack *node) final
  {
    // Only support CirclePack with one or more inputs
    assert(node->values_count() > 0);

    auto first_value_type = luci::dtype_get(node->values(0));
    for (uint32_t i = 1; i < node->values_count(); ++i)
      assert(first_value_type == luci::dtype_get(node->values(i)));

    return first_value_type;
  }

  loco::DataType visit(const luci::CirclePad *node) final { return luci::dtype_get(node->input()); }

  loco::DataType visit(const luci::CirclePadV2 *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CirclePow *node) final
  {
    // TODO make sure types cannot differ
    auto x_type = luci::dtype_get(node->x());
    auto y_type = luci::dtype_get(node->y());

    if (x_type != y_type)
      INTERNAL_EXN("Different datatype for x and y are not supported");

    return x_type;
  }

  loco::DataType visit(const luci::CirclePRelu *node) final
  {
    auto input_type = luci::dtype_get(node->input());
    auto alpha_type = luci::dtype_get(node->alpha());

    if (input_type != alpha_type)
      INTERNAL_EXN("Different datatype for input and alpha are not supported");

    return input_type;
  }

  loco::DataType visit(const luci::CircleQuantize *node) final { return luci::dtype_get(node); }

  loco::DataType visit(const luci::CircleRange *node) final
  {
    return luci::dtype_get(node->start());
  }

  loco::DataType visit(const luci::CircleRank *) final { return loco::DataType::S32; }

  loco::DataType visit(const luci::CircleMul *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleOneHot *node) final
  {
    return luci::dtype_get(node->on_value());
  }

  loco::DataType visit(const luci::CircleReduceAny *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleReduceMax *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleReduceMin *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleReduceProd *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleRelu *node) final
  {
    return luci::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleRelu6 *node) final
  {
    return luci::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleReluN1To1 *node) final
  {
    return luci::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleReshape *node) final
  {
    return luci::dtype_get(node->tensor());
  }

  loco::DataType visit(const luci::CircleResizeBilinear *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleResizeNearestNeighbor *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleReverseSequence *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleReverseV2 *node) final
  {
    return luci::dtype_get(node->tensor());
  }

  loco::DataType visit(const luci::CircleRound *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleRsqrt *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleScatterNd *node) final
  {
    return luci::dtype_get(node->updates());
  }

  loco::DataType visit(const luci::CircleSegmentSum *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleSelect *node) final
  {
    assert(luci::dtype_get(node->t()) == luci::dtype_get(node->e()));
    return luci::dtype_get(node->t());
  }

  loco::DataType visit(const luci::CircleSelectV2 *node) final
  {
    assert(luci::dtype_get(node->t()) == luci::dtype_get(node->e()));
    return luci::dtype_get(node->t());
  }

  loco::DataType visit(const luci::CircleShape *node) final { return node->out_type(); }

  loco::DataType visit(const luci::CircleSin *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleSlice *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleSoftmax *node) final
  {
    return luci::dtype_get(node->logits());
  }

  loco::DataType visit(const luci::CircleSpaceToBatchND *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleSpaceToDepth *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleSparseToDense *node) final
  {
    return luci::dtype_get(node->values());
  }

  loco::DataType visit(const luci::CircleSplit *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleSplitV *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleSqrt *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleSquare *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleSquaredDifference *node) final
  {
    return luci::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleSqueeze *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleStridedSlice *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleSub *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleSum *node) final { return luci::dtype_get(node->input()); }

  loco::DataType visit(const luci::CircleSVDF *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleTanh *node) final { return luci::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleTile *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleTopKV2 *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleTranspose *node) final
  {
    return luci::dtype_get(node->a());
  }

  loco::DataType visit(const luci::CircleTransposeConv *node) final
  {
    return luci::dtype_get(node->outBackprop());
  }

  loco::DataType visit(const luci::CircleUnidirectionalSequenceLSTM *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleUnique *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleUnpack *node) final
  {
    return luci::dtype_get(node->value());
  }

  loco::DataType visit(const luci::CircleWhere *) final { return loco::DataType::S64; }

  loco::DataType visit(const luci::CircleWhile *node) final
  {
    // Type of While is not used. Just use input 0
    assert(node->input_count() > 0);
    return luci::dtype_get(node->input(0));
  }

  loco::DataType visit(const luci::CircleZerosLike *node) final
  {
    return luci::dtype_get(node->input());
  }

  // Circle Only
  loco::DataType visit(const luci::CircleBCQFullyConnected *) final
  {
    return loco::DataType::FLOAT32;
  }

  loco::DataType visit(const luci::CircleBCQGather *) final { return loco::DataType::FLOAT32; }

  loco::DataType visit(const luci::CircleInstanceNorm *node) final
  {
    return luci::dtype_get(node->input());
  }

  // Virtual
  loco::DataType visit(const luci::CircleInput *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleOutput *node) final
  {
    auto graph_outputs = node->graph()->outputs();
    auto graph_output = graph_outputs->at(node->index());
    auto output_dtype = graph_output->dtype();

    if (dynamic_cast<luci::CircleOutputDummy *>(node->from()) == nullptr &&
        dynamic_cast<luci::CircleOutputExclude *>(node->from()) == nullptr)
    {
      // We don't care for the type if from() is CircleOutputDummy or CircleOutputExclude
      // from() type should match that of CircleOutput
      assert(output_dtype == luci::dtype_get(node->from()));
    }
    return output_dtype;
  }

  loco::DataType visit(const luci::CircleOutputDummy *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleOutputExclude *node) final
  {
    // NOTE We don't care CircleOutputExclude dtype, but set to FLOAT32
    //      if it's Unknown to make type inference happy.
    if (node->dtype() == loco::DataType::Unknown)
      return loco::DataType::FLOAT32;
    return node->dtype();
  }

  loco::DataType visit(const luci::CircleCustomOut *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleNonMaxSuppressionV4Out *node) final
  {
    (void)node;
    assert(node->index() == 0 || node->index() == 1);
    return loco::DataType::S32;
  }

  loco::DataType visit(const luci::CircleNonMaxSuppressionV5Out *node) final
  {
    (void)node;
    if (node->index() == 0 || node->index() == 2)
    {
      return loco::DataType::S32;
    }
    assert(node->index() == 1);
    return loco::DataType::FLOAT32;
  }

  loco::DataType visit(const luci::CircleSplitOut *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleSplitVOut *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleTopKV2Out *node) final
  {
    // First output is same as input
    if (node->index() == 0)
      return luci::dtype_get(node->input());
    // Second outout is always S32
    assert(node->index() == 1);
    return loco::DataType::S32;
  }

  loco::DataType visit(const luci::CircleVariable *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleUniqueOut *node) final
  {
    if (node->index() == 0)
    {
      return luci::dtype_get(node->input());
    }
    assert(node->index() == 1);
    auto unique = loco::must_cast<luci::CircleUnique *>(node->input());
    return unique->idx_out_type();
  }

  loco::DataType visit(const luci::CircleUnpackOut *node) final
  {
    return luci::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleWhileOut *node) final
  {
    /**
     * @note  WHILE operator's type is the same with the "cond"
     *        Graph Input.
     */
    auto circle_while = dynamic_cast<const luci::CircleWhile *>(node->input());
    if (circle_while == nullptr)
    {
      INTERNAL_EXN("CircleWhile IR is not configured correctly");
    }

    auto index = node->index();
    auto cond_graph = circle_while->cond_graph();
    assert(cond_graph != nullptr);

    // Assumption: the index of CircleWhileOut matches with the index of input nodes returned by
    // loco::input_nodes
    auto cond_inputs = loco::input_nodes(cond_graph);
    auto cond_in = loco::must_cast<luci::CircleInput *>(cond_inputs.at(index));

    auto cond_graph_inputs = cond_graph->inputs();
    auto cond_graph_input = cond_graph_inputs->at(cond_in->index());

    return cond_graph_input->dtype();
  }
};

} // namespace

namespace luci
{

bool CircleTypeInferenceRule::recognize(const loco::Dialect *d) const
{
  return CircleDialect::get() == d;
}

bool CircleTypeInferenceRule::infer(const loco::Node *node, loco::DataType &dtype) const
{
  assert(node->dialect() == CircleDialect::get());

  TypeInferenceAlgorithm alg;

  auto circle_node = loco::must_cast<const CircleNode *>(node);
  dtype = circle_node->accept(&alg);
  assert(dtype != loco::DataType::Unknown);

  return true;
}

} // namespace luci
