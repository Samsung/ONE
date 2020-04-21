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
  loco::DataType visit(const luci::CircleAbs *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleAdd *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleArgMax *node) final { return node->output_type(); }

  loco::DataType visit(const luci::CircleAveragePool2D *node) final
  {
    return loco::dtype_get(node->value());
  }

  loco::DataType visit(const luci::CircleBatchMatMul *node) final
  {
    return loco::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleBatchToSpaceND *node) final
  {
    return loco::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleConcatenation *node) final
  {
    // TODO Support when CircleConcatenation has 0 input
    assert(node->numValues() > 0);

    for (uint32_t i = 1; i < node->numValues(); ++i)
      assert(loco::dtype_get(node->values(i - 1)) == loco::dtype_get(node->values(i)));

    return loco::dtype_get(node->values(0));
  }

  loco::DataType visit(const luci::CircleConst *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleConv2D *node) final
  {
    return loco::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleCos *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleCustom *node) final
  {
    if (node->custom_code() == "BatchMatMulV2")
    {
      return loco::dtype_get(node->inputs()[0]);
    }
  }

  loco::DataType visit(const luci::CircleDepthwiseConv2D *node) final
  {
    return loco::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleDiv *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleEqual *) final { return loco::DataType::BOOL; }

  loco::DataType visit(const luci::CircleExp *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleFullyConnected *node) final
  {
    return loco::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CircleGather *node) final
  {
    return loco::dtype_get(node->params());
  }

  loco::DataType visit(const luci::CircleLogicalNot *node) final
  {
    return loco::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleLogicalOr *node) final
  {
    return loco::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleMaximum *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleMaxPool2D *node) final
  {
    return loco::dtype_get(node->value());
  }

  loco::DataType visit(const luci::CircleMean *node) final
  {
    return loco::dtype_get(node->input());
  }

  loco::DataType visit(const luci::CirclePack *node) final
  {
    // Only support CirclePack with one or more inputs
    assert(node->values_count() > 0);

    auto first_value_type = loco::dtype_get(node->values(0));
    for (uint32_t i = 1; i < node->values_count(); ++i)
      assert(first_value_type == loco::dtype_get(node->values(i)));

    return first_value_type;
  }

  loco::DataType visit(const luci::CirclePad *node) final { return loco::dtype_get(node->input()); }

  loco::DataType visit(const luci::CircleMul *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleRelu *node) final
  {
    return loco::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleRelu6 *node) final
  {
    return loco::dtype_get(node->features());
  }

  loco::DataType visit(const luci::CircleReshape *node) final
  {
    return loco::dtype_get(node->tensor());
  }

  loco::DataType visit(const luci::CircleRsqrt *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleSoftmax *node) final
  {
    return loco::dtype_get(node->logits());
  }

  loco::DataType visit(const luci::CircleSqrt *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleSquaredDifference *node) final
  {
    return loco::dtype_get(node->x());
  }

  loco::DataType visit(const luci::CircleSub *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleTanh *node) final { return loco::dtype_get(node->x()); }

  loco::DataType visit(const luci::CircleTranspose *node) final
  {
    return loco::dtype_get(node->a());
  }

  loco::DataType visit(const luci::CircleTransposeConv *node) final
  {
    return loco::dtype_get(node->outBackprop());
  }

  loco::DataType visit(const luci::CircleUnpack *node) final
  {
    return loco::dtype_get(node->value());
  }

  // Circle Only
  loco::DataType visit(const luci::CircleInstanceNorm *node) final
  {
    return loco::dtype_get(node->input());
  }

  // Virtual
  loco::DataType visit(const luci::CircleInput *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleOutput *node) final
  {
    return loco::dtype_get(node->from());
  }

  loco::DataType visit(const luci::CircleUnpackOut *node) final
  {
    return loco::dtype_get(node->unpack());
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

  dtype = dynamic_cast<const CircleNode *>(node)->accept(&alg);
  assert(dtype != loco::DataType::Unknown);

  return true;
}

} // namespace luci
