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
      return loco::dtype_get(node->inputs(0));
    }
    return loco::DataType::Unknown;
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

  loco::DataType visit(const luci::CircleIf *node) final
  {
    // Type of If is not used. Just use input 1
    assert(node->arity() > 1);
    return loco::dtype_get(node->input(1));
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

  loco::DataType visit(const luci::CircleMinimum *node) final { return loco::dtype_get(node->x()); }

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

  loco::DataType visit(const luci::CircleSin *node) final { return loco::dtype_get(node->x()); }

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

  loco::DataType visit(const luci::CircleTile *node) final
  {
    return loco::dtype_get(node->input());
  }

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
    auto graph_outputs = node->graph()->outputs();
    auto graph_output = graph_outputs->at(node->index());
    auto output_dtype = graph_output->dtype();

    if (dynamic_cast<luci::CircleOutputDummy *>(node->from()) == nullptr)
    {
      // We don't care for the type if from() is CircleOutputDummy
      // from() type should match that of CircleOutput
      assert(output_dtype == loco::dtype_get(node->from()));
    }
    return output_dtype;
  }

  loco::DataType visit(const luci::CircleOutputDummy *node) final { return node->dtype(); }

  loco::DataType visit(const luci::CircleIfOut *node) final
  {
    /**
     * @note  IF operator type and shape are that of the "then" and "else"
     *        Graph Outputs.
     */
    auto circle_if = dynamic_cast<const luci::CircleIf *>(node->input());
    if (circle_if == nullptr)
    {
      INTERNAL_EXN("CircleIf IR is not configured correctly");
    }

    auto index = node->index();
    auto then_graph = circle_if->then_graph();
    auto else_graph = circle_if->else_graph();
    assert(then_graph != nullptr);
    assert(else_graph != nullptr);

    // shape and type are assumed to be same
    // these are checked at post_import_graph() in Import
    auto then_outputs = loco::output_nodes(then_graph);
    auto else_outputs = loco::output_nodes(else_graph);
    assert(then_outputs.size() == else_outputs.size());
    assert(index < static_cast<int32_t>(then_outputs.size()));

    auto then_out = dynamic_cast<luci::CircleOutput *>(then_outputs.at(index));
    auto else_out = dynamic_cast<luci::CircleOutput *>(else_outputs.at(index));
    assert(then_out != nullptr);
    assert(else_out != nullptr);

    auto then_graph_outputs = then_graph->outputs(); // loco::GraphOutput items
    auto else_graph_outputs = else_graph->outputs();
    assert(then_graph_outputs->size() == else_graph_outputs->size());

    auto then_graph_output = then_graph_outputs->at(then_out->index());
    auto else_graph_output = else_graph_outputs->at(else_out->index());
    (void)else_graph_output; // make compiler happy for unused variable warnings
    assert(then_graph_output->dtype() == else_graph_output->dtype());

    return then_graph_output->dtype();
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
