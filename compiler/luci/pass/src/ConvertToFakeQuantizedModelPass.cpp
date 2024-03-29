/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/ConvertToFakeQuantizedModelPass.h"
#include "luci/Pass/QuantizationParameters.h"

#include "QuantizationUtils.h"

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/IR/CircleNodes.h>
#include <luci/IR/CircleNodeVisitor.h>
#include <luci/Log.h>

namespace
{

// Create Quantize Op whose dtype/shape/qparam are the same with node
luci::CircleQuantize *create_quantize(luci::CircleNode *node)
{
  auto quantize = node->graph()->nodes()->create<luci::CircleQuantize>();
  // DESIGN NOTE: Why use '_FQ_Quantize' instead of '_Quantize'?
  // '_Quantize' is used in mixed-precision quantization
  // We add '_FQ' to distinguish Op from mixed-precision quantization
  quantize->name(node->name() + "_FQ_Quantize");
  quantize->dtype(node->dtype());
  quantize->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    quantize->dim(i).set(node->dim(i).value());

  quantize->shape_status(luci::ShapeStatus::VALID);

  copy_quantparam(node, quantize);

  luci::add_origin(quantize, luci::get_origin(node));

  return quantize;
}

// Create Dequantize Op whose shape is the same with node
luci::CircleDequantize *create_dequantize(luci::CircleNode *node)
{
  auto dequantize = node->graph()->nodes()->create<luci::CircleDequantize>();
  // DESIGN NOTE: Why use '_FQ_Dequantize' instead of '_Dequantize'?
  // '_Dequantize' is used in mixed-precision quantization
  // We add '_FQ' to distinguish Op from mixed-precision quantization
  dequantize->name(node->name() + "_FQ_Dequantize");
  dequantize->dtype(loco::DataType::FLOAT32);
  dequantize->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    dequantize->dim(i).set(node->dim(i).value());

  dequantize->shape_status(luci::ShapeStatus::VALID);

  luci::add_origin(dequantize, luci::get_origin(node));

  return dequantize;
}

// Return true if node is quantized activation
// 1. dtype is u8 or s16
// 2. node has qparam
bool is_quant_act(const luci::CircleNode *node)
{
  if (node->dtype() != loco::DataType::U8 and node->dtype() != loco::DataType::S16)
    return false;

  if (not node->quantparam())
    return false;

  return true;
}

// Return true if node is quantized const
// 1. dtype is not fp32
// 2. node has qparam
// NOTE Quantized const can have the following types
// s4 (weights), u4 (weights), u8 (weights, activation),
// s16 (weights, activation), s32 (bias), s64 (bias)
bool is_quant_const(const luci::CircleConst *node)
{
  if (node->dtype() == loco::DataType::FLOAT32)
    return false;

  if (not node->quantparam())
    return false;

  return true;
}

// Insert dequantize Op after node
void insert_dequantize(loco::Node *lnode)
{
  auto node = loco::must_cast<luci::CircleNode *>(lnode);
  auto dequant = create_dequantize(node);
  loco::replace(node).with(dequant);
  dequant->input(node);
}

// Insert quantize Op after node and return the quantize Op
luci::CircleQuantize *insert_quantize(loco::Node *lnode)
{
  auto node = loco::must_cast<luci::CircleNode *>(lnode);
  auto quant = create_quantize(node);
  loco::replace(node).with(quant);
  quant->input(node);
  return quant;
}

// Dequantize node
void dequantize(luci::CircleNode *node)
{
  node->dtype(loco::DataType::FLOAT32);
  node->quantparam(nullptr);
}

// Do fake quantization on quantized activation
// 1. Insert Quantize-Dequantize Ops
// 2. Update dtype/quantparam of node
void fq_activation(luci::CircleNode *node)
{
  if (not is_quant_act(node))
    return;

  auto quant = insert_quantize(node);
  insert_dequantize(quant);

  dequantize(node);
}

#define RETURN_UNLESS(COND) \
  if (not(COND))            \
    return;

// Visitor to do fake quantization for each Op
// For non-const activation, insert Quantize-Dequantize after the ofm
// For quantized const, insert Dequantize after the const
struct FakeQuantize final : public luci::CircleNodeMutableVisitor<void>
{
  void visit(luci::CircleNode *node)
  {
    throw std::runtime_error("Unsupported op for fake quantization in " + node->name());
  }

  void visit(luci::CircleInput *node)
  {
    RETURN_UNLESS(is_quant_act(node));

    auto quant = insert_quantize(node);
    insert_dequantize(quant);

    dequantize(node);

    // Update graph input
    const auto inputs = node->graph()->inputs();
    auto graph_input = inputs->at(node->index());
    graph_input->dtype(loco::DataType::FLOAT32);
  }

  void visit(luci::CircleOutput *node)
  {
    RETURN_UNLESS(is_quant_act(node));

    dequantize(node);

    // Update graph output
    const auto outputs = node->graph()->outputs();
    auto graph_output = outputs->at(node->index());
    graph_output->dtype(loco::DataType::FLOAT32);
  }

  // For quantized const, insert Dequantize Op
  void visit(luci::CircleConst *node)
  {
    RETURN_UNLESS(is_quant_const(node));

    insert_dequantize(node);
  }

  // For non-const activation, insert Quantize-Dequantize Ops
  // and dequantize the node
  void visit(luci::CircleAbs *node) { fq_activation(node); }
  void visit(luci::CircleAdd *node) { fq_activation(node); }
  void visit(luci::CircleAveragePool2D *node) { fq_activation(node); }
  void visit(luci::CircleBatchMatMul *node) { fq_activation(node); }
  void visit(luci::CircleConv2D *node) { fq_activation(node); }
  void visit(luci::CircleDepthwiseConv2D *node) { fq_activation(node); }
  void visit(luci::CircleDiv *node) { fq_activation(node); }
  void visit(luci::CircleExp *node) { fq_activation(node); }
  void visit(luci::CircleFullyConnected *node) { fq_activation(node); }
  void visit(luci::CircleGelu *node) { fq_activation(node); }
  void visit(luci::CircleInstanceNorm *node) { fq_activation(node); }
  void visit(luci::CircleLeakyRelu *node) { fq_activation(node); }
  void visit(luci::CircleLogistic *node) { fq_activation(node); }
  void visit(luci::CircleLogSoftmax *node) { fq_activation(node); }
  void visit(luci::CircleMaxPool2D *node) { fq_activation(node); }
  void visit(luci::CircleMaximum *node) { fq_activation(node); }
  void visit(luci::CircleMinimum *node) { fq_activation(node); }
  void visit(luci::CircleMul *node) { fq_activation(node); }
  void visit(luci::CircleNeg *node) { fq_activation(node); }
  void visit(luci::CirclePad *node) { fq_activation(node); }
  void visit(luci::CirclePRelu *node) { fq_activation(node); }
  void visit(luci::CircleMean *node) { fq_activation(node); }
  void visit(luci::CircleReduceProd *node) { fq_activation(node); }
  void visit(luci::CircleReduceMax *node) { fq_activation(node); }
  void visit(luci::CircleRelu *node) { fq_activation(node); }
  void visit(luci::CircleRelu6 *node) { fq_activation(node); }
  void visit(luci::CircleResizeBilinear *node) { fq_activation(node); }
  void visit(luci::CircleResizeNearestNeighbor *node) { fq_activation(node); }
  void visit(luci::CircleRsqrt *node) { fq_activation(node); }
  void visit(luci::CircleSoftmax *node) { fq_activation(node); }
  void visit(luci::CircleSqrt *node) { fq_activation(node); }
  void visit(luci::CircleSquaredDifference *node) { fq_activation(node); }
  void visit(luci::CircleSub *node) { fq_activation(node); }
  void visit(luci::CircleSum *node) { fq_activation(node); }
  void visit(luci::CircleTanh *node) { fq_activation(node); }
  void visit(luci::CircleTransposeConv *node) { fq_activation(node); }

  // For Ops that do not change the value of input, do nothing
  // (dtype will be automatically updated by type inference)
  void visit(luci::CircleCast *) {}
  void visit(luci::CircleConcatenation *) {}
  void visit(luci::CircleDepthToSpace *) {}
  void visit(luci::CircleGather *) {}
  void visit(luci::CircleSlice *) {}
  void visit(luci::CircleStridedSlice *) {}
  void visit(luci::CircleReshape *) {}
  void visit(luci::CircleSpaceToDepth *) {}
  void visit(luci::CircleSplit *) {}
  void visit(luci::CircleSplitOut *) {}
  void visit(luci::CircleSplitV *) {}
  void visit(luci::CircleSplitVOut *) {}
  void visit(luci::CircleTranspose *) {}
  void visit(luci::CirclePack *) {}
  void visit(luci::CircleUnpack *) {}
  void visit(luci::CircleUnpackOut *) {}

  // For Ops that return index, fake quantization is unnecessary
  void visit(luci::CircleArgMax *) {}

  // Virtual node
  void visit(luci::CircleOutputExclude *) {}

  void visit(luci::CircleQuantize *node)
  {
    RETURN_UNLESS(is_quant_act(node));

    insert_dequantize(node);
  }

  // Dequantize Op does nothing in fp32 model
  void visit(luci::CircleDequantize *) {}
};

#undef RETURN_UNLESS

} // namespace

namespace luci
{

bool ConvertToFakeQuantizedModelPass::run(loco::Graph *g)
{
  LOGGER(l);
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    INFO(l) << "ConvertToFakeQuantizedModelPass visit node: " << circle_node->name() << std::endl;

    FakeQuantize fq;
    circle_node->accept(&fq);
  }

  // One time run
  return false;
}

} // namespace luci
