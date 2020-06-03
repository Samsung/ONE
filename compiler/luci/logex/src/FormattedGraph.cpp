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

#include "luci/FormattedGraph.h"

#include <luci/IR/CircleDialect.h>
#include <luci/IR/CircleNodes.h>

#include <pepper/str.h>

#include <cassert>
#include <sstream>
#include <vector>

/**
 * @brief dump std::vector<int64_t> values to stream
 */
std::ostream &operator<<(std::ostream &os, const std::vector<int64_t> &vi64)
{
  for (auto vi : vi64)
  {
    os << vi << " ";
  }
  return os;
}

// For TF lite
namespace
{

const char *to_str(loco::DataType type)
{
  switch (type)
  {
    case loco::DataType::U8:
      return "UINT8";
    case loco::DataType::U16:
      return "UINT16";
    case loco::DataType::U32:
      return "UINT32";
    case loco::DataType::U64:
      return "UINT64";

    case loco::DataType::S8:
      return "INT8";
    case loco::DataType::S16:
      return "INT16";
    case loco::DataType::S32:
      return "INT32";
    case loco::DataType::S64:
      return "INT64";

    case loco::DataType::FLOAT16:
      return "FLOAT16";
    case loco::DataType::FLOAT32:
      return "FLOAT32";
    case loco::DataType::FLOAT64:
      return "FLOAT64";

    case loco::DataType::BOOL:
      return "BOOL";

    default:
      return "Error";
  }
}

const char *to_str(bool value) { return value ? "true" : "false"; }

const char *to_str(luci::FusedActFunc fused)
{
  switch (fused)
  {
    case luci::FusedActFunc::NONE:
      return "NONE";
    case luci::FusedActFunc::RELU:
      return "RELU";
    case luci::FusedActFunc::RELU_N1_TO_1:
      return "RELU_N1_TO_1";
    case luci::FusedActFunc::RELU6:
      return "RELU6";
    default:
      return "Error";
  }
}

const char *to_str(luci::Padding padding)
{
  switch (padding)
  {
    case luci::Padding::SAME:
      return "SAME";
    case luci::Padding::VALID:
      return "VALID";
    default:
      return "Error";
  }
}

const char *to_str(luci::MirrorPadMode mode)
{
  switch (mode)
  {
    case luci::MirrorPadMode::REFLECT:
      return "REFLECT";
    case luci::MirrorPadMode::SYMMETRIC:
      return "SYMMETRIC";
    default:
      return "Error";
  }
}

std::string to_str(const luci::Stride *stride)
{
  return pepper::str(stride->h(), ",", stride->w());
}

std::string to_str(const luci::Filter *filter)
{
  return pepper::str(filter->h(), ",", filter->w());
}

std::string circle_opname(uint32_t opnum)
{
  static const std::string prefix{"circle."};

  switch (static_cast<luci::CircleOpcode>(opnum))
  {
#define CIRCLE_NODE(OPCODE, CLASS) \
  case luci::CircleOpcode::OPCODE: \
    return prefix + #OPCODE;
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_NODE
    default:
      break;
  };

  return prefix + "Invalid";
}

// CircleNodeSummaryBuilder with default implementation
class CircleNodeSummaryBuilderBase : public locop::NodeSummaryBuilder
{
public:
  CircleNodeSummaryBuilderBase(const locop::SymbolTable *tbl) : _tbl{tbl}
  {
    // DO NOTHING
  }

public:
  bool build(const loco::Node *, locop::NodeSummary &s) const final;

protected:
#define CIRCLE_NODE(OPCODE, CLASS)                                      \
  virtual bool summary(const CLASS *, locop::NodeSummary &s) const      \
  {                                                                     \
    s.comments().append("Emitted by Default CircleNodeSummaryBuilder"); \
    s.state(locop::NodeSummary::State::PartiallyKnown);                 \
    return true;                                                        \
  }
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_NODE

protected:
  const locop::SymbolTable *tbl(void) const { return _tbl; }

  // Please do not use _tbl directly and use tbl().
  // This will be changed to private in near future.
protected:
  const locop::SymbolTable *_tbl;
};

class CircleNodeSummaryBuilder final : public CircleNodeSummaryBuilderBase
{
public:
  CircleNodeSummaryBuilder(const locop::SymbolTable *tbl) : CircleNodeSummaryBuilderBase(tbl)
  {
    // DO NOTHING
  }

private:
#define IMPLEMENT(CLASS) bool summary(const CLASS *, locop::NodeSummary &) const final;
  IMPLEMENT(luci::CircleAbs)
  IMPLEMENT(luci::CircleAdd)
  IMPLEMENT(luci::CircleArgMax)
  IMPLEMENT(luci::CircleAveragePool2D)
  IMPLEMENT(luci::CircleBatchMatMul)
  IMPLEMENT(luci::CircleBatchToSpaceND)
  IMPLEMENT(luci::CircleCast)
  IMPLEMENT(luci::CircleConcatenation)
  IMPLEMENT(luci::CircleConst)
  IMPLEMENT(luci::CircleConv2D)
  IMPLEMENT(luci::CircleCos)
  IMPLEMENT(luci::CircleCustom)
  IMPLEMENT(luci::CircleDepthwiseConv2D)
  IMPLEMENT(luci::CircleDiv)
  IMPLEMENT(luci::CircleElu)
  IMPLEMENT(luci::CircleExp)
  IMPLEMENT(luci::CircleExpandDims)
  IMPLEMENT(luci::CircleFill)
  IMPLEMENT(luci::CircleFloor)
  IMPLEMENT(luci::CircleFloorDiv)
  IMPLEMENT(luci::CircleFloorMod)
  IMPLEMENT(luci::CircleFullyConnected)
  IMPLEMENT(luci::CircleGather)
  IMPLEMENT(luci::CircleGatherNd)
  IMPLEMENT(luci::CircleGreater)
  IMPLEMENT(luci::CircleGreaterEqual)
  IMPLEMENT(luci::CircleIf)
  IMPLEMENT(luci::CircleL2Normalize)
  IMPLEMENT(luci::CircleLeakyRelu)
  IMPLEMENT(luci::CircleLess)
  IMPLEMENT(luci::CircleLocalResponseNormalization)
  IMPLEMENT(luci::CircleLog)
  IMPLEMENT(luci::CircleLogicalAnd)
  IMPLEMENT(luci::CircleLogicalNot)
  IMPLEMENT(luci::CircleLogicalOr)
  IMPLEMENT(luci::CircleLogistic)
  IMPLEMENT(luci::CircleMaximum)
  IMPLEMENT(luci::CircleMaxPool2D)
  IMPLEMENT(luci::CircleMean)
  IMPLEMENT(luci::CircleMinimum)
  IMPLEMENT(luci::CircleMirrorPad)
  IMPLEMENT(luci::CircleMul)
  IMPLEMENT(luci::CircleNeg)
  IMPLEMENT(luci::CircleNotEqual)
  IMPLEMENT(luci::CircleOneHot)
  IMPLEMENT(luci::CirclePack)
  IMPLEMENT(luci::CirclePad)
  IMPLEMENT(luci::CirclePow)
  IMPLEMENT(luci::CircleRange)
  IMPLEMENT(luci::CircleReduceAny)
  IMPLEMENT(luci::CircleReduceMax)
  IMPLEMENT(luci::CircleReduceProd)
  IMPLEMENT(luci::CircleRelu)
  IMPLEMENT(luci::CircleRelu6)
  IMPLEMENT(luci::CircleReluN1To1)
  IMPLEMENT(luci::CircleReshape)
  IMPLEMENT(luci::CircleResizeBilinear)
  IMPLEMENT(luci::CircleResizeNearestNeighbor)
  IMPLEMENT(luci::CircleRsqrt)
  IMPLEMENT(luci::CircleSelect)
  IMPLEMENT(luci::CircleShape)
  IMPLEMENT(luci::CircleSin)
  IMPLEMENT(luci::CircleSlice)
  IMPLEMENT(luci::CircleSoftmax)
  IMPLEMENT(luci::CircleSpaceToBatchND)
  IMPLEMENT(luci::CircleSpaceToDepth)
  IMPLEMENT(luci::CircleSplit)
  IMPLEMENT(luci::CircleSplitV)
  IMPLEMENT(luci::CircleSqrt)
  IMPLEMENT(luci::CircleSquare)
  IMPLEMENT(luci::CircleSquaredDifference)
  IMPLEMENT(luci::CircleSqueeze)
  IMPLEMENT(luci::CircleStridedSlice)
  IMPLEMENT(luci::CircleSub)
  IMPLEMENT(luci::CircleSum)
  IMPLEMENT(luci::CircleTanh)
  IMPLEMENT(luci::CircleTile)
  IMPLEMENT(luci::CircleTopKV2)
  IMPLEMENT(luci::CircleTranspose)
  IMPLEMENT(luci::CircleTransposeConv)
  IMPLEMENT(luci::CircleUnpack)
  IMPLEMENT(luci::CircleWhile)
  IMPLEMENT(luci::CircleZerosLike)
  // Circle Only
  IMPLEMENT(luci::CircleBCQFullyConnected)
  IMPLEMENT(luci::CircleBCQGather)
  IMPLEMENT(luci::CircleInstanceNorm)
  // Virtual nodes
  IMPLEMENT(luci::CircleInput)
  IMPLEMENT(luci::CircleOutput)
  IMPLEMENT(luci::CircleIfOut)
  IMPLEMENT(luci::CircleSplitOut)
  IMPLEMENT(luci::CircleSplitVOut)
  IMPLEMENT(luci::CircleTopKV2Out)
  IMPLEMENT(luci::CircleUnpackOut)
  IMPLEMENT(luci::CircleWhileOut)
#undef IMPLEMENT
};

bool CircleNodeSummaryBuilderBase::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (node->dialect() != luci::CircleDialect::get())
    return false;

#define CIRCLE_NODE(OPCODE, CLASS)                        \
  if (dynamic_cast<const CLASS *>(node))                  \
  {                                                       \
    s.opname(circle_opname(node->opnum()));               \
    return summary(dynamic_cast<const CLASS *>(node), s); \
  }
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_NODE

  return false;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleAbs *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleAdd *node, locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.args().append("fused_activation_function", to_str(node->fusedActivationFunction()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleArgMax *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("dimension", tbl()->lookup(node->dimension()));
  s.args().append("output_type", to_str(node->output_type()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleAveragePool2D *node,
                                       locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  s.args().append("value", tbl()->lookup(node->value()));
  s.args().append("filter(h,w)", to_str(node->filter()));
  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleBatchMatMul *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.args().append("adj_x", to_str(node->adj_x()));
  s.args().append("adj_y", to_str(node->adj_y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleBatchToSpaceND *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("block_shape", tbl()->lookup(node->block_shape()));
  s.args().append("crops", tbl()->lookup(node->crops()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleCast *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("in_data_type", to_str(node->in_data_type()));
  s.args().append("out_data_type", to_str(node->out_data_type()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleConcatenation *node,
                                       locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  for (uint32_t i = 0; i < node->numValues(); ++i)
    s.args().append("values", tbl()->lookup(node->values(i)));
  s.args().append("axis", pepper::str(node->axis()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleConst *, locop::NodeSummary &s) const
{
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleConv2D *node, locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);
  assert(node->padding() != luci::Padding::UNDEFINED);

  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("filter", tbl()->lookup(node->filter()));
  s.args().append("bias", tbl()->lookup(node->bias()));

  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleCos *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleCustom *node, locop::NodeSummary &s) const
{
  for (uint32_t i = 0; i < node->numInputs(); i++)
  {
    s.args().append("input" + std::to_string(i), tbl()->lookup(node->inputs(i)));
  }
  s.args().append("custom_code", node->custom_code());
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleDepthwiseConv2D *node,
                                       locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);
  assert(node->padding() != luci::Padding::UNDEFINED);

  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("filter", tbl()->lookup(node->filter()));
  s.args().append("bias", tbl()->lookup(node->bias()));

  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));
  s.args().append("depthMultiplier", std::to_string(node->depthMultiplier()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleDiv *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleElu *node, locop::NodeSummary &s) const
{
  s.args().append("features", tbl()->lookup(node->features()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleExp *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleExpandDims *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("axis", tbl()->lookup(node->axis()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleFloor *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleFloorDiv *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleFloorMod *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleFill *node, locop::NodeSummary &s) const
{
  s.args().append("dims", tbl()->lookup(node->dims()));
  s.args().append("value", tbl()->lookup(node->value()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleFullyConnected *node,
                                       locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("weights", tbl()->lookup(node->weights()));
  s.args().append("bias", tbl()->lookup(node->bias()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleGather *node, locop::NodeSummary &s) const
{
  s.args().append("params", tbl()->lookup(node->params()));
  s.args().append("indices", tbl()->lookup(node->indices()));
  s.args().append("axis", pepper::str(node->axis()));

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleGatherNd *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("params", tbl()->lookup(node->params()));
  s.args().append("indices", tbl()->lookup(node->indices()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleGreater *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleGreaterEqual *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleIf *node, locop::NodeSummary &s) const
{
  s.args().append("cond", tbl()->lookup(node->cond()));
  for (uint32_t i = 0; i < node->input_count(); ++i)
    s.args().append("input", tbl()->lookup(node->input(i)));

  if (node->then_graph() != nullptr)
    s.args().append("then_graph", node->then_graph()->name());
  else
    s.args().append("then_branch", pepper::str(node->then_branch()));

  if (node->else_graph() != nullptr)
    s.args().append("else_graph", node->else_graph()->name());
  else
    s.args().append("else_branch", pepper::str(node->else_branch()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleL2Normalize *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("fused_activation_function", to_str(node->fusedActivationFunction()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleLess *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleLeakyRelu *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("features", tbl()->lookup(node->features()));
  s.args().append("alpha", std::to_string(node->alpha()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleLocalResponseNormalization *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("radius", pepper::str(node->radius()));
  s.args().append("bias", pepper::str(node->bias()));
  s.args().append("alpha", pepper::str(node->alpha()));
  s.args().append("beta", pepper::str(node->beta()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleLog *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleLogicalAnd *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleLogicalNot *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleLogicalOr *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleLogistic *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleMaximum *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleMaxPool2D *node,
                                       locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  s.args().append("value", tbl()->lookup(node->value()));
  s.args().append("filter(h,w)", to_str(node->filter()));
  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleMean *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("reduction_indices", tbl()->lookup(node->reduction_indices()));
  s.args().append("keep_dims", node->keep_dims() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleMinimum *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleMirrorPad *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("paddings", tbl()->lookup(node->paddings()));
  s.args().append("mode", to_str(node->mode()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleMul *node, locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.args().append("fused_activation_function", to_str(node->fusedActivationFunction()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleNeg *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleNotEqual *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleOneHot *node, locop::NodeSummary &s) const
{
  s.args().append("indices", tbl()->lookup(node->indices()));
  s.args().append("depth", tbl()->lookup(node->depth()));
  s.args().append("on_value", tbl()->lookup(node->on_value()));
  s.args().append("off_value", tbl()->lookup(node->off_value()));
  s.args().append("axis", pepper::str(node->axis()));

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CirclePack *node, locop::NodeSummary &s) const
{
  for (uint32_t i = 0; i < node->values_count(); ++i)
    s.args().append("values", tbl()->lookup(node->values(i)));
  s.args().append("values_count", pepper::str(node->values_count()));
  s.args().append("axis", pepper::str(node->axis()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CirclePad *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("paddings", tbl()->lookup(node->paddings()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CirclePow *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleRange *node, locop::NodeSummary &s) const
{
  s.args().append("start", tbl()->lookup(node->start()));
  s.args().append("limit", tbl()->lookup(node->limit()));
  s.args().append("delta", tbl()->lookup(node->delta()));

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleReduceAny *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("reduction_indices", tbl()->lookup(node->reduction_indices()));
  s.args().append("keep_dims", node->keep_dims() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleReduceMax *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("axis", tbl()->lookup(node->axis()));
  s.args().append("keep_dims", node->keep_dims() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleReduceProd *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("reduction_indices", tbl()->lookup(node->reduction_indices()));
  s.args().append("keep_dims", node->keep_dims() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleRelu *node, locop::NodeSummary &s) const
{
  s.args().append("features", tbl()->lookup(node->features()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleRelu6 *node, locop::NodeSummary &s) const
{
  s.args().append("features", tbl()->lookup(node->features()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleReluN1To1 *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("features", tbl()->lookup(node->features()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleReshape *node, locop::NodeSummary &s) const
{
  s.args().append("tensor", tbl()->lookup(node->tensor()));
  s.args().append("shape", tbl()->lookup(node->shape()));
  // TODO Show newShape info
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleResizeBilinear *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("size", tbl()->lookup(node->size()));
  s.args().append("align_corners", node->align_corners() ? "true" : "false");
  s.args().append("half_pixel_centers", node->half_pixel_centers() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleResizeNearestNeighbor *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("size", tbl()->lookup(node->size()));
  s.args().append("align_corners", node->align_corners() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleRsqrt *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSelect *node, locop::NodeSummary &s) const
{
  s.args().append("condition", tbl()->lookup(node->condition()));
  s.args().append("t", tbl()->lookup(node->t()));
  s.args().append("e", tbl()->lookup(node->e()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleShape *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("out_type", to_str(node->out_type()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSin *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSlice *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("begin", tbl()->lookup(node->begin()));
  s.args().append("size", tbl()->lookup(node->size()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSoftmax *node, locop::NodeSummary &s) const
{
  s.args().append("logits", tbl()->lookup(node->logits()));
  s.args().append("beta", pepper::str(node->beta()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSpaceToBatchND *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("block_shape", tbl()->lookup(node->block_shape()));
  s.args().append("paddings", tbl()->lookup(node->paddings()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSpaceToDepth *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("block_size", pepper::str(node->block_size()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSplit *node, locop::NodeSummary &s) const
{
  s.args().append("split_dim", tbl()->lookup(node->split_dim()));
  s.args().append("input", tbl()->lookup(node->input()));

  s.args().append("num_split", pepper::str(node->num_split()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSplitV *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("size_splits", tbl()->lookup(node->size_splits()));
  s.args().append("split_dim", tbl()->lookup(node->split_dim()));

  s.args().append("num_split", pepper::str(node->num_split()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSqrt *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSquare *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSquaredDifference *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSqueeze *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));

  std::stringstream ss{"("};
  for (size_t i = 0; i < node->squeeze_dims().size(); ++i)
  {
    if (i != 0)
      ss << ", ";
    ss << node->squeeze_dims()[i];
  }
  ss << ")";

  s.args().append("squeeze_dims", ss.str());
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleStridedSlice *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("begin", tbl()->lookup(node->begin()));
  s.args().append("end", tbl()->lookup(node->end()));
  s.args().append("strides", tbl()->lookup(node->strides()));

  s.args().append("begin_mask", pepper::str(node->begin_mask()));
  s.args().append("end_mask", pepper::str(node->end_mask()));
  s.args().append("ellipsis_mask", pepper::str(node->ellipsis_mask()));
  s.args().append("new_axis_mask", pepper::str(node->new_axis_mask()));
  s.args().append("shrink_axis_mask", pepper::str(node->shrink_axis_mask()));

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSub *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSum *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("reduction_indices", tbl()->lookup(node->reduction_indices()));
  s.args().append("keep_dims", node->keep_dims() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleTanh *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleTile *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("multiples", tbl()->lookup(node->multiples()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleTopKV2 *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("k", tbl()->lookup(node->k()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleTranspose *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("a", tbl()->lookup(node->a()));
  s.args().append("perm", tbl()->lookup(node->perm()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleTransposeConv *node,
                                       locop::NodeSummary &s) const
{
  assert(node->padding() != luci::Padding::UNDEFINED);

  s.args().append("inputSizes", tbl()->lookup(node->inputSizes()));
  s.args().append("filter", tbl()->lookup(node->filter()));
  s.args().append("outBackprop", tbl()->lookup(node->outBackprop()));

  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleUnpack *node, locop::NodeSummary &s) const
{
  s.args().append("value", tbl()->lookup(node->value()));

  s.args().append("num", pepper::str(node->num()));
  s.args().append("axis", pepper::str(node->axis()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleWhile *node, locop::NodeSummary &s) const
{
  for (uint32_t i = 0; i < node->input_count(); ++i)
    s.args().append("input", tbl()->lookup(node->input(i)));

  if (node->cond_graph() != nullptr)
    s.args().append("cond_graph", node->cond_graph()->name());
  else
    s.args().append("cond_branch", pepper::str(node->cond_branch()));

  if (node->body_graph() != nullptr)
    s.args().append("body_graph", node->body_graph()->name());
  else
    s.args().append("body_branch", pepper::str(node->body_branch()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleZerosLike *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSplitOut *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleSplitVOut *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleTopKV2Out *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("topkv2", tbl()->lookup(node->input()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleUnpackOut *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("unpack", tbl()->lookup(node->input()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleIfOut *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleWhileOut *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("while", tbl()->lookup(node->input()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleInput *, locop::NodeSummary &s) const
{
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleOutput *node, locop::NodeSummary &s) const
{
  s.args().append("from", tbl()->lookup(node->from()));

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleBCQFullyConnected *node,
                                       locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("weights_scales", tbl()->lookup(node->weights_scales()));
  s.args().append("weights_binary", tbl()->lookup(node->weights_binary()));
  s.args().append("bias", tbl()->lookup(node->bias()));

  s.args().append("fused", to_str(node->fusedActivationFunction()));
  s.args().append("weights_hidden_size", pepper::str(node->weights_hidden_size()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleBCQGather *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input_scales", tbl()->lookup(node->input_scales()));
  s.args().append("input_binary", tbl()->lookup(node->input_binary()));
  s.args().append("indices", tbl()->lookup(node->indices()));

  s.args().append("axis", pepper::str(node->axis()));
  s.args().append("input_hidden_size", pepper::str(node->input_hidden_size()));

  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleInstanceNorm *node,
                                       locop::NodeSummary &s) const
{
  auto fused = node->fusedActivationFunction();
  assert(fused != luci::FusedActFunc::UNDEFINED);

  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("gamma", tbl()->lookup(node->gamma()));
  s.args().append("beta", tbl()->lookup(node->beta()));
  s.args().append("epsilon", pepper::str(node->epsilon()));
  s.args().append("fused_activation_function", to_str(fused));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

} // namespace

namespace luci
{

bool NodeSummaryBuilder::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (locop::CanonicalNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  if (CircleNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  return false;
}

} // namespace luci
