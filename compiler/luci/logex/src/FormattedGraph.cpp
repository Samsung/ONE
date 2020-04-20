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
  IMPLEMENT(luci::CircleBatchToSpaceND)
  IMPLEMENT(luci::CircleConcatenation)
  IMPLEMENT(luci::CircleConst)
  IMPLEMENT(luci::CircleConv2D)
  IMPLEMENT(luci::CircleCos)
  IMPLEMENT(luci::CircleDepthwiseConv2D)
  IMPLEMENT(luci::CircleDiv)
  IMPLEMENT(luci::CircleExp)
  IMPLEMENT(luci::CircleFullyConnected)
  IMPLEMENT(luci::CircleLogicalNot)
  IMPLEMENT(luci::CircleLogicalOr)
  IMPLEMENT(luci::CircleMaximum)
  IMPLEMENT(luci::CircleMaxPool2D)
  IMPLEMENT(luci::CircleMean)
  IMPLEMENT(luci::CircleMul)
  IMPLEMENT(luci::CirclePack)
  IMPLEMENT(luci::CirclePad)
  IMPLEMENT(luci::CircleRelu)
  IMPLEMENT(luci::CircleRelu6)
  IMPLEMENT(luci::CircleReshape)
  IMPLEMENT(luci::CircleRsqrt)
  IMPLEMENT(luci::CircleSoftmax)
  IMPLEMENT(luci::CircleSqrt)
  IMPLEMENT(luci::CircleSquaredDifference)
  IMPLEMENT(luci::CircleSub)
  IMPLEMENT(luci::CircleTanh)
  IMPLEMENT(luci::CircleTranspose)
  IMPLEMENT(luci::CircleTransposeConv)
  IMPLEMENT(luci::CircleUnpack)
  // Circle Only
  IMPLEMENT(luci::CircleInstanceNorm)
  // Virtual nodes
  IMPLEMENT(luci::CircleInput)
  IMPLEMENT(luci::CircleOutput)
  IMPLEMENT(luci::CircleUnpackOut)
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

bool CircleNodeSummaryBuilder::summary(const luci::CircleBatchToSpaceND *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("block_shape", tbl()->lookup(node->block_shape()));
  s.args().append("crops", tbl()->lookup(node->crops()));

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

bool CircleNodeSummaryBuilder::summary(const luci::CircleExp *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
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

bool CircleNodeSummaryBuilder::summary(const luci::CircleMul *node, locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.args().append("fused_activation_function", to_str(node->fusedActivationFunction()));
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

bool CircleNodeSummaryBuilder::summary(const luci::CircleReshape *node, locop::NodeSummary &s) const
{
  s.args().append("tensor", tbl()->lookup(node->tensor()));
  s.args().append("shape", tbl()->lookup(node->shape()));
  // TODO Show newShape info
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleRsqrt *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
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

bool CircleNodeSummaryBuilder::summary(const luci::CircleSqrt *node, locop::NodeSummary &s) const
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

bool CircleNodeSummaryBuilder::summary(const luci::CircleSub *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool CircleNodeSummaryBuilder::summary(const luci::CircleTanh *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
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

bool CircleNodeSummaryBuilder::summary(const luci::CircleUnpackOut *node,
                                       locop::NodeSummary &s) const
{
  s.args().append("unpack", tbl()->lookup(node->unpack()));

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

} // namespace exo
