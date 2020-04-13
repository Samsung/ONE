/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExoFormattedGraph.h"

#include "Dialect/IR/TFLDialect.h"
#include "Dialect/IR/TFLNodes.h"

#include "Dialect/IR/CircleDialect.h"
#include "Dialect/IR/CircleNodes.h"

#include <locoex/Service/COpFormattedGraph.h>
#include <pepper/str.h>

#include <sstream>
#include <cassert>

// For TF lite
namespace
{

const char *to_str(locoex::FusedActFunc fused)
{
  switch (fused)
  {
    case locoex::FusedActFunc::NONE:
      return "NONE";
    case locoex::FusedActFunc::RELU:
      return "RELU";
    case locoex::FusedActFunc::RELU6:
      return "RELU6";
    default:
      return "Error";
  }
}

const char *to_str(locoex::Padding padding)
{
  switch (padding)
  {
    case locoex::Padding::SAME:
      return "SAME";
    case locoex::Padding::VALID:
      return "VALID";
    default:
      return "Error";
  }
}

std::string to_str(const locoex::Stride *stride)
{
  return pepper::str(stride->h(), ",", stride->w());
}

std::string to_str(const locoex::Filter *filter)
{
  return pepper::str(filter->h(), ",", filter->w());
}

std::string tfl_opname(uint32_t opnum)
{
  static std::string prefix{"tfl."};

  switch (static_cast<locoex::TFLOpcode>(opnum))
  {
#define TFL_NODE(OPCODE, CLASS)   \
  case locoex::TFLOpcode::OPCODE: \
    return prefix + #OPCODE;
#include "Dialect/IR/TFLNodes.lst"
#undef TFL_NODE
    default:
      break;
  };

  return prefix + "Invalid";
}

// TFLNodeSummaryBuilder with default implementation
class TFLNodeSummaryBuilderBase : public locop::NodeSummaryBuilder
{
public:
  TFLNodeSummaryBuilderBase(const locop::SymbolTable *tbl) : _tbl{tbl}
  {
    // DO NOTHING
  }

public:
  bool build(const loco::Node *, locop::NodeSummary &s) const final;

protected:
#define TFL_NODE(OPCODE, CLASS)                                      \
  virtual bool summary(const CLASS *, locop::NodeSummary &s) const   \
  {                                                                  \
    s.comments().append("Emitted by Default TFLNodeSummaryBuilder"); \
    s.state(locop::NodeSummary::State::PartiallyKnown);              \
    return true;                                                     \
  }
#include "Dialect/IR/TFLNodes.lst"
#undef TFL_NODE

protected:
  const locop::SymbolTable *tbl(void) const { return _tbl; }

  // Please do not use _tbl directly and use tbl().
  // This will be changed to private in near future.
protected:
  const locop::SymbolTable *_tbl;
};

class TFLNodeSummaryBuilder final : public TFLNodeSummaryBuilderBase
{
public:
  TFLNodeSummaryBuilder(const locop::SymbolTable *tbl) : TFLNodeSummaryBuilderBase(tbl)
  {
    // DO NOTHING
  }

private:
#define IMPLEMENT(CLASS) bool summary(const CLASS *, locop::NodeSummary &) const final;
  IMPLEMENT(locoex::TFLAdd)
  IMPLEMENT(locoex::TFLAveragePool2D)
  IMPLEMENT(locoex::TFLConcatenation)
  IMPLEMENT(locoex::TFLConst)
  IMPLEMENT(locoex::TFLConv2D)
  IMPLEMENT(locoex::TFLDepthwiseConv2D)
  IMPLEMENT(locoex::TFLDiv)
  IMPLEMENT(locoex::TFLMaximum)
  IMPLEMENT(locoex::TFLMaxPool2D)
  IMPLEMENT(locoex::TFLMean)
  IMPLEMENT(locoex::TFLMul)
  IMPLEMENT(locoex::TFLRelu)
  IMPLEMENT(locoex::TFLRelu6)
  IMPLEMENT(locoex::TFLReshape)
  IMPLEMENT(locoex::TFLRsqrt)
  IMPLEMENT(locoex::TFLSqrt)
  IMPLEMENT(locoex::TFLSquaredDifference)
  IMPLEMENT(locoex::TFLSub)
  IMPLEMENT(locoex::TFLTranspose)
  IMPLEMENT(locoex::TFLTransposeConv)
#undef IMPLEMENT
};

bool TFLNodeSummaryBuilderBase::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (node->dialect() != locoex::TFLDialect::get())
    return false;

#define TFL_NODE(OPCODE, CLASS)                           \
  if (dynamic_cast<const CLASS *>(node))                  \
  {                                                       \
    s.opname(tfl_opname(node->opnum()));                  \
    return summary(dynamic_cast<const CLASS *>(node), s); \
  }
#include "Dialect/IR/TFLNodes.lst"
#undef TFL_NODE

  return false;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLAdd *node, locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != locoex::FusedActFunc::UNDEFINED);

  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.args().append("fused_activation_function", to_str(node->fusedActivationFunction()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLAveragePool2D *node,
                                    locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != locoex::FusedActFunc::UNDEFINED);

  s.args().append("value", tbl()->lookup(node->value()));
  s.args().append("filter(h,w)", to_str(node->filter()));
  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLConcatenation *node,
                                    locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != locoex::FusedActFunc::UNDEFINED);

  for (uint32_t i = 0; i < node->numValues(); ++i)
    s.args().append("values", tbl()->lookup(node->values(i)));
  s.args().append("axis", pepper::str(node->axis()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLConst *, locop::NodeSummary &s) const
{
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLConv2D *node, locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != locoex::FusedActFunc::UNDEFINED);
  assert(node->padding() != locoex::Padding::UNDEFINED);

  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("filter", tbl()->lookup(node->filter()));
  s.args().append("bias", tbl()->lookup(node->bias()));

  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLDepthwiseConv2D *node,
                                    locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != locoex::FusedActFunc::UNDEFINED);
  assert(node->padding() != locoex::Padding::UNDEFINED);

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

bool TFLNodeSummaryBuilder::summary(const locoex::TFLDiv *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLMaximum *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLMaxPool2D *node, locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != locoex::FusedActFunc::UNDEFINED);

  s.args().append("value", tbl()->lookup(node->value()));
  s.args().append("filter(h,w)", to_str(node->filter()));
  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLMean *node, locop::NodeSummary &s) const
{
  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("reduction_indices", tbl()->lookup(node->reduction_indices()));
  s.args().append("keep_dims", node->keep_dims() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLMul *node, locop::NodeSummary &s) const
{
  assert(node->fusedActivationFunction() != locoex::FusedActFunc::UNDEFINED);

  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.args().append("fused_activation_function", to_str(node->fusedActivationFunction()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLRelu *node, locop::NodeSummary &s) const
{
  s.args().append("features", tbl()->lookup(node->features()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLRelu6 *node, locop::NodeSummary &s) const
{
  s.args().append("features", tbl()->lookup(node->features()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLReshape *node, locop::NodeSummary &s) const
{
  s.args().append("tensor", tbl()->lookup(node->tensor()));
  s.args().append("shape", tbl()->lookup(node->shape()));
  // TODO Show newShape info
  s.state(locop::NodeSummary::State::PartiallyKnown);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLRsqrt *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

// TODO TFLSoftmax

bool TFLNodeSummaryBuilder::summary(const locoex::TFLSqrt *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLSquaredDifference *node,
                                    locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLSub *node, locop::NodeSummary &s) const
{
  s.args().append("x", tbl()->lookup(node->x()));
  s.args().append("y", tbl()->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

// TODO TFLTanh

bool TFLNodeSummaryBuilder::summary(const locoex::TFLTranspose *node, locop::NodeSummary &s) const
{
  s.args().append("a", tbl()->lookup(node->a()));
  s.args().append("perm", tbl()->lookup(node->perm()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool TFLNodeSummaryBuilder::summary(const locoex::TFLTransposeConv *node,
                                    locop::NodeSummary &s) const
{
  assert(node->padding() != locoex::Padding::UNDEFINED);

  s.args().append("inputSizes", tbl()->lookup(node->inputSizes()));
  s.args().append("filter", tbl()->lookup(node->filter()));
  s.args().append("outBackprop", tbl()->lookup(node->outBackprop()));

  s.args().append("stride(h,w)", to_str(node->stride()));
  s.args().append("padding", to_str(node->padding()));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

} // namespace

// For Circle
namespace
{

std::string circle_opname(uint32_t opnum)
{
  static std::string prefix{"circle."};

  switch (static_cast<locoex::CircleOpcode>(opnum))
  {
#define CIRCLE_NODE(OPCODE, CLASS)   \
  case locoex::CircleOpcode::OPCODE: \
    return prefix + #OPCODE;
#include "Dialect/IR/CircleNodes.lst"
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
#include "Dialect/IR/CircleNodes.lst"
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
  IMPLEMENT(locoex::CircleInstanceNorm)
#undef IMPLEMENT
};

bool CircleNodeSummaryBuilderBase::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (node->dialect() != locoex::CircleDialect::get())
    return false;

#define CIRCLE_NODE(OPCODE, CLASS)                        \
  if (dynamic_cast<const CLASS *>(node))                  \
  {                                                       \
    s.opname(circle_opname(node->opnum()));               \
    return summary(dynamic_cast<const CLASS *>(node), s); \
  }
#include "Dialect/IR/CircleNodes.lst"
#undef CIRCLE_NODE

  return false;
}

bool CircleNodeSummaryBuilder::summary(const locoex::CircleInstanceNorm *node,
                                       locop::NodeSummary &s) const
{
  auto fused = node->fusedActivationFunction();
  assert(fused != locoex::FusedActFunc::UNDEFINED);

  s.args().append("input", tbl()->lookup(node->input()));
  s.args().append("gamma", tbl()->lookup(node->gamma()));
  s.args().append("beta", tbl()->lookup(node->beta()));
  s.args().append("epsilon", pepper::str(node->epsilon()));
  s.args().append("fused_activation_function", to_str(fused));

  s.state(locop::NodeSummary::State::Complete);

  return true;
}

} // namespace

namespace exo
{

bool NodeSummaryBuilder::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (locop::CanonicalNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  if (TFLNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  if (CircleNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  if (locoex::COpNodeSummaryBuilder(_tbl).build(node, s))
  {
    return true;
  }

  return false;
}

} // namespace exo
