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

#include "CircleNodeSummaryBuilder.h"
#include "luci/FormattedGraph.h"

#include <luci/IR/CircleDialect.h>
#include <luci/IR/CircleNodes.h>

#include <pepper/str.h>

#include <cassert>
#include <sstream>
#include <vector>

using namespace luci;
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
    case luci::FusedActFunc::TANH:
      return "TANH";
    case luci::FusedActFunc::SIGN_BIT:
      return "SIGN_BIT";
    default:
      return "Error";
  }
}

std::string circle_opname(uint32_t opnum)
{
  static const std::string prefix{"circle."};

  switch (static_cast<luci::CircleOpcode>(opnum))
  {
#define CIRCLE_NODE(OPCODE, CLASS) \
  case luci::CircleOpcode::OPCODE: \
    return prefix + #OPCODE;
#define CIRCLE_VNODE CIRCLE_NODE
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_VNODE
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
#define CIRCLE_NODE(OPCODE, CLASS) \
  virtual bool summary(const CLASS *, locop::NodeSummary &) const { return false; }
#define CIRCLE_VNODE CIRCLE_NODE
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_VNODE
#undef CIRCLE_NODE

protected:
  const locop::SymbolTable *tbl(void) const { return _tbl; }

private:
  const locop::SymbolTable *_tbl;
};

template <class CIRCLENODE>
bool use_x(const locop::SymbolTable *tbl, const CIRCLENODE *node, locop::NodeSummary &s)
{
  s.args().append("x", tbl->lookup(node->x()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

template <class CIRCLENODE>
bool use_input(const locop::SymbolTable *tbl, const CIRCLENODE *node, locop::NodeSummary &s)
{
  s.args().append("input", tbl->lookup(node->input()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

template <class CIRCLENODE>
bool use_features(const locop::SymbolTable *tbl, const CIRCLENODE *node, locop::NodeSummary &s)
{
  s.args().append("features", tbl->lookup(node->features()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

template <class CIRCLENODE>
bool use_xy(const locop::SymbolTable *tbl, const CIRCLENODE *node, locop::NodeSummary &s)
{
  s.args().append("x", tbl->lookup(node->x()));
  s.args().append("y", tbl->lookup(node->y()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

template <class CIRCLENODE>
bool use_xy_act(const locop::SymbolTable *tbl, const CIRCLENODE *node, locop::NodeSummary &s)
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);

  s.args().append("x", tbl->lookup(node->x()));
  s.args().append("y", tbl->lookup(node->y()));
  s.args().append("fused_activation_function", to_str(node->fusedActivationFunction()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

template <class CIRCLENODE>
bool use_reducer(const locop::SymbolTable *tbl, const CIRCLENODE *node, locop::NodeSummary &s)
{
  s.args().append("input", tbl->lookup(node->input()));
  s.args().append("reduction_indices", tbl->lookup(node->reduction_indices()));
  s.args().append("keep_dims", node->keep_dims() ? "true" : "false");
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

template <class CIRCLENODE>
bool use_ido(const locop::SymbolTable *tbl, const CIRCLENODE *node, locop::NodeSummary &s)
{
  s.args().append("input", tbl->lookup(node->input()));
  s.args().append("dimension", tbl->lookup(node->dimension()));
  s.args().append("output_type", to_str(node->output_type()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleWhere *node,
                  locop::NodeSummary &s)
{
  s.args().append("condition", tbl->lookup(node->condition()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleWhile *node,
                  locop::NodeSummary &s)
{
  for (uint32_t i = 0; i < node->input_count(); ++i)
    s.args().append("input", tbl->lookup(node->input(i)));

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

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleTopKV2Out *node,
                  locop::NodeSummary &s)
{
  s.args().append("topkv2", tbl->lookup(node->input()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleUniqueOut *node,
                  locop::NodeSummary &s)
{
  s.args().append("unique", tbl->lookup(node->input()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleUnpackOut *node,
                  locop::NodeSummary &s)
{
  s.args().append("unpack", tbl->lookup(node->input()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *, const luci::CircleVariable *, locop::NodeSummary &s)
{
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleWhileOut *node,
                  locop::NodeSummary &s)
{
  s.args().append("while", tbl->lookup(node->input()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleOutput *node,
                  locop::NodeSummary &s)
{
  s.args().append("from", tbl->lookup(node->from()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *, const luci::CircleOutputDummy *,
                  locop::NodeSummary &s)
{
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *, const luci::CircleOutputExclude *,
                  locop::NodeSummary &s)
{
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleBCQFullyConnected *node,
                  locop::NodeSummary &s)
{
  assert(node->fusedActivationFunction() != luci::FusedActFunc::UNDEFINED);
  s.args().append("input", tbl->lookup(node->input()));
  s.args().append("weights_scales", tbl->lookup(node->weights_scales()));
  s.args().append("weights_binary", tbl->lookup(node->weights_binary()));
  s.args().append("bias", tbl->lookup(node->bias()));
  s.args().append("weights_clusters", tbl->lookup(node->weights_clusters()));
  s.args().append("fused", to_str(node->fusedActivationFunction()));
  s.args().append("weights_hidden_size", pepper::str(node->weights_hidden_size()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleBCQGather *node,
                  locop::NodeSummary &s)
{
  s.args().append("input_scales", tbl->lookup(node->input_scales()));
  s.args().append("input_binary", tbl->lookup(node->input_binary()));
  s.args().append("indices", tbl->lookup(node->indices()));
  s.args().append("input_clusters", tbl->lookup(node->input_clusters()));
  s.args().append("axis", pepper::str(node->axis()));
  s.args().append("input_hidden_size", pepper::str(node->input_hidden_size()));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool summary_node(const locop::SymbolTable *tbl, const luci::CircleInstanceNorm *node,
                  locop::NodeSummary &s)
{
  auto fused = node->fusedActivationFunction();
  assert(fused != luci::FusedActFunc::UNDEFINED);

  s.args().append("input", tbl->lookup(node->input()));
  s.args().append("gamma", tbl->lookup(node->gamma()));
  s.args().append("beta", tbl->lookup(node->beta()));
  s.args().append("epsilon", pepper::str(node->epsilon()));
  s.args().append("fused_activation_function", to_str(fused));
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

// SummaryBuilderLet type
enum class SB
{
  WXYZ,
  CIRC, // circle only
  VIRT, // virtual
};

template <SB sb> class SummaryBuilderLet;

#define IMPLEMENT(CLASS) bool summary(const CLASS *, locop::NodeSummary &) const final;

template <> class SummaryBuilderLet<SB::WXYZ> final : public CircleNodeSummaryBuilderBase
{
public:
  SummaryBuilderLet(const locop::SymbolTable *tbl) : CircleNodeSummaryBuilderBase(tbl)
  {
    // DO NOTHING
  }

private:
  IMPLEMENT(luci::CircleWhere)
  IMPLEMENT(luci::CircleWhile)
  IMPLEMENT(luci::CircleZerosLike)
};

template <> class SummaryBuilderLet<SB::CIRC> final : public CircleNodeSummaryBuilderBase
{
public:
  SummaryBuilderLet(const locop::SymbolTable *tbl) : CircleNodeSummaryBuilderBase(tbl)
  {
    // DO NOTHING
  }

private:
  IMPLEMENT(luci::CircleBCQFullyConnected)
  IMPLEMENT(luci::CircleBCQGather)
  IMPLEMENT(luci::CircleInstanceNorm)
};

template <> class SummaryBuilderLet<SB::VIRT> final : public CircleNodeSummaryBuilderBase
{
public:
  SummaryBuilderLet(const locop::SymbolTable *tbl) : CircleNodeSummaryBuilderBase(tbl)
  {
    // DO NOTHING
  }

private:
  IMPLEMENT(luci::CircleInput)
  IMPLEMENT(luci::CircleOutput)
  IMPLEMENT(luci::CircleCustomOut)
  IMPLEMENT(luci::CircleIfOut)
  IMPLEMENT(luci::CircleNonMaxSuppressionV4Out)
  IMPLEMENT(luci::CircleNonMaxSuppressionV5Out)
  IMPLEMENT(luci::CircleOutputDummy)
  IMPLEMENT(luci::CircleOutputExclude)
  IMPLEMENT(luci::CircleSplitOut)
  IMPLEMENT(luci::CircleSplitVOut)
  IMPLEMENT(luci::CircleTopKV2Out)
  IMPLEMENT(luci::CircleUniqueOut)
  IMPLEMENT(luci::CircleUnpackOut)
  IMPLEMENT(luci::CircleVariable)
  IMPLEMENT(luci::CircleWhileOut)
};

#undef IMPLEMENT

bool CircleNodeSummaryBuilderBase::build(const loco::Node *node, locop::NodeSummary &s) const
{
  if (node->dialect() != luci::CircleDialect::get())
    return false;

  auto ptr_to_str = [](const void *ptr) {
    std::stringstream ss;
    ss << ptr;
    return ss.str();
  };

  auto add_comment = [&]() {
    auto cnode = loco::must_cast<const luci::CircleNode *>(node);
    s.opname(circle_opname(node->opnum()));
    s.comments().append("[" + cnode->name() + "] = " + ptr_to_str(node));
  };

#define CIRCLE_NODE(OPCODE, CLASS)                     \
  if (dynamic_cast<const CLASS *>(node))               \
  {                                                    \
    if (summary(dynamic_cast<const CLASS *>(node), s)) \
    {                                                  \
      add_comment();                                   \
      return true;                                     \
    }                                                  \
  }
#define CIRCLE_VNODE CIRCLE_NODE
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_VNODE
#undef CIRCLE_NODE

  return false;
}

bool SummaryBuilderLet<SB::WXYZ>::summary(const luci::CircleWhere *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::WXYZ>::summary(const luci::CircleWhile *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::WXYZ>::summary(const luci::CircleZerosLike *node,
                                          locop::NodeSummary &s) const
{
  return use_input(tbl(), node, s);
}

bool SummaryBuilderLet<SB::CIRC>::summary(const luci::CircleBCQFullyConnected *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::CIRC>::summary(const luci::CircleBCQGather *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::CIRC>::summary(const luci::CircleInstanceNorm *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleInput *, locop::NodeSummary &s) const
{
  s.state(locop::NodeSummary::State::Complete);
  return true;
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleOutput *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleCustomOut *node,
                                          locop::NodeSummary &s) const
{
  return use_input(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleIfOut *node,
                                          locop::NodeSummary &s) const
{
  return use_input(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleNonMaxSuppressionV4Out *node,
                                          locop::NodeSummary &s) const
{
  return use_input(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleNonMaxSuppressionV5Out *node,
                                          locop::NodeSummary &s) const
{
  return use_input(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleOutputDummy *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleOutputExclude *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleSplitOut *node,
                                          locop::NodeSummary &s) const
{
  return use_input(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleSplitVOut *node,
                                          locop::NodeSummary &s) const
{
  return use_input(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleTopKV2Out *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleUniqueOut *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleUnpackOut *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleVariable *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
}

bool SummaryBuilderLet<SB::VIRT>::summary(const luci::CircleWhileOut *node,
                                          locop::NodeSummary &s) const
{
  return summary_node(tbl(), node, s);
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

#define BUILD_GRP(GRP)                                   \
  do                                                     \
  {                                                      \
    if (SummaryBuilderLet<SB::GRP>(_tbl).build(node, s)) \
      return true;                                       \
  } while (false)

  // TODO Replace with CircleNodeSummaryBuilder and then remove these
  BUILD_GRP(WXYZ);
  BUILD_GRP(CIRC);
  BUILD_GRP(VIRT);

  if (CircleNodeSummaryBuilder().build(node, _tbl, s))
  {
    return true;
  }

  return false;
}

} // namespace luci
