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

#include "locop/CanonicalNodeSummaryBuilder.h"

#include "locop/FormattedTensorShape.h"

#include <loco/IR/CanonicalOpcode.h>
#include <loco/IR/CanonicalDialect.h>
#include <loco/IR/CanonicalNodeVisitor.h>
#include <loco/IR/CanonicalNodeImpl.h>

#include <pp/Format.h>

#include <map>
#include <set>

#include <cassert>

using locop::SymbolTable;

namespace
{

// TODO Move this into loco
loco::TensorShape tensor_shape(const loco::NodeMixin<loco::NodeTrait::TensorShape> *m)
{
  loco::TensorShape res;

  res.rank(m->rank());

  for (uint32_t axis = 0; axis < m->rank(); ++axis)
  {
    res.dim(axis) = m->dim(axis);
  }

  return res;
}

using PrettyTensorShape = locop::FormattedTensorShape<locop::TensorShapeFormat::Bracket>;

inline PrettyTensorShape pretty(const loco::TensorShape &shape)
{
  return PrettyTensorShape{&shape};
}

} // namespace

namespace
{

/**
 * @brief Return the opname as "<dialect>.<op>"
 */
std::string opname(const loco::Node *node)
{
  if (node->dialect() == loco::CanonicalDialect::get())
  {
    auto canonical_node = loco::must_cast<const loco::CanonicalNode *>(node);

    switch (canonical_node->opcode())
    {
#define CANONICAL_NODE(OPCODE, CLASS) \
  case loco::CanonicalOpcode::OPCODE: \
    return "canonical." #OPCODE;
#include "loco/IR/CanonicalNodes.lst"
#undef CANONICAL_NODE
      default:
        break;
    };

    return "canonical."
           "Invalid";
  }

  return "unknown."
         "Unknown";
}

struct NodeDesc : public locop::NodeDesc
{
public:
  NodeDesc() = default;
  NodeDesc(const locop::OpName &opname) : locop::NodeDesc{opname}
  {
    // DO NOTHING
  }

public:
  // DEPRECATED
  const locop::OpName &name(void) const { return opname(); }

  // DEPRECATED
  uint32_t arg_size(void) const { return args().count(); }
  // DEPRECATED
  const locop::ArgElem &arg(uint32_t n) const { return args().at(n); }
  // DEPRECATED
  void arg(const locop::ArgName &name, const locop::ArgValue &value) { args().append(name, value); }
};

NodeDesc default_node_desc(const SymbolTable &tbl, const loco::Node *node)
{
  NodeDesc res{opname(node)};

  for (uint32_t n = 0; n < node->arity(); ++n)
  {
    res.arg(std::string{"arg"} + std::to_string(n), tbl.lookup(node->arg(n)));
  }
  res.state(NodeDesc::State::PartiallyKnown);

  return res;
}

class CanonicalNodeDescBuilder final : public loco::CanonicalNodeVisitor<NodeDesc>
{
public:
  CanonicalNodeDescBuilder(const SymbolTable *symtbl) : _symtbl{symtbl}
  {
    // DO NOTHING
  }

private:
  std::string nodename(const loco::Node *node) const { return _symtbl->lookup(node); }

public:
  // TODO Build a node description for each canonical node
  NodeDesc visit(const loco::Push *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("index", node->indexed() ? pp::fmt(node->index()) : pp::fmt('?'));
    res.arg("from", nodename(node->from()));
    res.state(NodeDesc::State::Complete);

    return res;
  }

  NodeDesc visit(const loco::Pull *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("index", node->indexed() ? pp::fmt(node->index()) : pp::fmt('?'));
    res.state(NodeDesc::State::Complete);

    return res;
  }

  NodeDesc visit(const loco::Forward *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("input", nodename(node->input()));
    res.state(NodeDesc::State::Complete);

    return res;
  }

  NodeDesc visit(const loco::ConstGen *node) final
  {
    NodeDesc res{opname(node)};

    // TODO Print data type
    res.arg("shape", pp::fmt(pretty(tensor_shape(node))));
    res.state(NodeDesc::State::PartiallyKnown);

    return res;
  }

  NodeDesc visit(const loco::TensorConcat *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("lhs", nodename(node->lhs()));
    res.arg("rhs", nodename(node->rhs()));
    res.arg("axis", pp::fmt(node->axis()));
    res.state(NodeDesc::State::Complete);

    return res;
  }

  NodeDesc visit(const loco::EltwiseAdd *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("lhs", nodename(node->lhs()));
    res.arg("rhs", nodename(node->rhs()));
    res.state(NodeDesc::State::Complete);

    return res;
  }

  NodeDesc visit(const loco::EltwiseMul *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("lhs", nodename(node->lhs()));
    res.arg("rhs", nodename(node->rhs()));
    res.state(NodeDesc::State::Complete);

    return res;
  }

  NodeDesc visit(const loco::TensorReduce *node) final
  {
    NodeDesc res{opname(node)};

    // TODO Print TensorAxisSet
    res.arg("input", nodename(node->input()));
    res.arg("func", pp::fmt((int32_t)node->func()));

    res.state(NodeDesc::State::PartiallyKnown);

    return res;
  }

  NodeDesc visit(const loco::Reshape<loco::ReshapeType::Fixed> *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("input", nodename(node->input()));
    res.arg("shape", pp::fmt(pretty(tensor_shape(node))));
    res.state(NodeDesc::State::Complete);

    return res;
  }

  NodeDesc visit(const loco::Tanh *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("input", nodename(node->input()));
    res.state(NodeDesc::State::Complete);

    return res;
  }

  NodeDesc visit(const loco::TensorSoftmax *node) final
  {
    NodeDesc res{opname(node)};

    res.arg("input", nodename(node->input()));
    res.arg("axis", pp::fmt(node->axis()));
    res.state(NodeDesc::State::Complete);

    return res;
  }

public:
  NodeDesc visit(const loco::Node *node) final { return default_node_desc(*_symtbl, node); }

private:
  const SymbolTable *_symtbl;
};

NodeDesc canonical_node_desc(const SymbolTable &tbl, const loco::CanonicalNode *canonical_node)
{
  CanonicalNodeDescBuilder builder{&tbl};
  return canonical_node->accept(&builder);
}

} // namespace

namespace locop
{

bool CanonicalNodeSummaryBuilder::build(const loco::Node *node, locop::NodeSummary &out) const
{
  // Skip if a given node does not belong to loco.canonical
  if (node->dialect() != loco::CanonicalDialect::get())
  {
    return false;
  }

  auto canonical_node = loco::must_cast<const loco::CanonicalNode *>(node);
  out = canonical_node_desc(*_tbl, canonical_node);
  return true;
}

} // namespace locop
