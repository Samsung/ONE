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

#include "locop/FormattedGraph.h"
#include "locop/FormattedTensorShape.h"
#include "locop/GenericNodeSummaryBuilder.h"

#include <loco/Service/TypeInference.h>
#include <loco/Service/ShapeInference.h>

#include <pp/Format.h>

#include <memory>
#include <map>
#include <set>

#include <cassert>

using locop::SymbolTable;

namespace
{

std::string str(const loco::DataType &dtype)
{
  switch (dtype)
  {
    case loco::DataType::Unknown:
      return "Unknown";

    case loco::DataType::U4:
      return "U4";
    case loco::DataType::U8:
      return "U8";
    case loco::DataType::U16:
      return "U16";
    case loco::DataType::U32:
      return "U32";
    case loco::DataType::U64:
      return "U64";

    case loco::DataType::S4:
      return "S4";
    case loco::DataType::S8:
      return "S8";
    case loco::DataType::S16:
      return "S16";
    case loco::DataType::S32:
      return "S32";
    case loco::DataType::S64:
      return "S64";

    case loco::DataType::FLOAT16:
      return "FLOAT16";
    case loco::DataType::FLOAT32:
      return "FLOAT32";
    case loco::DataType::FLOAT64:
      return "FLOAT64";

    case loco::DataType::BOOL:
      return "BOOL";

    default:
      break;
  };

  throw std::invalid_argument{"dtype"};
}

std::string str(const loco::Domain &domain)
{
  // TODO Generate!
  switch (domain)
  {
    case loco::Domain::Unknown:
      return "Unknown";
    case loco::Domain::Tensor:
      return "Tensor";
    case loco::Domain::Feature:
      return "Feature";
    case loco::Domain::Filter:
      return "Filter";
    case loco::Domain::DepthwiseFilter:
      return "DWFilter";
    case loco::Domain::Bias:
      return "Bias";
    default:
      break;
  }

  throw std::invalid_argument{"domain"};
}

std::string str(const loco::NodeShape &node_shape)
{
  using namespace locop;

  switch (node_shape.domain())
  {
    case loco::Domain::Tensor:
    {
      auto tensor_shape = node_shape.as<loco::TensorShape>();
      return pp::fmt(locop::fmt<TensorShapeFormat::Plain>(&tensor_shape));
    }
    // TODO Show details
    case loco::Domain::Feature:
    case loco::Domain::Filter:
    case loco::Domain::DepthwiseFilter:
    case loco::Domain::Bias:
      return "...";

    default:
      break;
  }

  throw std::invalid_argument{"domain"};
}

// TODO Use locop::fmt<TensorShapeFormat ...>
locop::FormattedTensorShape<locop::TensorShapeFormat::Bracket>
formatted_tensor_shape(const loco::TensorShape *ptr)
{
  return locop::FormattedTensorShape<locop::TensorShapeFormat::Bracket>{ptr};
}

} // namespace

namespace
{

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

} // namespace

// TODO Remove this workaround
namespace locop
{

std::ostream &operator<<(std::ostream &os, const NodeDesc &d)
{
  assert(d.state() != NodeDesc::State::Invalid);

  std::vector<std::string> values;

  for (uint32_t n = 0; n < d.args().count(); ++n)
  {
    values.emplace_back(d.args().at(n).first + ": " + d.args().at(n).second);
  }

  if (d.state() == NodeDesc::State::PartiallyKnown)
  {
    values.emplace_back("...");
  }

  os << d.opname();
  os << "(";
  if (values.size() > 0)
  {
    os << values.at(0);
    for (uint32_t n = 1; n < values.size(); ++n)
    {
      os << ", " << values.at(n);
    }
  }
  os << ")";

  return os;
}

} // namespace locop

namespace locop
{

std::ostream &operator<<(std::ostream &os, const FormattedGraph &fmt)
{
  fmt.dump(os);
  return os;
}

} // namespace locop

namespace locop
{

void FormattedGraphImpl<Formatter::LinearV1>::dump(std::ostream &os) const
{
  struct SymbolTableImpl final : public SymbolTable
  {
    std::string lookup(const loco::Node *node) const final
    {
      if (node == nullptr)
      {
        return "(null)";
      }

      return _content.at(node);
    }

    std::map<const loco::Node *, std::string> _content;
  };

  SymbolTableImpl symbols;

  auto symbol = [&symbols](const loco::Node *node) { return symbols.lookup(node); };

  for (uint32_t n = 0; n < _graph->nodes()->size(); ++n)
  {
    symbols._content[_graph->nodes()->at(n)] = pp::fmt("%", n);
  }

  // Find the disjoint node clusters
  //
  // TODO Move this implementation into loco Algorithms.h
  std::map<loco::Node *, loco::Node *> parents;

  for (auto node : loco::all_nodes(_graph))
  {
    parents[node] = nullptr;
  }

  for (auto node : loco::all_nodes(_graph))
  {
    for (uint32_t n = 0; n < node->arity(); ++n)
    {
      if (auto arg = node->arg(n))
      {
        parents[arg] = node;
      }
    }
  }

  auto find = [&parents](loco::Node *node) {
    loco::Node *cur = node;

    while (parents.at(cur) != nullptr)
    {
      cur = parents.at(cur);
    }

    return cur;
  };

  std::set<loco::Node *> roots;

  for (auto node : loco::all_nodes(_graph))
  {
    roots.insert(find(node));
  }

  std::map<loco::Node *, std::set<loco::Node *>> clusters;

  // Create clusters
  for (auto root : roots)
  {
    clusters[root] = std::set<loco::Node *>{};
  }

  for (auto node : loco::all_nodes(_graph))
  {
    clusters.at(find(node)).insert(node);
  }

  std::unique_ptr<locop::NodeSummaryBuilder> node_summary_builder;

  if (_factory)
  {
    // Use User-defined NodeSummaryBuilder if NodeSummaryBuilderFactory is present
    node_summary_builder = _factory->create(&symbols);
  }
  else
  {
    // Use Built-in NodeSummaryBuilder otherwise
    node_summary_builder = std::make_unique<GenericNodeSummaryBuilder>(&symbols);
  }

  // Print Graph Input(s)
  for (uint32_t n = 0; n < _graph->inputs()->size(); ++n)
  {
    auto input = _graph->inputs()->at(n);

    std::string name = input->name();

    std::string shape = "?";
    if (input->shape() != nullptr)
    {
      shape = pp::fmt(formatted_tensor_shape(input->shape()));
    }

    // TODO Print dtype
    os << pp::fmt("In #", n, " { name: ", name, ", shape: ", shape, " }") << std::endl;
  }

  // Print Graph Output(s)
  for (uint32_t n = 0; n < _graph->outputs()->size(); ++n)
  {
    auto output = _graph->outputs()->at(n);

    std::string name = output->name();

    std::string shape = "?";
    if (output->shape() != nullptr)
    {
      shape = pp::fmt(formatted_tensor_shape(output->shape()));
    }

    // TODO Print dtype
    os << pp::fmt("Out #", n, " { name: ", name, ", shape: ", shape, " }") << std::endl;
  }

  if (_graph->inputs()->size() + _graph->outputs()->size() != 0)
  {
    os << std::endl;
  }

  for (auto it = clusters.begin(); it != clusters.end(); ++it)
  {
    std::vector<loco::Node *> cluster_outputs;

    for (auto node : it->second)
    {
      // NOTE This is inefficient but anyway working :)
      if (loco::succs(node).empty())
      {
        cluster_outputs.emplace_back(node);
      }
    }

    for (auto node : loco::postorder_traversal(cluster_outputs))
    {
      locop::NodeSummary node_summary;

      // Build a node summary
      if (!node_summary_builder->build(node, node_summary))
      {
        throw std::runtime_error{"Fail to build a node summary"};
      }

      for (uint32_t n = 0; n < node_summary.comments().count(); ++n)
      {
        os << "; " << node_summary.comments().at(n) << std::endl;
      }

      os << symbol(node);

      if (loco::shape_known(node))
      {
        auto node_shape = loco::shape_get(node);
        os << " : " << str(node_shape.domain());
        os << "<";
        os << str(node_shape);
        os << ", ";
        // Show DataType
        os << (loco::dtype_known(node) ? str(loco::dtype_get(node)) : std::string{"?"});
        os << ">";
      }

      os << " = " << node_summary << std::endl;
    }
    os << std::endl;
  }
}

} // namespace locop
