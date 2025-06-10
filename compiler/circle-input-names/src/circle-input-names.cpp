/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
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

#include <iostream>
#include <crew/PConfigJson.h>
#include <luci/CircleNodeSummaryBuilders.h>

using namespace luci;

// clang-format off
// For Variadic Arity Nodes which have multiple inputs.
template <typename T> struct is_VariadicArity : std::false_type {};
template <> struct is_VariadicArity<CircleConcatenation> : std::true_type {};
template <> struct is_VariadicArity<CircleCustom> : std::true_type {};
template <> struct is_VariadicArity<CirclePack> : std::true_type {};
template <> struct is_VariadicArity<CircleAddN> : std::true_type {};
template <> struct is_VariadicArity<CircleIf> : std::true_type {};
template <> struct is_VariadicArity<CircleWhile> : std::true_type {};

// For Variadic Outputs Nodes which have multiple outputs.
template <typename T> struct is_VariadicOut : std::false_type {};
template <> struct is_VariadicOut<CircleCustom> : std::true_type{};
template <> struct is_VariadicOut<CircleIf> : std::true_type {};
template <> struct is_VariadicOut<CircleWhile> : std::true_type {};
// clang-format on

// For Circle Nodes which have variadic arity and variadic outputs
template <typename CircleOp, typename std::enable_if_t<is_VariadicArity<CircleOp>::value &&
                                                       is_VariadicOut<CircleOp>::value> * = nullptr>
auto CircleNodeCreator()
{
  return CircleOp(1, 1);
}

// For Circle Nodes which have variadic arity but single output
template <typename CircleOp,
          typename std::enable_if_t<is_VariadicArity<CircleOp>::value &&
                                    !is_VariadicOut<CircleOp>::value> * = nullptr>
auto CircleNodeCreator()
{
  return CircleOp(1);
}

// For Circle Nodes which have fixed arity
template <typename CircleOp,
          typename std::enable_if_t<!is_VariadicArity<CircleOp>::value> * = nullptr>
auto CircleNodeCreator()
{
  return CircleOp();
}

// Add fused activation function option to CircleNode if it supports FusedActFunc traits
void add_fused_actfn_option(CircleNode *node)
{
  auto node_ = dynamic_cast<CircleNodeMixin<CircleNodeTrait::FusedActFunc> *>(node);
  if (node_)
  {
    node_->fusedActivationFunction(luci::FusedActFunc::RELU);
  }
}

// Add padding option to AVERAGE_POOL_2D, CONV_2D, DEPTHWISE_CONV_2D, L2_POOL_2D, MAX_POOL_2D,
// TRANSPOSE_CONV nodes
void add_padding_option(CircleNode *node)
{
  switch (node->opcode())
  {
#define CIRCLE_NODE(OPCODE, CLASS)            \
  case luci::CircleOpcode::OPCODE:            \
  {                                           \
    auto node_ = dynamic_cast<CLASS *>(node); \
    node_->padding(Padding::SAME);            \
    break;                                    \
  }
    CIRCLE_NODE(AVERAGE_POOL_2D, CircleAveragePool2D)
    CIRCLE_NODE(CONV_2D, CircleConv2D)
    CIRCLE_NODE(DEPTHWISE_CONV_2D, CircleDepthwiseConv2D)
    CIRCLE_NODE(L2_POOL_2D, CircleL2Pool2D)
    CIRCLE_NODE(MAX_POOL_2D, CircleMaxPool2D)
    CIRCLE_NODE(TRANSPOSE_CONV, CircleTransposeConv)
#undef CIRCLE_NODE
    default:
    {
      break;
    }
  }
  return;
}

// Add mode option to MIRROR_PAD, ROPE nodes
void add_mode_option(CircleNode *node)
{
  switch (node->opcode())
  {
#define CIRCLE_NODE(OPCODE, CLASS)            \
  case luci::CircleOpcode::OPCODE:            \
  {                                           \
    auto node_ = dynamic_cast<CLASS *>(node); \
    auto mode_ = node_->mode();               \
    node_->mode(decltype(mode_)(1));          \
    break;                                    \
  }
    CIRCLE_NODE(MIRROR_PAD, CircleMirrorPad)
    CIRCLE_NODE(ROPE, CircleRoPE)
#undef CIRCLE_NODE
    default:
    {
      break;
    }
  }
  return;
}

// Fill dummy values to CircleNode for creating NodeSummary
void fill_dummies_for_summary_creation(CircleNode *node)
{
  add_fused_actfn_option(node);
  add_padding_option(node);
  add_mode_option(node);
}

// Mock Symbol Table for CircleNodeSummaryBuilder
class MockSymbolTable : public locop::SymbolTable
{
  std::string lookup(const loco::Node *) const override { return ""; }
};

// Create NodeSummary using CircleNodeSummaryBuilder and MockSymbolTable
locop::NodeSummary create_circle_node_summary(CircleNode *node)
{
  locop::NodeSummary s;
  MockSymbolTable tbl;
  CircleNodeSummaryBuilder builder;

  builder.build(node, &tbl, s);
  return s;
}

// Get input names of CircleNode and export as JSON format
void get_input_names_from_summary(CircleNode *node, locop::NodeSummary &s,
                                  crew::JsonExport &json_export)
{
  std::vector<std::string> arg_names;
  for (int i = 0; i < node->arity(); i++)
  {
    auto args = s.args().at(i);
    // args : pair(name, value)
    arg_names.emplace_back(args.first);
  }

  // s.opname() : "Circle.Opname""
  auto opname = s.opname().substr(7);
  // "Opname" : ["arg1", "arg2",...],"
  json_export.key_val(opname, arg_names, true);
}

int main(void)
{
  std::stringstream ss;
  crew::JsonExport json_export(ss);
  // "{"
  json_export.open_brace();
#define CIRCLE_NODE(OP, CIRCLE_OP)                             \
  {                                                            \
    auto node = CircleNodeCreator<CIRCLE_OP>();                \
    fill_dummies_for_summary_creation(&node);                  \
    auto summary = create_circle_node_summary(&node);          \
    get_input_names_from_summary(&node, summary, json_export); \
  }
#define CIRCLE_VNODE(_1, _2)
#include <luci/IR/CircleNodes.lst>
#undef CIRCLE_NODE
#undef CIRCLE_VNODE
  // Remove last comma from stringstream 'ss'
  ss.seekp(-2, std::ios_base::end) << '\n';
  // "}"
  json_export.close_brace(false);
  std::cout << ss.str();

  return 0;
}
