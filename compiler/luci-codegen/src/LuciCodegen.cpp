/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "LuciCodegen.h"
#include "loco/IR/Algorithm.h"
#include "luci/IR/CircleNodeVisitor.h"

#include "Halide.h"

#include <map>

namespace luci_codegen
{

class CodegenContext
{
public:

  std::map<luci::CircleNode *, Halide::Func> &generated_funcs()
  {
    return _generated_funcs;
  }

  std::vector<Halide::Argument> inputs()
  {
    return _inputs;
  }

  void add_input(Halide::Argument input)
  {
    _inputs.push_back(input);
  }

private:
  std::map<luci::CircleNode *, Halide::Func> _generated_funcs;
  std::vector<Halide::Argument> _inputs;
};

LuciCodegen::LuciCodegen(const Options &options) : _context(new CodegenContext), _options(options) {}

LuciCodegen::~LuciCodegen() {}

namespace
{

class CodegenKernelBuilder: public luci::CircleNodeMutableVisitor<void>
{
private:
  CodegenContext &_context;

  // elementwise operator supports
  struct AddOp
  {
    static Halide::Expr op(Halide::Expr a, Halide::Expr b)
    {
      return a + b;
    }
  };

  struct SubOp
  {
    static Halide::Expr op(Halide::Expr a, Halide::Expr b)
    {
      return a - b;
    }
  };

  struct MulOp
  {
    static Halide::Expr op(Halide::Expr a, Halide::Expr b)
    {
      return a * b;
    }
  };

  struct DivOp
  {
    static Halide::Expr op(Halide::Expr a, Halide::Expr b)
    {
      return a / b;
    }
  };

  template <typename OP>
  void binary_operator(luci::CircleNode *node)
  {
    auto rank = node->rank();
    std::vector<Halide::Expr> output_vars = iter_space(node);
    std::vector<Halide::Expr> arg_a_vars = debroadcast_iter_space(output_vars, node); // separate variables according broadcasting rules
    std::vector<Halide::Expr> arg_b_vars = debroadcast_iter_space(output_vars, node);
    Halide::Func output_func;
    Halide::Func input_a = get_func(static_cast<luci::CircleNode *>(node->arg(0)));
    Halide::Func input_b = get_func(static_cast<luci::CircleNode *>(node->arg(1)));

    output_func(output_vars) = OP::op(input_a(arg_a_vars), input_b(arg_b_vars));
    _context.generated_funcs()[node] = output_func;
  }

public:

  Halide::Type transform_type(loco::DataType dtype)
  {
    switch (dtype)
    {
      case loco::DataType::FLOAT32:
        return Halide::Type(Halide::Type::Float, 32, 1);
      case loco::DataType::FLOAT64:
        return Halide::Type(Halide::Type::Float, 64, 1);
      case loco::DataType::S32:
        return Halide::Type(Halide::Type::Int, 32, 1);
      case loco::DataType::S64:
        return Halide::Type(Halide::Type::Int, 64, 1);
      default:
        assert("NYI");
    }
    return Halide::Type();
  }

  Halide::Func get_func(luci::CircleNode *node)
  {
    if (_context.generated_funcs().count(node))
      return _context.generated_funcs()[node];
    // No function found, need to create input
    Halide::ImageParam input = Halide::ImageParam(transform_type(node->dtype()), node->rank());
    _context.add_input(input);
    return input;
  }

  std::vector<Halide::Expr> debroadcast_iter_space(const std::vector<Halide::Expr> &output_space, const luci::CircleNode *node)
  {
    int rank = node->rank();
    std::vector<Halide::Expr> iter_space(rank);
    for (int i = rank-1; i >= 0; --i)
    {
      if (node->dim(i) == 1)
        iter_space[i] = Halide::Expr(0);
      else
        iter_space[i] = output_space[i];
    }
    return iter_space;
  }

  std::vector<Halide::Expr> iter_space(const luci::CircleNode *node)
  {
    int rank = node->rank();
    std::vector<Halide::Expr> iter_space(rank);
    for (int i = 0; i < rank; ++i)
      iter_space[i] = Halide::Var();
    return iter_space;
  }

  CodegenKernelBuilder(CodegenContext &context) : _context(context) {}

  void visit(luci::CircleConst *node)
  {
    size_t rank = node->rank();
    std::vector<int> dims(rank);
    for (int i = 0; i < rank; ++i)
      dims[i] = node->dim(rank - i - 1).value();
    switch (node->dtype())
    {
      case loco::DataType::FLOAT32:
      {
        Halide::Buffer<float> buf(dims);
        size_t size = node->size<loco::DataType::FLOAT32>();
        std::vector<Halide::Expr> iter_space(rank);
        for (int i = 0; i < size; ++i)
        {
          buf.data()[i] = node->at<loco::DataType::FLOAT32>(i);
          iter_space[i] = Halide::Var();
        }
        _context.generated_funcs()[node] = Halide::Func(buf(iter_space));
      }
    }
  }

  void visit(luci::CircleAdd *node)
  {
    binary_operator<AddOp>(node);
  }

  void visit(luci::CircleSub *node)
  {
    binary_operator<SubOp>(node);
  }

  void visit(luci::CircleMul *node)
  {
    binary_operator<MulOp>(node);
  }

  void visit(luci::CircleDiv *node)
  {
    binary_operator<DivOp>(node);
  }

  /// @brief Default fallback
  void visit(luci::CircleNode *) override
  {
    INTERNAL_EXN("CodegenKernelBuilder: unsupported node");
  }
};

size_t const_node_size(const luci::CircleNode *node)  // TODO do something with luci to simplify this shame
{
  assert(node->opcode() == luci::CircleOpcode::CIRCLECONST);
  auto const_node = static_cast<const luci::CircleConst *>(node);
  switch (node->dtype())
  {
    case loco::DataType::S32:
     return sizeof(std::int32_t) * const_node->size<loco::DataType::S32>();
    case loco::DataType::S64:
      return sizeof(std::int32_t) * const_node->size<loco::DataType::S64>();
    case loco::DataType::FLOAT32:
      return sizeof(std::int32_t) * const_node->size<loco::DataType::FLOAT32>();
//    case loco::DataType::FLOAT64:
//      return sizeof(std::int32_t) * const_node->size<loco::DataType::FLOAT64>(); // double is not supported in luci
  }
  return 0;
}

} // unnamed namespace

void LuciCodegen::add_operator(luci::CircleNode *node)
{
  CodegenKernelBuilder builder(*_context);
  node->accept(&builder);
  assert(supported(node));
}


bool LuciCodegen::supported(luci::CircleNode *node)
{
  assert(dynamic_cast<luci::CircleNode *>(node));
  luci::CircleNode *circle_node = static_cast<luci::CircleNode *>(node);
  switch (circle_node->opcode())
  {
    case luci::CircleOpcode::CIRCLECONST:
      return const_node_size(node) <= _options.max_inline_buffer_threshold;
    case luci::CircleOpcode::ADD:
    case luci::CircleOpcode::SUB:
    case luci::CircleOpcode::MUL:
    case luci::CircleOpcode::DIV:
      return circle_node->shape_status() == luci::ShapeStatus::VALID;
  }
  return false;
}

void LuciCodegen::process(loco::Graph &graph)
{
  auto *inputs = graph.inputs();
  auto input = inputs->at(0);
  auto outputs = loco::output_nodes(&graph);
  for (loco::Node *node: loco::postorder_traversal(outputs))
  {
    auto circle_node = static_cast<luci::CircleNode *>(node);
    if (supported(circle_node))
      add_operator(circle_node);
  }
}

void LuciCodegen::process(luci::Module &module)
{
  auto num_graphs = module.size();
  for (size_t i = 0; i < num_graphs; ++i)
    process(*module.graph(i));
}

void LuciCodegen::emit_code(std::string package_name)
{
  int no = 0;
  for (auto &node_func: _context->generated_funcs())
  {
    ++no;
    node_func.second.compile_to_lowered_stmt("func_" + std::to_string(no) + ".html", _context->inputs(), Halide::StmtOutputFormat::HTML);
  }
  // TODO generate object files?
}

} // namespace luci_codegen
