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

#include "Codegen.h"
#include "SubgraphContext.h"
#include "Filesystem.h"

#include <luci/IR/Nodes/CircleInput.h>
#include "luci/IR/CircleNodes.h"

#include "gtest/gtest.h"

#include <cstdlib>

namespace fs = luci_codegen_filesystem;

class CodegenTest : public ::testing::Test
{
public:
  static bool has_self_dependency_subgraph(const std::vector<luci::CircleNode *> &nodes)
  {
    return luci_codegen::Codegen::has_self_dependency_subgraph(nodes);
  }

  static std::vector<luci_codegen::SubgraphContext> &get_compiled_subgraphs(luci_codegen::Codegen &c)
  {
    return c._compiled_subgraphs;
  }
};

template <loco::DataType DType>
static void constructBasicNode(luci::CircleNode &node, const std::vector<int> &dims)
{
  node.dtype(DType);
  node.rank(dims.size());
  for (int i = 0; i < dims.size(); ++i)
  {
    node.dim(i).set(dims[i]);
  }
  node.shape_status(luci::ShapeStatus::VALID);
}

template <loco::DataType DType> using Type = typename loco::DataTypeImpl<DType>::Type;

template <loco::DataType DType>
void fill_data(luci::CircleConst *node, const std::vector<Type<DType>> &data)
{
  assert(node->shape_status() == luci::ShapeStatus::VALID);
  int size = 1;
  for (int i = 0; i < node->rank(); ++i)
    size *= node->dim(i).value();
  node->size<DType>(size);
  assert(data.size() == size);
  for (int i = 0; i < size; ++i)
    node->at<DType>(i) = data[i];
}

TEST_F(CodegenTest, self_dependency_subgraph_supported)
{
  constexpr auto dtype = loco::DataType::FLOAT32;
  std::vector<int> shape = {2, 2};
  std::vector<float> const_data = {2.f, 2.f, 2.f, 2.f};

  luci::CircleInput input;
  constructBasicNode<dtype>(input, shape);

  luci::CircleConst const_node;
  constructBasicNode<dtype>(const_node, shape);
  fill_data<loco::DataType::FLOAT32>(&const_node, const_data);

  luci::CircleAdd branch_add;
  constructBasicNode<dtype>(branch_add, shape);

  luci::CircleMul branch_mul;
  constructBasicNode<dtype>(branch_mul, shape);

  luci::CircleAdd sink;
  constructBasicNode<dtype>(sink, shape);

  branch_add.x(&input);
  branch_add.y(&const_node);
  branch_mul.x(&input);
  branch_mul.y(&const_node);
  sink.x(&branch_add);
  sink.y(&branch_mul);

  std::vector<luci::CircleNode *> nodes = {&input, &const_node, &branch_add, &branch_mul, &sink};

  ASSERT_FALSE(has_self_dependency_subgraph(nodes));
}

TEST_F(CodegenTest, self_dependency_subgraph_unsupported)
{
  constexpr auto dtype = loco::DataType::FLOAT32;
  std::vector<int> shape = {2, 2};
  std::vector<float> const_data = {2.f, 2.f, 2.f, 2.f};

  luci::CircleInput input;
  constructBasicNode<dtype>(input, shape);

  luci::CircleConst const_node;
  constructBasicNode<dtype>(const_node, shape);
  fill_data<loco::DataType::FLOAT32>(&const_node, const_data);

  luci::CircleAdd branch_add;
  constructBasicNode<dtype>(branch_add, shape);

  // this custom node does not have related CustomOut nodes, so this IR is incorrect
  // but for purposes of this test it is not important
  luci::CircleCustom unsupported_node(1, 1);
  constructBasicNode<dtype>(unsupported_node, shape);

  luci::CircleAdd sink;
  constructBasicNode<dtype>(sink, shape);

  branch_add.x(&input);
  branch_add.y(&const_node);
  unsupported_node.inputs(0, &input);
  sink.x(&branch_add);
  sink.y(&unsupported_node);

  std::vector<luci::CircleNode *> nodes = {&input, &branch_add, &sink};

  ASSERT_TRUE(has_self_dependency_subgraph(nodes));
}

TEST_F(CodegenTest, no_scheduler_regression_test)
{
  fs::path tmp_dir_path = fs::temp_directory_path() / "generated-XXXXXX";
  std::string string_path = tmp_dir_path;
  std::vector<char> raw_path(string_path.c_str(), string_path.c_str() + string_path.length() + 1);
  mkdtemp(raw_path.data());
  tmp_dir_path = raw_path.data();

  constexpr auto dtype = loco::DataType::FLOAT32;
  std::vector<int> shape = {2, 2};

  luci::Module module;
  std::unique_ptr<loco::Graph> graph = std::make_unique<loco::Graph>();

  auto *in = graph->nodes()->create<luci::CircleInput>();
  constructBasicNode<dtype>(*in, shape);
  auto *add = graph->nodes()->create<luci::CircleAdd>();
  constructBasicNode<dtype>(*add, shape);
  auto *out = graph->nodes()->create<luci::CircleOutput>();
  constructBasicNode<dtype>(*out, shape);
  in->index(graph->inputs()->create()->index());
  out->index(graph->outputs()->create()->index());

  module.add(std::move(graph));

  add->x(in);
  add->y(in);
  out->from(add);

  luci_codegen::Codegen codegen;
  codegen.process_module(module);

  auto &compiled_subgraphs = get_compiled_subgraphs(codegen);

  ASSERT_EQ(compiled_subgraphs.size(), 1);

  codegen.emit_code(tmp_dir_path);

  fs::remove(tmp_dir_path / (compiled_subgraphs[0].get_name() + ".h"));
  fs::remove(tmp_dir_path / (compiled_subgraphs[0].get_name() + ".cpp"));
  fs::remove(tmp_dir_path / (compiled_subgraphs[0].get_name() + ".o"));
  fs::remove(tmp_dir_path);
}

TEST_F(CodegenTest, no_compilation_for_constant_graphs_regression)
{
  constexpr auto dtype = loco::DataType::FLOAT32;
  std::vector<int> shape = {2, 2};

  luci::Module module;
  std::unique_ptr<loco::Graph> graph = std::make_unique<loco::Graph>();

  auto *in = graph->nodes()->create<luci::CircleConst>();
  constructBasicNode<dtype>(*in, shape);
  auto *add = graph->nodes()->create<luci::CircleAdd>();
  constructBasicNode<dtype>(*add, shape);
  auto *out = graph->nodes()->create<luci::CircleOutput>();
  constructBasicNode<dtype>(*out, shape);
  out->index(graph->outputs()->create()->index());

  module.add(std::move(graph));

  add->x(in);
  add->y(in);
  out->from(add);

  luci_codegen::Codegen codegen;
  codegen.process_module(module);

  auto &compiled_subgraphs = get_compiled_subgraphs(codegen);

  ASSERT_EQ(compiled_subgraphs.size(), 0);
}
