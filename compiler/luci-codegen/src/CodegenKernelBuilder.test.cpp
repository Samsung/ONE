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

#include <luci/IR/Nodes/CircleInput.h>

#include "luci/IR/CircleNodes.h"

#include "gtest/gtest.h"

#include "CodegenKernelBuilder.h"

#include "Halide.h"

void constructBasicNode(luci::CircleNode &node, const std::vector<int> &dims)
{
  node.dtype(loco::DataType::FLOAT32);
  node.rank(dims.size());
  for (int i = 0; i < dims.size(); ++i)
  {
    node.dim(i).set(dims[i]);
  }
  node.shape_status(luci::ShapeStatus::VALID);
}

TEST(codegen_kernels, add_scalar)
{
  // construct test graph
  luci::CircleInput input_x;
  constructBasicNode(input_x, {1});
  luci::CircleInput input_y;
  constructBasicNode(input_y, {1});

  luci::CircleAdd add;
  constructBasicNode(add, {1});
  add.x(&input_x);
  add.y(&input_y);

  luci::CircleOutput output_node;
  constructBasicNode(output_node, {1});
  output_node.from(&add);

  ASSERT_TRUE(luci_codegen::CodegenKernelBuilder::is_supported(&add));

  luci_codegen::SubgraphContext subgraph;
  subgraph.add_node(&add);
  subgraph.finish_construction();

  luci_codegen::CodegenKernelBuilder builder(subgraph);
  builder.process();

  float x_data[] = {1.5f};
  float y_data[] = {3.5f};
  std::vector<float> ref_res_data{5.0f};

  Halide::Buffer<float> x(x_data);
  Halide::Buffer<float> y(y_data);
  Halide::Buffer<float> res(1);

  Halide::ImageParam input_param_x = subgraph.get_inputs()[0].second;
  Halide::ImageParam input_param_y = subgraph.get_inputs()[1].second;

  Halide::ParamMap params;
  params.set(input_param_x, x);
  params.set(input_param_y, y);

  Halide::Func target_func = subgraph.get_outputs()[0].second;
  target_func.realize(res, Halide::Target(), params);

  for (int i = 0; i < ref_res_data.size(); ++i)
    ASSERT_EQ(ref_res_data[i], res.data()[i]);
}
