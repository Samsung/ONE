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

#ifndef __LUCI_PASS_TEST_GRAPHS_H__
#define __LUCI_PASS_TEST_GRAPHS_H__

#include <loco.h>
#include <luci/IR/CircleNodes.h>

namespace luci
{

/**
 *  ConstantFoldingTestGraph is a parent class for testing
 *  constant folding passes. It creates Input, Add, Output
 *  in the below graph. Child class must implement the folded pattern
 *  in createFoldedPattern().
 *
 *      [Input]   [Folded pattern] (Implemented by child class)
 *           \     /
 *            [Add]
 *              |
 *           [Output]
 *
 *    - Input type == Output type
 *    - Input shape == Output shape
 *    - Folded pattern must have the same type with Input/Output
 */
class ConstantFoldingTestGraph
{
public:
  ConstantFoldingTestGraph(std::vector<uint32_t> input_shape, loco::DataType input_dtype)
  {
    input = g.nodes()->create<luci::CircleInput>();
    add = g.nodes()->create<luci::CircleAdd>();
    output = g.nodes()->create<luci::CircleOutput>();

    auto graph_input = g.inputs()->create();
    input->index(graph_input->index());
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());

    graph_input->dtype(input_dtype);
    graph_output->dtype(input_dtype);
    input->dtype(input_dtype);
    add->dtype(input_dtype);
    output->dtype(input_dtype);

    auto input_tensor_shape = std::make_unique<loco::TensorShape>();
    input_tensor_shape->rank(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++)
      input_tensor_shape->dim(i).set(input_shape[i]);
    graph_input->shape(std::move(input_tensor_shape));

    auto output_tensor_shape = std::make_unique<loco::TensorShape>();
    output_tensor_shape->rank(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++)
      output_tensor_shape->dim(i).set(input_shape[i]);
    graph_output->shape(std::move(output_tensor_shape));

    input->rank(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++)
      input->dim(i).set(input_shape[i]);

    add->rank(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++)
      add->dim(i).set(input_shape[i]);

    output->rank(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++)
      output->dim(i).set(input_shape[i]);

    add->x(input);
    output->from(add);
  }

  void init() { add->y(createFoldedPattern()); }

  virtual ~ConstantFoldingTestGraph() = default;

  virtual loco::Node *createFoldedPattern() = 0;

  // NOTE: we're not adding _ prefix as these class members are public
public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleAdd *add = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace luci

#endif // __LUCI_PASS_TEST_GRAPHS_H__
