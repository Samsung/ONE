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

#include <luci/test/TestIOGraph.h>
#include <luci/test/TestShape.h>

namespace luci
{

/**
 *  ConstantFoldingTestGraph is a base class for testing
 *  constant folding passes. It creates Input and Output
 *  in the below graph. Child classes must implement Connector
 *  and Folded pattern.
 *
 *      [Input]   [Folded pattern] (Implemented by child class)
 *           \    /
 *         [Connector] (Implemented by child class)
 *              |
 *           [Output]
 *
 *    Connector should satisfy the below conditions
 *      - Input type == Output type == Folded pattern type
 *      - Input shape == Output shape == Folded pattern shape
 *
 *    For example, Add, Mul, Sub, .. can be a Connector
 */
class ConstantFoldingTestGraph
{
public:
  ConstantFoldingTestGraph(std::vector<uint32_t> input_shape, loco::DataType input_dtype)
  {
    _input = _g.nodes()->create<luci::CircleInput>();
    _output = _g.nodes()->create<luci::CircleOutput>();

    auto graph_input = _g.inputs()->create();
    _input->index(graph_input->index());
    auto graph_output = _g.outputs()->create();
    _output->index(graph_output->index());

    graph_input->dtype(input_dtype);
    graph_output->dtype(input_dtype);
    _input->dtype(input_dtype);
    _output->dtype(input_dtype);

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

    _input->rank(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++)
      _input->dim(i).set(input_shape[i]);

    _output->rank(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++)
      _output->dim(i).set(input_shape[i]);

    _input->name("input");
    _output->name("output");
  }

  virtual void init() = 0;

  virtual ~ConstantFoldingTestGraph() = default;

  virtual loco::Node *createFoldedPattern() = 0;

  virtual luci::CircleConst *getFoldedPattern() = 0;

  loco::Graph *graph() { return &_g; }

  // NOTE: we're not adding _ prefix as these class members are public
protected:
  loco::Graph _g;
  luci::CircleInput *_input = nullptr;
  luci::CircleOutput *_output = nullptr;
};

/**
 *  ConstantFoldingTestAddGraph is ConstantFoldingTestGraph
 *  whose Connector is Add.
 */
class ConstantFoldingAddTestGraph : public ConstantFoldingTestGraph
{
protected:
  ConstantFoldingAddTestGraph(std::vector<uint32_t> input_shape, loco::DataType input_dtype)
    : ConstantFoldingTestGraph(input_shape, input_dtype)
  {
    _add = _g.nodes()->create<luci::CircleAdd>();
    _add->dtype(input_dtype);

    _add->rank(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++)
      _add->dim(i).set(input_shape[i]);

    _add->x(_input);

    _output->from(_add);

    _add->name("add");
  }

protected:
  void init() override { _add->y(createFoldedPattern()); }

protected:
  luci::CircleConst *getFoldedPattern() override
  {
    return dynamic_cast<luci::CircleConst *>(_add->y());
  }

protected:
  luci::CircleAdd *_add = nullptr;
};

/**
 *  CommonSubExpressionEliminationTestGraph is a base class for testing
 *  common subexpression elimination pass. It creates Input and Output
 *  in the below graph. Child classes must implement Expression.
 *
 *           [Input]
 *           /     \
 *  [Expression]    [Expression]
 *         |              |
 *    [Output 1]      [Output 2]
 *
 *    Expression should satisfy the below conditions
 *    - Input type == Output type
 *    - Input shape == Output shape
 *    - Expression 1 and 2 are semantically equal
 */
class CommonSubExpressionEliminationTestGraph : public test::TestIsGraphlet<1>,
                                                public test::TestOsGraphlet<2>
{
public:
  virtual void init(const std::initializer_list<test::ShapeU32> shape_in,
                    const std::initializer_list<test::ShapeU32> shape_out)
  {
    test::TestIsGraphlet<1>::init(g(), shape_in);
    test::TestOsGraphlet<2>::init(g(), shape_out);

    auto expr1 = createExpression(input(0), "expr1");
    auto expr2 = createExpression(input(0), "expr2");

    output(0)->from(expr1);
    output(1)->from(expr2);
  }

  virtual ~CommonSubExpressionEliminationTestGraph() = default;

  virtual loco::Node *createExpression(luci::CircleNode *ifm, const std::string &name) = 0;
};

} // namespace luci

#endif // __LUCI_PASS_TEST_GRAPHS_H__
