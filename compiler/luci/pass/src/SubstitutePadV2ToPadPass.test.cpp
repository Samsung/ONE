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
#include "luci/Pass/SubstitutePadV2ToPadPass.h"
#include "luci/Pass/CircleShapeInferencePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

using UIntList = std::initializer_list<uint32_t>;
using IntList = std::initializer_list<int32_t>;

// convert shape in UIntList to loco::TensorShape
std::unique_ptr<loco::TensorShape> tensor_shape(const UIntList &values)
{
  auto shape = std::make_unique<loco::TensorShape>();
  {
    shape->rank(values.size());

    uint32_t r = 0;
    for (auto v : values)
      shape->dim(r++).set(v);
  }
  return shape;
}

class TestGraph
{
public:
  void init(const UIntList &input_shape, const UIntList &output_shape)
  {
    _input = _g.nodes()->create<luci::CircleInput>();
    {
      _input->name("input");
      _input->dtype(loco::DataType::FLOAT32);
      _input->shape(input_shape);

      auto graph_input = _g.inputs()->create();
      {
        _input->index(graph_input->index());
        graph_input->shape(std::move(tensor_shape(input_shape)));
      }
    }

    _output = _g.nodes()->create<luci::CircleOutput>();
    {
      _output->name("output");
      _output->dtype(loco::DataType::FLOAT32);
      _output->shape(output_shape);

      auto graph_output = _g.outputs()->create();
      {
        _output->index(graph_output->index());
        graph_output->shape(std::move(tensor_shape(output_shape)));
      }
    }

    // subclass should implement build_body()
    auto graphlet_before_output = build_body(_input);

    _output->from(graphlet_before_output);
  }

  // build luci::CircleConst for paddings
  luci::CircleConst *paddings_const(const std::vector<int32_t> &plist)
  {
    assert(plist.size() == 8);

    auto node = _g.nodes()->create<luci::CircleConst>();
    {
      node->dtype(loco::DataType::S32);
      node->shape({4, 2});
      node->size<loco::DataType::S32>(8);

      for (int32_t t = 0; t < 8; t++)
        node->at<loco::DataType::S32>(t) = plist.at(t);
    }

    return node;
  }

  // build luci::CircleConst for paddings value
  luci::CircleConst *padding_val_const(float val)
  {
    auto node = _g.nodes()->create<luci::CircleConst>();
    {
      node->dtype(loco::DataType::FLOAT32);
      node->shape({1});
      node->size<loco::DataType::FLOAT32>(1);

      node->at<loco::DataType::FLOAT32>(0) = val;
    }

    return node;
  }

  // build luci::CirclePadV2
  luci::CirclePadV2 *padV2(loco::Node *input, const std::vector<int32_t> &paddings,
                           float padding_constant)
  {
    auto padv2 = _g.nodes()->create<luci::CirclePadV2>();
    {
      padv2->name("PadV2");
      padv2->dtype(loco::DataType::FLOAT32);

      padv2->input(input);
      padv2->paddings(paddings_const(paddings));
      padv2->constant_values(padding_val_const(padding_constant));
      // No shape setting. ShapeInference should be run later
    }
    return padv2;
  }

  // build luci::CircleMaxPool2D
  luci::CircleMaxPool2D *maxpool2d(loco::Node *input,
                                   const std::pair<uint32_t, uint32_t> &kernel_HW)
  {
    auto mp = _g.nodes()->create<luci::CircleMaxPool2D>();
    {
      mp->value(input);
      mp->fusedActivationFunction(luci::FusedActFunc::NONE);
      mp->padding(luci::Padding::VALID);
      mp->filter()->h(kernel_HW.first);
      mp->filter()->w(kernel_HW.second);
      mp->stride()->h(1);
      mp->stride()->w(1);

      mp->dtype(loco::DataType::FLOAT32);
      // No shape setting. ShapeInference should be run later
    }
    return mp;
  }

  // build luci::CircleRelu
  luci::CircleRelu *relu(loco::Node *input)
  {
    auto relu = _g.nodes()->create<luci::CircleRelu>();
    {
      relu->features(input);
      relu->dtype(loco::DataType::FLOAT32);
      // No shape setting. ShapeInference should be run later
    }
    return relu;
  }

  // build luci::CircleTranspose
  luci::CircleTranspose *transpose(loco::Node *input, const std::vector<int32_t> &perm_v)
  {
    auto perm = _g.nodes()->create<luci::CircleConst>();
    {
      auto rank = static_cast<uint32_t>(perm_v.size());
      perm->dtype(loco::DataType::S32);
      perm->size<loco::DataType::S32>(rank);
      perm->shape({rank});
      for (decltype(rank) d = 0; d < rank; d++)
        perm->at<loco::DataType::S32>(d) = perm_v.at(d);
    }
    auto transpose_node = _g.nodes()->create<luci::CircleTranspose>();
    {
      transpose_node->a(input);
      transpose_node->perm(perm);
      transpose_node->dtype(loco::DataType::S32);
      // No shape setting. ShapeInference should be run later
    }
    return transpose_node;
  }

  loco::Graph *g() { return &_g; }
  luci::CircleOutput *output() { return _output; }

  virtual loco::Node *build_body(loco::Node *input) = 0;

private:
  loco::Graph _g;
  luci::CircleInput *_input = nullptr;
  luci::CircleOutput *_output = nullptr;
};

class SubstitutePadV2ToPadPassTest : public ::testing::Test
{
public:
  SubstitutePadV2ToPadPassTest() = default;

  bool run_pass(loco::Graph *g)
  {
    _shapeinf_pass.run(g);

    return _pad_pass.run(g);
  }

protected:
  luci::SubstitutePadV2ToPadPass _pad_pass;
  luci::CircleShapeInferencePass _shapeinf_pass;
};

} // namespace

/**
 * Graph that is changed by SubstitutePadV2ToPadPass
 *
 *    [CircleInput]
 *         |
 *      [Relu]
 *         |
 *    [CirclePadV2]  pad.H.front = 1, pad.H.end = 1, pad.W.front = 1, pad.W.end = 1
 *         |
 *     [MaxPool2D]    filter.H = 2, filter.W = 2
 *         |
 *    [CircleOutput]
 */
TEST_F(SubstitutePadV2ToPadPassTest, basic_case)
{
  struct Graph_basic : public TestGraph
  {
    Graph_basic()
    {
      UIntList input_shape = {1, 4, 4, 3};
      UIntList output_shape = {1, 6, 6, 3};
      init(input_shape, output_shape);
    }

    loco::Node *build_body(loco::Node *input) final
    {
      auto relu_node = relu(input);

      IntList paddings = {0, 0, 1, 1, 1, 1, 0, 0};
      auto padding_const = -10.0;
      auto padV2_node = padV2(relu_node, paddings, padding_const);

      return maxpool2d(padV2_node, {2, 2});
    }
  } graph;

  auto result = run_pass(graph.g());
  ASSERT_TRUE(result);

  // Checking CircleMaxPool2D
  auto maxpool = dynamic_cast<luci::CircleMaxPool2D *>(graph.output()->from());
  ASSERT_TRUE(maxpool != nullptr);

  // Checking CirclePad
  auto pad = dynamic_cast<luci::CirclePad *>(maxpool->value());
  ASSERT_TRUE(pad != nullptr);

  // Checking CircleRelu
  auto relu = dynamic_cast<luci::CircleRelu *>(pad->input());
  ASSERT_TRUE(relu != nullptr);

  auto input = dynamic_cast<luci::CircleInput *>(relu->features());
  ASSERT_TRUE(input != nullptr);
}

/**
 * Graph that is changed by SubstitutePadV2ToPadPass
 *
 * Transpose ops are inserted, e.g., to switch layout between NHWC and NCHW
 *
 *    [CircleInput]
 *         |
 *      [Relu]
 *         | 1x4x4x3  (NHWC)
 *     [Transpose]  perm=[0,3,1,2]
 *         | 1x3x4x4  (NCHW)
 *    [CirclePadV2] paddings=[0,0,0,0,1,1,1,1]
 *         | 1x3x6x6  (NCHW)
 *     [Transpose]  perm=[0,2,3,1]
 *         | 1x6x6x3  (NHWC)
 *     [MaxPool2D]  filter.H = 3, filter.W = 3
 *         | 1x4x4x3  (NHWC)
 *    [CircleOutput]
 */
TEST_F(SubstitutePadV2ToPadPassTest, reshaping_op_case)
{
  struct Graph_Reshaping_Op : public TestGraph
  {
    Graph_Reshaping_Op()
    {
      UIntList input_shape = {1, 4, 4, 3};
      UIntList output_shape = {1, 4, 4, 3};
      init(input_shape, output_shape);
    }

    loco::Node *build_body(loco::Node *input) final
    {
      auto relu_node = relu(input);

      auto transpose1_node = transpose(relu_node, {0, 3, 1, 2});

      IntList paddings = {0, 0, 0, 0, 1, 1, 1, 1};
      auto padding_const = -10.0;
      auto padV2_node = padV2(transpose1_node, paddings, padding_const);

      auto transpose2_node = transpose(padV2_node, {0, 2, 3, 1});

      return maxpool2d(transpose2_node, {3, 3});
    }
  } graph;

  auto result = run_pass(graph.g());
  ASSERT_TRUE(result);

  // Checking CircleMaxPool2D
  auto maxpool = dynamic_cast<luci::CircleMaxPool2D *>(graph.output()->from());
  ASSERT_TRUE(maxpool != nullptr);

  // Checking Transpose
  auto transpose1 = dynamic_cast<luci::CircleTranspose *>(maxpool->value());
  ASSERT_TRUE(transpose1 != nullptr);

  // Checking CirclePad
  auto pad = dynamic_cast<luci::CirclePad *>(transpose1->a());
  ASSERT_TRUE(pad != nullptr);

  // Checking Transpose
  auto transpose2 = dynamic_cast<luci::CircleTranspose *>(pad->input());
  ASSERT_TRUE(transpose2 != nullptr);

  // Checking CircleRelu
  auto relu = dynamic_cast<luci::CircleRelu *>(transpose2->a());
  ASSERT_TRUE(relu != nullptr);

  auto input = dynamic_cast<luci::CircleInput *>(relu->features());
  ASSERT_TRUE(input != nullptr);
}

//
// Negative Tests
//

/**
 * Graph that is not changed by SubstitutePadV2ToPadPass
 *
 *    [CircleInput]
 *         |
 *    [CirclePadV2]
 *         |
 *    [CircleOutput]
 */
TEST_F(SubstitutePadV2ToPadPassTest, no_relu_maxpool_NEG)
{
  struct Graph_No_MaxPool : public TestGraph
  {
    Graph_No_MaxPool()
    {
      UIntList input_shape = {1, 4, 4, 3};
      UIntList output_shape = {1, 6, 8, 3};
      init(input_shape, output_shape);
    }

    loco::Node *build_body(loco::Node *input) final
    {
      IntList paddings = {0, 0, 1, 1, 2, 2, 0, 0};
      auto padding_const = -10.0;
      return padV2(input, paddings, padding_const);
    }
  } graph;

  auto result = run_pass(graph.g());

  ASSERT_FALSE(result);
}

/**
 * Graph that is not changed by SubstitutePadV2ToPadPass
 *
 * There is no CircleMaxPool2D.
 *
 *    [CircleInput]
 *         |
 *    [CircleRelu]
 *         |
 *    [CirclePadV2]
 *         |
 *    [CircleOutput]
 */
TEST_F(SubstitutePadV2ToPadPassTest, no_maxpool_NEG)
{
  struct Graph_No_MaxPool : public TestGraph
  {
    Graph_No_MaxPool()
    {
      UIntList input_shape = {1, 4, 4, 3};
      UIntList output_shape = {1, 6, 8, 3};
      init(input_shape, output_shape);
    }

    loco::Node *build_body(loco::Node *input) final
    {
      auto relu_node = relu(input);

      IntList paddings = {0, 0, 1, 1, 2, 2, 0, 0};
      auto padding_const = -10.0;
      return padV2(relu_node, paddings, padding_const);
    }
  } graph;

  auto result = run_pass(graph.g());

  ASSERT_FALSE(result);
}

/**
 * Graph where PadV2 has non-negative constant value
 *
 *    [CircleInput]
 *         |
 *      [Relu]
 *         |
 *    [CirclePadV2]
 *         |
 *     [MaxPool2D]
 *         |
 *    [CircleOutput]
 */
TEST_F(SubstitutePadV2ToPadPassTest, non_negative_NEG)
{
  struct NegGraph : public TestGraph
  {
    NegGraph()
    {
      UIntList input_shape = {1, 4, 4, 3};
      UIntList output_shape = {1, 6, 6, 3};
      init(input_shape, output_shape);
    }

    loco::Node *build_body(loco::Node *input) final
    {
      constexpr auto POSITIVE_CONST_VALUE = 0.1f;

      auto relu_node = relu(input);

      IntList paddings = {0, 0, 1, 1, 1, 1, 0, 0};
      auto padV2_node = padV2(relu_node, paddings, POSITIVE_CONST_VALUE);

      return maxpool2d(padV2_node, {2, 2});
    }
  } graph;

  auto result = run_pass(graph.g());

  ASSERT_FALSE(result);
}

/**
 * Graph that has PadV2.padding wider than MaxPool2D.Filter
 *
 *    [CircleInput]
 *         |
 *    [CircleRelu]
 *         |
 *    [CirclePadV2]      paddings=[0, 0, 3, 3, 1, 1, 0, 0]
 *         |
 *    [CircleMaxPool2D]  Filter_H = 2, Filter_W = 2  (Filter_H < paddings for H)
 *         |
 *    [CircleOutput]
 */
TEST_F(SubstitutePadV2ToPadPassTest, wider_paddings_01_NEG)
{
  struct NegGraph : public TestGraph
  {
    NegGraph()
    {
      UIntList input_shape = {1, 4, 4, 3};
      UIntList output_shape = {1, 9, 5, 3};
      init(input_shape, output_shape);
    }

    loco::Node *build_body(loco::Node *input) final
    {
      auto relu_node = relu(input);

      constexpr auto TOO_WIDE_H_FRONT = 3;
      constexpr auto TOO_WIDE_H_END = 3;

      IntList paddings = {0, 0, TOO_WIDE_H_FRONT, TOO_WIDE_H_END, 1, 1, 0, 0};
      auto padding_const = -10.0;
      auto padv2 = padV2(relu_node, paddings, padding_const);

      return maxpool2d(padv2, {2, 2});
    }
  } graph;

  auto result = run_pass(graph.g());

  ASSERT_FALSE(result);
}

/**
 * Graph that has PadV2.paddings wider than MaxPool2D.Filter
 *
 * Transpose ops are inserted, e.g., to switch layout between NHWC and NCHW
 *
 *    [CircleInput]
 *         |
 *      [Relu]
 *         | 1x4x4x3 (NHWC)
 *     [Transpose]  perm=[0,3,1,2]
 *         | 1x3x4x4 (NCHW)
 *    [CirclePadV2] paddings=[0,0,0,0,3,3,1,1]
 *         | 1x3x6x6 (NCHW)
 *     [Transpose]  perm=[0,2,3,1]
 *         | 1x6x6x3 (NHWC)
 *     [MaxPool2D]  filter.H = 2, filter.W = 2
 *         | 1x4x4x3
 *    [CircleOutput]
 */
TEST_F(SubstitutePadV2ToPadPassTest, wider_paddings_02_NEG)
{
  struct Graph_Reshaping_Op : public TestGraph
  {
    Graph_Reshaping_Op()
    {
      UIntList input_shape = {1, 4, 4, 3};
      UIntList output_shape = {1, 9, 5, 3};
      init(input_shape, output_shape);
    }

    loco::Node *build_body(loco::Node *input) final
    {
      auto relu_node = relu(input);

      auto transpose1_node = transpose(relu_node, {0, 3, 1, 2});

      constexpr auto TOO_WIDE_H_FRONT = 3;
      constexpr auto TOO_WIDE_H_END = 3;

      IntList paddings = {0, 0, 0, 0, TOO_WIDE_H_FRONT, TOO_WIDE_H_END, 1, 1};
      auto padding_const = -10.0;
      auto padV2_node = padV2(transpose1_node, paddings, padding_const);

      auto transpose2_node = transpose(padV2_node, {0, 2, 3, 1});

      return maxpool2d(transpose2_node, {3, 3});
    }
  } graph;

  auto result = run_pass(graph.g());
  ASSERT_FALSE(result);
}
