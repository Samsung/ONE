/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <luci/IR/CircleNodes.h>
#include <locop/NodeSummary.h>
#include <locop/SymbolTable.h>

#include <gtest/gtest.h>

namespace
{

class MockSymbolTable : public locop::SymbolTable
{
  std::string lookup(const loco::Node *) const override
  {
    return "Do nothing because it is mocking Symbol Table!";
  }
};

class CircleNodeSummaryBuilderTest : public ::testing::Test
{
protected:
  bool mock_build(const loco::Node *node)
  {
    return luci::CircleNodeSummaryBuilder().build(node, &_tbl, _s);
  }

protected:
  MockSymbolTable _tbl;
  locop::NodeSummary _s;
};

} // namespace

TEST_F(CircleNodeSummaryBuilderTest, Add_validate)
{
  luci::CircleAdd node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, Add_validate_fused_NEG)
{
  luci::CircleAdd node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, AveragePool2D_validate)
{
  luci::CircleAveragePool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::SAME);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, AveragePool2D_validate_fused_NEG)
{
  luci::CircleAveragePool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node.padding(luci::Padding::SAME);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, AveragePool2D_validate_padding_NEG)
{
  luci::CircleAveragePool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, BCQFullyConnected_validate)
{
  luci::CircleBCQFullyConnected node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, BCQFullyConnected_validate_fused_NEG)
{
  luci::CircleBCQFullyConnected node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, Concatenation_validate)
{
  luci::CircleConcatenation node(2);
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, Concatenation_validate_fused_NEG)
{
  luci::CircleConcatenation node(2);
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, Conv2D_validate)
{
  luci::CircleConv2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::SAME);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, Conv2D_validate_fused_NEG)
{
  luci::CircleConv2D node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node.padding(luci::Padding::SAME);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, Conv2D_validate_padding_NEG)
{
  luci::CircleConv2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, DepthwiseConv2D_validate)
{
  luci::CircleDepthwiseConv2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::SAME);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, DepthwiseConv2D_validate_fused_NEG)
{
  luci::CircleDepthwiseConv2D node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node.padding(luci::Padding::SAME);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, DepthwiseConv2D_validate_padding_NEG)
{
  luci::CircleDepthwiseConv2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, FullyConnected_validate)
{
  luci::CircleFullyConnected node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, FullyConnected_validate_fused_NEG)
{
  luci::CircleFullyConnected node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, InstanceNorm_validate)
{
  luci::CircleInstanceNorm node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, InstanceNorm_validate_fused_NEG)
{
  luci::CircleInstanceNorm node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, L2Normalize_validate)
{
  luci::CircleL2Normalize node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, L2Normalize_validate_fused_NEG)
{
  luci::CircleL2Normalize node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, L2Pool2D_validate)
{
  luci::CircleL2Pool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::SAME);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, L2Pool2D_validate_fused_NEG)
{
  luci::CircleL2Pool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node.padding(luci::Padding::SAME);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, L2Pool2D_validate_padding_NEG)
{
  luci::CircleL2Pool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, MaxPool2D_validate)
{
  luci::CircleMaxPool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::SAME);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, MaxPool2D_validate_fused_NEG)
{
  luci::CircleMaxPool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  node.padding(luci::Padding::SAME);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, MaxPool2D_validate_padding_NEG)
{
  luci::CircleMaxPool2D node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  node.padding(luci::Padding::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, MirrorPad_validate)
{
  luci::CircleMirrorPad node;
  node.mode(luci::MirrorPadMode::REFLECT);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, MirrorPad_validate_mirror_padding_NEG)
{
  luci::CircleMirrorPad node;
  node.mode(luci::MirrorPadMode::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, Mul_validate)
{
  luci::CircleMul node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, Mul_validate_fused_NEG)
{
  luci::CircleMul node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, SVDF_validate)
{
  luci::CircleSVDF node;
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, SVDF_validate_fused_NEG)
{
  luci::CircleSVDF node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, TransposeConv_validate)
{
  luci::CircleTransposeConv node;
  node.padding(luci::Padding::SAME);
  node.fusedActivationFunction(luci::FusedActFunc::RELU);
  EXPECT_TRUE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, TransposeConv_validate_padding_NEG)
{
  luci::CircleTransposeConv node;
  node.padding(luci::Padding::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}

TEST_F(CircleNodeSummaryBuilderTest, TransposeConv_validate_fused_NEG)
{
  luci::CircleTransposeConv node;
  node.fusedActivationFunction(luci::FusedActFunc::UNDEFINED);
  EXPECT_FALSE(mock_build(&node));
}
