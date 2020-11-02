/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FusePreActivationBatchNormPassInternal.h"

#include <luci/IR/CircleNodes.h>

#include <math.h>
#include <vector>

#include <gtest/gtest.h>

namespace
{

/**
 *  Simple graph for test
 *
 *  BEFORE
 *
 *   [Conv] W + bias
 *        \     [Conv]
 *         \     /
 *          [Add]
 *         /    \
 *        /    [Mul]   gamma
 *       |       |
 *       |   [Add+Relu]   beta
 *       |       |
 *       |     [Conv]   W + bias
 *        \     /
 *         [Add]
 *
 *  AFTER
 *
 *   [Conv] W + (bias + beta/gamma)
 *        \     [Conv]
 *         \     /
 *          [Add]
 *         /    \
 *       |     [Relu]
 *       |       |
 *       |     [Conv]  (gamma * W) + (bias - beta/gamma)
 *        \     /
 *         [Add]
 *
 */
class SimpleGraph
{
public:
  SimpleGraph()
  {
    pred_conv = g.nodes()->create<luci::CircleConv2D>();
    pred_conv_filter = g.nodes()->create<luci::CircleConst>();
    pred_conv_bias = g.nodes()->create<luci::CircleConst>();
    pred_conv2 = g.nodes()->create<luci::CircleConv2D>();
    pred_add = g.nodes()->create<luci::CircleAdd>();
    mul = g.nodes()->create<luci::CircleMul>();
    mul_gamma = g.nodes()->create<luci::CircleConst>();
    add = g.nodes()->create<luci::CircleAdd>();
    add_beta = g.nodes()->create<luci::CircleConst>();
    conv = g.nodes()->create<luci::CircleConv2D>();
    conv_filter = g.nodes()->create<luci::CircleConst>();
    conv_bias = g.nodes()->create<luci::CircleConst>();
    succ_add = g.nodes()->create<luci::CircleAdd>();

    pred_conv->dtype(loco::DataType::FLOAT32);
    pred_conv_filter->dtype(loco::DataType::FLOAT32);
    pred_conv_bias->dtype(loco::DataType::FLOAT32);
    pred_conv2->dtype(loco::DataType::FLOAT32);
    pred_add->dtype(loco::DataType::FLOAT32);
    mul->dtype(loco::DataType::FLOAT32);
    mul_gamma->dtype(loco::DataType::FLOAT32);
    add->dtype(loco::DataType::FLOAT32);
    add->fusedActivationFunction(luci::FusedActFunc::RELU);
    add_beta->dtype(loco::DataType::FLOAT32);
    conv->dtype(loco::DataType::FLOAT32);
    conv_filter->dtype(loco::DataType::FLOAT32);
    conv_bias->dtype(loco::DataType::FLOAT32);
    succ_add->dtype(loco::DataType::FLOAT32);

    pred_conv->shape({1, 4, 4, 16});
    pred_conv_filter->shape({16, 1, 1, 16});
    pred_conv_bias->shape({16});
    pred_conv2->shape({1, 4, 4, 16});
    pred_add->shape({1, 4, 4, 16});
    mul->shape({1, 4, 4, 16});
    mul_gamma->shape({16});
    add->shape({1, 4, 4, 16});
    add_beta->shape({16});
    conv->shape({1, 4, 4, 16});
    conv_filter->shape({16, 1, 1, 16});
    conv_bias->shape({16});
    succ_add->shape({1, 4, 4, 16});

    pred_conv->filter(pred_conv_filter);
    pred_conv->bias(pred_conv_bias);
    pred_add->x(pred_conv);
    pred_add->y(pred_conv2);
    mul->x(pred_add);
    mul->y(mul_gamma);
    add->x(mul);
    add->y(add_beta);
    conv->input(add);
    conv->filter(conv_filter);
    conv->bias(conv_bias);
    succ_add->x(pred_add);
    succ_add->y(conv);

    uint32_t channel_size = 16;
    uint32_t out_size = 16;
    add_beta->size<loco::DataType::FLOAT32>(channel_size);
    mul_gamma->size<loco::DataType::FLOAT32>(channel_size);
    conv_filter->size<loco::DataType::FLOAT32>(channel_size * out_size);
    conv_bias->size<loco::DataType::FLOAT32>(out_size);
    pred_conv_bias->size<loco::DataType::FLOAT32>(channel_size);
    for (uint32_t i = 0; i < channel_size; i++)
    {
      add_beta->at<loco::DataType::FLOAT32>(i) = i;
      mul_gamma->at<loco::DataType::FLOAT32>(i) = i;
      pred_conv_bias->at<loco::DataType::FLOAT32>(i) = i;
      conv_bias->at<loco::DataType::FLOAT32>(i) = i;
      for (uint32_t j = 0; j < out_size; j++)
      {
        conv_filter->at<loco::DataType::FLOAT32>(i * out_size + j) = i * out_size + j;
      }
    }
  }

public:
  loco::Graph g;
  luci::CircleConv2D *pred_conv;
  luci::CircleConst *pred_conv_filter;
  luci::CircleConst *pred_conv_bias;
  luci::CircleConv2D *pred_conv2;
  luci::CircleAdd *pred_add;
  luci::CircleMul *mul;
  luci::CircleConst *mul_gamma;
  luci::CircleAdd *add;
  luci::CircleConst *add_beta;
  luci::CircleConv2D *conv;
  luci::CircleConst *conv_filter;
  luci::CircleConst *conv_bias;
  luci::CircleAdd *succ_add;
};

} // namespace

TEST(FusePreActivationBatchNorm, swap_mul_add)
{
  SimpleGraph g;
  int channel_size = 16;
  std::vector<luci::CircleMul *> mul_list;
  std::vector<luci::CircleAdd *> add_list;

  EXPECT_TRUE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_EQ(1, mul_list.size());
  EXPECT_EQ(1, add_list.size());
  EXPECT_EQ(g.mul, mul_list[0]);
  EXPECT_EQ(g.add, add_list[0]);

  for (uint32_t i = 0; i < channel_size; ++i)
  {
    float beta = g.add_beta->at<loco::DataType::FLOAT32>(i);
    float gamma = g.mul_gamma->at<loco::DataType::FLOAT32>(i);
    EXPECT_FLOAT_EQ(1.0, beta);
    EXPECT_FLOAT_EQ(i, gamma);
  }

  auto relu = static_cast<luci::CircleRelu *>(g.conv->input());
  EXPECT_TRUE(relu != nullptr);

  EXPECT_EQ(g.mul, relu->features());
  EXPECT_EQ(g.add, g.mul->x());
  EXPECT_EQ(luci::FusedActFunc::NONE, g.add->fusedActivationFunction());
  EXPECT_EQ(g.pred_add, g.add->x());
}

TEST(FusePreActivationBatchNorm, swap_mul_add_NEG)
{
  SimpleGraph g;
  std::vector<luci::CircleMul *> mul_list;
  std::vector<luci::CircleAdd *> add_list;

  // Add does not have fused activation
  g.add->fusedActivationFunction(luci::FusedActFunc::NONE);
  EXPECT_FALSE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_EQ(0, mul_list.size());
  EXPECT_EQ(0, add_list.size());
  g.add->fusedActivationFunction(luci::FusedActFunc::RELU);

  // Add is element-wise
  g.add_beta->shape({1, 4, 4, 16});
  EXPECT_FALSE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_EQ(0, mul_list.size());
  EXPECT_EQ(0, add_list.size());
  g.add_beta->shape({16});

  // Mul is element-wise
  g.mul_gamma->shape({1, 4, 4, 16});
  EXPECT_FALSE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_EQ(0, mul_list.size());
  EXPECT_EQ(0, add_list.size());
  g.mul_gamma->shape({16});

  // Negative gamma
  g.mul_gamma->at<loco::DataType::FLOAT32>(0) = -10;
  EXPECT_FALSE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_EQ(0, mul_list.size());
  EXPECT_EQ(0, add_list.size());
}

TEST(FusePreActivationBatchNorm, fuse_mul_with_conv)
{
  SimpleGraph g;
  int channel_size = 16;
  int out_size = 16;
  std::vector<luci::CircleMul *> mul_list;
  std::vector<luci::CircleAdd *> add_list;

  EXPECT_TRUE(luci::swap_mul_add(g.add, mul_list, add_list));

  EXPECT_TRUE(luci::fuse_mul_with_conv(g.mul));
  for (uint32_t o = 0; o < out_size; o++)
  {
    for (uint32_t c = 0; c < channel_size; c++)
    {
      auto val = g.conv_filter->at<loco::DataType::FLOAT32>(o * channel_size + c);
      auto gamma = g.mul_gamma->at<loco::DataType::FLOAT32>(c);
      EXPECT_FLOAT_EQ((o * channel_size + c) * gamma, val);
    }
  }

  auto relu = static_cast<luci::CircleRelu *>(g.conv->input());
  EXPECT_EQ(g.add, relu->features());
}

TEST(FusePreActivationBatchNorm, fuse_mul_with_conv_NEG)
{
  SimpleGraph g;
  std::vector<luci::CircleMul *> mul_list;
  std::vector<luci::CircleAdd *> add_list;

  EXPECT_TRUE(luci::swap_mul_add(g.add, mul_list, add_list));

  // Non-conv layer uses the output of relu
  auto relu = static_cast<luci::CircleRelu *>(g.conv->input());
  auto fc = g.g.nodes()->create<luci::CircleFullyConnected>();
  fc->input(relu);
  EXPECT_FALSE(luci::fuse_mul_with_conv(g.mul));
}

TEST(FusePreActivationBatchNorm, fuse_add_with_conv)
{
  SimpleGraph g;
  int channel_size = 16;
  std::vector<luci::CircleMul *> mul_list;
  std::vector<luci::CircleAdd *> add_list;
  std::vector<luci::CircleSub *> sub_list;

  EXPECT_TRUE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_TRUE(luci::fuse_mul_with_conv(g.mul));
  EXPECT_TRUE(luci::fuse_add_with_conv(g.add, sub_list));

  for (uint32_t c = 0; c < channel_size; c++)
  {
    auto bias = g.pred_conv_bias->at<loco::DataType::FLOAT32>(c);
    EXPECT_FLOAT_EQ(c + 1.0, bias);
  }

  auto relu = static_cast<luci::CircleRelu *>(g.conv->input());
  EXPECT_EQ(relu, g.conv->input());
  EXPECT_EQ(g.pred_add, relu->features());
  EXPECT_EQ(g.pred_conv, g.pred_add->x());

  auto sub = static_cast<luci::CircleSub *>(sub_list[0]);
  EXPECT_EQ(sub, g.succ_add->x());
  EXPECT_EQ(g.pred_add, sub->x());
  for (uint32_t c = 0; c < channel_size; c++)
  {
    auto beta = static_cast<luci::CircleConst *>(sub->y());
    EXPECT_FLOAT_EQ(1.0, beta->at<loco::DataType::FLOAT32>(c));
  }
}

TEST(FusePreActivationBatchNorm, fuse_add_with_conv_NEG)
{
  SimpleGraph g;
  int channel_size = 16;
  std::vector<luci::CircleMul *> mul_list;
  std::vector<luci::CircleAdd *> add_list;
  std::vector<luci::CircleSub *> sub_list;

  EXPECT_TRUE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_TRUE(luci::fuse_mul_with_conv(g.mul));

  // No conv layer to fuse add
  auto fc1 = g.g.nodes()->create<luci::CircleFullyConnected>();
  auto fc2 = g.g.nodes()->create<luci::CircleFullyConnected>();
  g.pred_add->x(fc1);
  g.pred_add->y(fc2);
  EXPECT_FALSE(luci::fuse_add_with_conv(g.add, sub_list));
  EXPECT_EQ(0, sub_list.size());
}

TEST(FusePreActivationBatchNorm, fuse_sub_with_conv)
{
  SimpleGraph g;
  int channel_size = 16;
  std::vector<luci::CircleMul *> mul_list;
  std::vector<luci::CircleAdd *> add_list;
  std::vector<luci::CircleSub *> sub_list;

  EXPECT_TRUE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_TRUE(luci::fuse_mul_with_conv(g.mul));
  EXPECT_TRUE(luci::fuse_add_with_conv(g.add, sub_list));
  EXPECT_TRUE(luci::fuse_sub_with_conv(sub_list[0]));

  for (uint32_t c = 0; c < channel_size; c++)
  {
    auto bias = g.conv_bias->at<loco::DataType::FLOAT32>(c);
    EXPECT_FLOAT_EQ(c - 1.0, bias);
  }

  EXPECT_EQ(g.pred_add, g.succ_add->x());
  EXPECT_EQ(g.conv, g.succ_add->y());
}

TEST(FusePreActivationBatchNorm, fuse_sub_with_conv_NEG)
{
  SimpleGraph g;
  int channel_size = 16;
  std::vector<luci::CircleMul *> mul_list;
  std::vector<luci::CircleAdd *> add_list;
  std::vector<luci::CircleSub *> sub_list;

  EXPECT_TRUE(luci::swap_mul_add(g.add, mul_list, add_list));
  EXPECT_TRUE(luci::fuse_mul_with_conv(g.mul));
  EXPECT_TRUE(luci::fuse_add_with_conv(g.add, sub_list));

  // No suitable pattern (relu was inserted between add and conv)
  auto relu = g.g.nodes()->create<luci::CircleRelu>();
  relu->features(g.conv);
  g.succ_add->y(relu);
  EXPECT_FALSE(luci::fuse_sub_with_conv(sub_list[0]));
  g.succ_add->y(g.conv);
  relu->drop();

  // No suitable pattern (add was replaced with mul)
  auto mul = g.g.nodes()->create<luci::CircleMul>();
  mul->x(sub_list[0]);
  mul->y(g.conv);
  g.succ_add->drop();
  EXPECT_FALSE(luci::fuse_sub_with_conv(sub_list[0]));
}
