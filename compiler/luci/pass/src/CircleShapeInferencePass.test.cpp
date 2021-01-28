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

#include "luci/Pass/CircleShapeInferencePass.h"

#include <loco.h>

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

/**
 * This test is to check whether shape inference is done by topological order.
 *
 * When perm() of "transpose1" is changed from "old_perm" to "new_perm"
 * by some of luci/Pass like below diagram, shape_status of "transpose1" is
 * still VALID even the shape should be changed.
 * If "transpose2" is visited first before shape of "transpose1" is updated,
 * "transpose2" can reference the shape of "relu" which is not updated yet.
 * Then shape of "transpose2" becomes 3x5x5x1 and it causes an error at "conv2d".
 *
 * <Initial graph>
 *                                                4x1x1x3
 * [old_perm] ----------+              [filter] ----------+
 * (0,2,1,3)            |                                 |
 *                      |               [bias]  ----------+
 *                      |                                 |
 *  input  ------> [transpose1] ------> [relu] ------> [conv2d] ------>  output
 *         1x5x5x3              1x5x5x3        1x5x5x3          1x5x5x4
 *
 *
 * <Right after transformation>
 *                                                                        4x1x1x3
 * [new_perm] ----------+-----------------------------------+    [filter] ------+
 * (3,2,1,0)            |                                   |                   |
 *                      |                                   |      [bias] ------+
 *                      |                                   |                   |
 *  input  ------> [transpose1] ------> [relu] ------> [transpose2] ------> [conv2d] ------>  output
 *         1x5x5x3              1x5x5x3        1x5x5x3                 ?             1x5x5x4
 *
 *
 * <Expected result>
 *                                                                        4x1x1x3
 * [new_perm] ----------+-----------------------------------+    [filter] ------+
 * (3,2,1,0)            |                                   |                   |
 *                      |                                   |      [bias] ------+
 *                      |                                   |                   |
 *  input  ------> [transpose1] ------> [relu] ------> [transpose2] ------> [conv2d] ------>  output
 *         1x5x5x3              3x5x5x1        3x5x5x1              1x5x5x3          1x5x5x4
 *
 */
TEST(CircleShapeInferencePassTest, original_node_change)
{
  luci::CircleShapeInferencePass pass;
  auto g = loco::make_graph();

  // Have to be packed into lambda to check throw
  auto shape_inference_run = [&]() {
    while (pass.run(g.get()) == true)
      ;
  };

  // Create nodes to make relu traversed first
  auto input = g->nodes()->create<luci::CircleInput>();
  auto relu = g->nodes()->create<luci::CircleRelu>();
  auto old_perm = g->nodes()->create<luci::CircleConst>();
  auto transpose1 = g->nodes()->create<luci::CircleTranspose>();
  auto filter = g->nodes()->create<luci::CircleConst>();
  auto bias = g->nodes()->create<luci::CircleConst>();
  auto conv2d = g->nodes()->create<luci::CircleConv2D>();
  auto output = g->nodes()->create<luci::CircleOutput>();
  auto new_perm = g->nodes()->create<luci::CircleConst>();
  auto transpose2 = g->nodes()->create<luci::CircleTranspose>();

  // Build up initial graph
  auto graph_input = g->inputs()->create();
  graph_input->shape({1, 5, 5, 3});

  input->index(graph_input->index());
  input->shape({1, 5, 5, 3});
  input->shape_status(luci::ShapeStatus::VALID);

  old_perm->dtype(loco::DataType::S32);
  old_perm->size<loco::DataType::S32>(4);
  old_perm->shape({4});
  old_perm->at<loco::DataType::S32>(0) = 0;
  old_perm->at<loco::DataType::S32>(1) = 2;
  old_perm->at<loco::DataType::S32>(2) = 1;
  old_perm->at<loco::DataType::S32>(3) = 3;
  old_perm->shape_status(luci::ShapeStatus::VALID);

  transpose1->a(input);
  transpose1->perm(old_perm);

  relu->features(transpose1);

  filter->dtype(loco::DataType::FLOAT32);
  filter->size<loco::DataType::FLOAT32>(4 * 1 * 1 * 3);
  filter->shape({4, 1, 1, 3});
  filter->shape_status(luci::ShapeStatus::VALID);

  bias->dtype(loco::DataType::FLOAT32);
  bias->size<loco::DataType::FLOAT32>(4);
  bias->shape({4});
  bias->shape_status(luci::ShapeStatus::VALID);

  conv2d->input(relu);
  conv2d->filter(filter);
  conv2d->bias(bias);
  conv2d->padding(luci::Padding::VALID);
  conv2d->stride()->h(1);
  conv2d->stride()->w(1);
  conv2d->dilation()->h(1);
  conv2d->dilation()->w(1);

  output->from(conv2d);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
  graph_output->shape({1, 5, 5, 4});

  ASSERT_NO_THROW(shape_inference_run());

  // Transform graph
  new_perm->dtype(loco::DataType::S32);
  new_perm->size<loco::DataType::S32>(4);
  new_perm->shape({4});
  new_perm->at<loco::DataType::S32>(0) = 3;
  new_perm->at<loco::DataType::S32>(1) = 2;
  new_perm->at<loco::DataType::S32>(2) = 1;
  new_perm->at<loco::DataType::S32>(3) = 0;
  new_perm->shape_status(luci::ShapeStatus::VALID);

  transpose1->perm(new_perm);

  transpose2->a(relu);
  transpose2->perm(new_perm);

  conv2d->input(transpose2);

  ASSERT_NO_THROW(shape_inference_run());

  // Check result of shape inference is correct
  ASSERT_EQ(3, transpose1->dim(0).value());
  ASSERT_EQ(5, transpose1->dim(1).value());
  ASSERT_EQ(5, transpose1->dim(2).value());
  ASSERT_EQ(1, transpose1->dim(3).value());

  ASSERT_EQ(3, relu->dim(0).value());
  ASSERT_EQ(5, relu->dim(1).value());
  ASSERT_EQ(5, relu->dim(2).value());
  ASSERT_EQ(1, relu->dim(3).value());

  ASSERT_EQ(1, transpose2->dim(0).value());
  ASSERT_EQ(5, transpose2->dim(1).value());
  ASSERT_EQ(5, transpose2->dim(2).value());
  ASSERT_EQ(3, transpose2->dim(3).value());

  ASSERT_EQ(1, conv2d->dim(0).value());
  ASSERT_EQ(5, conv2d->dim(1).value());
  ASSERT_EQ(5, conv2d->dim(2).value());
  ASSERT_EQ(4, conv2d->dim(3).value());

  SUCCEED();
}

/**
 * This test is for checking when imported shape is wrong.
 *
 * Even "concat1" has wrong shape at first, correct shape should be inferred.
 *
 * <Initial graph>
 *
 *         1x1x1x1
 *  input1 ------+                 8x7x6x5
 *               +-----> [concat1] ------+
 *  input2 ------+       (axis=3)        |                  1x1x2x3
 *         1x1x1x2                       +------> [concat2] ------> output
 *                                       |        (axis=2)
 *                     1x1x1x3           |
 *  input3 ------------------------------+
 *
 *
 * <Expected result>
 *
 *         1x1x1x1
 *  input1 ------+                 1x1x1x3
 *               +-----> [concat1] ------+
 *  input2 ------+       (axis=3)        |                  1x1x2x3
 *         1x1x1x2                       +------> [concat2] ------> output
 *                                       |        (axis=2)
 *                     1x1x1x3           |
 *  input3 ------------------------------+
 */
TEST(CircleShapeInferencePassTest, wrong_imported_shape)
{
  luci::CircleShapeInferencePass pass;
  auto g = loco::make_graph();

  // Have to be packed into lambda to check throw
  auto shape_inference_run = [&]() {
    while (pass.run(g.get()) == true)
      ;
  };

  // Create nodes to make concat2 traversed first
  auto concat2 = g->nodes()->create<luci::CircleConcatenation>(2);
  auto concat1 = g->nodes()->create<luci::CircleConcatenation>(2);
  auto input1 = g->nodes()->create<luci::CircleInput>();
  auto input2 = g->nodes()->create<luci::CircleInput>();
  auto input3 = g->nodes()->create<luci::CircleInput>();

  // Build up initial graph
  auto graph_input1 = g->inputs()->create();
  auto graph_input2 = g->inputs()->create();
  auto graph_input3 = g->inputs()->create();
  graph_input1->shape({1, 1, 1, 1});
  graph_input2->shape({1, 1, 1, 2});
  graph_input2->shape({1, 1, 1, 3});

  input1->index(graph_input1->index());
  input1->shape({1, 1, 1, 1});
  input1->shape_status(luci::ShapeStatus::VALID);

  input2->index(graph_input2->index());
  input2->shape({1, 1, 1, 2});
  input2->shape_status(luci::ShapeStatus::VALID);

  input3->index(graph_input3->index());
  input3->shape({1, 1, 1, 3});
  input3->shape_status(luci::ShapeStatus::VALID);

  concat1->values(0, input1);
  concat1->values(1, input2);
  concat1->axis(3);
  concat1->shape({8, 7, 6, 5}); // Intentionally set wrong shape
  concat1->shape_status(luci::ShapeStatus::VALID);

  concat2->values(0, concat1);
  concat2->values(1, input3);
  concat2->axis(2);

  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(concat2);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
  graph_output->shape({1, 1, 2, 3});

  ASSERT_NO_THROW(shape_inference_run());

  // Check result of shape inference is correct
  ASSERT_EQ(1, concat1->dim(0).value());
  ASSERT_EQ(1, concat1->dim(1).value());
  ASSERT_EQ(1, concat1->dim(2).value());
  ASSERT_EQ(3, concat1->dim(3).value());

  ASSERT_EQ(1, concat2->dim(0).value());
  ASSERT_EQ(1, concat2->dim(1).value());
  ASSERT_EQ(2, concat2->dim(2).value());
  ASSERT_EQ(3, concat2->dim(3).value());

  SUCCEED();
}

/**
 * This test is for checking that virtual operations which is not used for graph output
 * but shape should be exported.
 *
 * Although "split_out2" is not used for graph output, shape should be inferenced.
 *
 * <Initial graph>
 *
 *
 *          1x6                +----> [split_out1] ----> output
 *  input ------> [split] -----+
 *             (split_dim=1)   +----> [split_out2]
 *             (num_split=2)
 *
 *
 * <Expected result>
 *                               1x3                1x3
 *          1x6                +----> [split_out1] ----> output
 *  input ------> [split] -----+
 *             (split_dim=1)   +----> [split_out2]
 *             (num_split=2)     1x3
 */
TEST(CircleShapeInferencePassTest, not_used_virtual_op)
{
  luci::CircleShapeInferencePass pass;
  auto g = loco::make_graph();

  // Have to be packed into lambda to check throw
  auto shape_inference_run = [&]() {
    while (pass.run(g.get()) == true)
      ;
  };

  // Create nodes
  auto input = g->nodes()->create<luci::CircleInput>();
  auto split = g->nodes()->create<luci::CircleSplit>();
  auto split_out1 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_out2 = g->nodes()->create<luci::CircleSplitOut>();
  auto split_dim = g->nodes()->create<luci::CircleConst>();

  // Build up initial graph
  auto graph_input1 = g->inputs()->create();
  graph_input1->shape({1, 6});

  input->index(graph_input1->index());
  input->shape({1, 6});
  input->shape_status(luci::ShapeStatus::VALID);

  split_dim->dtype(loco::DataType::S32);
  split_dim->size<loco::DataType::S32>(1);
  split_dim->shape({1});
  split_dim->at<loco::DataType::S32>(0) = 1;
  split_dim->shape_status(luci::ShapeStatus::VALID);

  split->split_dim(split_dim);
  split->input(input);
  split->num_split(2);

  split_out1->input(split);
  split_out1->index(0);

  split_out2->input(split);
  split_out2->index(1);

  auto output = g->nodes()->create<luci::CircleOutput>();
  output->from(split_out1);
  auto graph_output = g->outputs()->create();
  output->index(graph_output->index());
  graph_output->shape({1, 3});

  ASSERT_NO_THROW(shape_inference_run());

  // Check result of shape inference is correct
  ASSERT_EQ(1, split_out1->dim(0).value());
  ASSERT_EQ(3, split_out1->dim(1).value());

  ASSERT_EQ(1, split_out2->dim(0).value());
  ASSERT_EQ(3, split_out2->dim(1).value());

  SUCCEED();
}
