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
#include "luci/Pass/SubstituteStridedSliceToReshapePass.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

luci::CircleConst *build_rank1_const(loco::Graph *graph, const std::vector<uint32_t> values)
{
  auto const_node = graph->nodes()->create<luci::CircleConst>();
  const_node->dtype(loco::DataType::S32);
  const_node->size<loco::DataType::S32>(values.size());
  const_node->shape_status(luci::ShapeStatus::VALID);
  const_node->rank(1);
  const_node->dim(0) = values.size();

  for (int32_t i = 0; i < values.size(); i++)
  {
    const_node->at<loco::DataType::S32>(i) = values.at(i);
  }

  return const_node;
}

class SubstituteStridedSliceToReshapeTest : public ::testing::Test
{
public:
  SubstituteStridedSliceToReshapeTest() {}

  void buildGraph(const std::initializer_list<uint32_t> input_shape,
                  const std::initializer_list<uint32_t> begin_vals,
                  const std::initializer_list<uint32_t> end_vals,
                  const std::initializer_list<uint32_t> strides_vals, int32_t begin_mask,
                  int32_t end_mask, int32_t ellipsis_mask, int32_t new_axis_mask,
                  int32_t shrink_axis_mask)
  {
    // Input node
    input = g.nodes()->create<luci::CircleInput>();
    {
      auto graph_input = g.inputs()->create();
      input->index(graph_input->index());
      input->shape_status(luci::ShapeStatus::VALID);
      input->rank(input_shape.size());
      input->shape(input_shape);
      input->name("input");
    }

    // StridedSlice node
    auto ss_node = g.nodes()->create<luci::CircleStridedSlice>();
    {
      auto *graph = &g;
      auto build_attr = [&graph](const std::string &name,
                                 const std::initializer_list<uint32_t> vals) {
        auto node = build_rank1_const(graph, vals);
        node->name(name);

        return node;
      };

      ss_node->input(input);
      auto begin = build_attr("begin", begin_vals);
      auto end = build_attr("end", end_vals);
      auto strides = build_attr("strides", strides_vals);

      ss_node->begin(begin);
      ss_node->end(end);
      ss_node->strides(strides);

      ss_node->begin_mask(begin_mask);
      ss_node->end_mask(end_mask);
      ss_node->ellipsis_mask(ellipsis_mask);
      ss_node->new_axis_mask(new_axis_mask);
      ss_node->shrink_axis_mask(shrink_axis_mask);
    }

    // Output node
    output = g.nodes()->create<luci::CircleOutput>();
    output->from(ss_node);
    auto graph_output = g.outputs()->create();
    output->index(graph_output->index());
    output->name("output");
  }

  void assert_not_converted()
  {
    luci::SubstituteStridedSliceToReshapePass pass;
    while (pass.run(&g))
      ;

    auto reshape_node = dynamic_cast<luci::CircleReshape *>(output->from());
    ASSERT_TRUE(reshape_node == nullptr);

    auto strided_slice_node = dynamic_cast<luci::CircleStridedSlice *>(output->from());
    ASSERT_TRUE(strided_slice_node != nullptr);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleOutput *output = nullptr;
};

} // namespace

TEST(SubstituteStridedSliceToReshapePassTest, name)
{
  luci::SubstituteStridedSliceToReshapePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST_F(SubstituteStridedSliceToReshapeTest, simple_case)
{
  buildGraph({1, 1, 5, 1, 9}, // input shape
             {0, 0, 0, 0, 0}, // begin
             {1, 1, 5, 1, 9}, // end
             {1, 1, 1, 1, 1}, // strides
             0,               // begin mask
             0,               // end mask
             0,               // ellipsis axis mask
             0,               // new axis mask
             0b01001          // shrink axis mask, 0th and 3rd dim will be shrunk
  );

  luci::SubstituteStridedSliceToReshapePass pass;
  while (pass.run(&g))
    ;

  auto reshape_node = dynamic_cast<luci::CircleReshape *>(output->from());
  ASSERT_TRUE(reshape_node != nullptr);

  auto new_shape = loco::must_cast<luci::CircleConst *>(reshape_node->shape());
  ASSERT_EQ(new_shape->rank(), 1);
  ASSERT_EQ(new_shape->dim(0).value(), 3);
  ASSERT_EQ(new_shape->at<loco::DataType::S32>(0), 1);
  ASSERT_EQ(new_shape->at<loco::DataType::S32>(1), 5);
  ASSERT_EQ(new_shape->at<loco::DataType::S32>(2), 9);
}

TEST_F(SubstituteStridedSliceToReshapeTest, with_begin_end_mask)
{
  buildGraph({5, 1, 9}, // input shape
             {0, 0, 5}, // begin
             {3, 1, 9}, // end
             {1, 1, 1}, // strides
             0b100,     // begin mask
             0b001,     // end mask
             0,         // ellipsis axis mask
             0,         // new axis mask
             0b010      // shrink axis mask, 0th and 3rd dim will be shrunk
  );

  luci::SubstituteStridedSliceToReshapePass pass;
  while (pass.run(&g))
    ;

  auto reshape_node = dynamic_cast<luci::CircleReshape *>(output->from());
  ASSERT_TRUE(reshape_node != nullptr);

  auto new_shape = loco::must_cast<luci::CircleConst *>(reshape_node->shape());
  ASSERT_EQ(new_shape->rank(), 1);
  ASSERT_EQ(new_shape->dim(0).value(), 2);
  ASSERT_EQ(new_shape->at<loco::DataType::S32>(0), 5);
  ASSERT_EQ(new_shape->at<loco::DataType::S32>(1), 9);
}

TEST_F(SubstituteStridedSliceToReshapeTest, with_large_end_mask)
{
  buildGraph({5, 1, 9},       // input shape
             {0, 0, 0},       // begin
             {100, 100, 100}, // large end
             {1, 1, 1},       // strides
             0,               // begin mask
             0,               // end mask
             0,               // ellipsis axis mask
             0,               // new axis mask
             0b010            // shrink axis mask, 0th and 3rd dim will be shrunk
  );

  luci::SubstituteStridedSliceToReshapePass pass;
  while (pass.run(&g))
    ;

  auto reshape_node = dynamic_cast<luci::CircleReshape *>(output->from());
  ASSERT_TRUE(reshape_node != nullptr);

  auto new_shape = loco::must_cast<luci::CircleConst *>(reshape_node->shape());
  ASSERT_EQ(new_shape->rank(), 1);
  ASSERT_EQ(new_shape->dim(0).value(), 2);
  ASSERT_EQ(new_shape->at<loco::DataType::S32>(0), 5);
  ASSERT_EQ(new_shape->at<loco::DataType::S32>(1), 9);
}

TEST_F(SubstituteStridedSliceToReshapeTest, not_matching_begin_index_NEG)
{
  buildGraph({1, 3, 5, 7}, // input shape
             {0, 0, 2, 0}, // begin[2] does not start from 0
             {1, 3, 5, 7}, // end
             {1, 1, 1, 1}, // strides
             0,            // begin mask
             0,            // end mask
             0,            // ellipsis axis mask
             0,            // new axis mask
             0b0001        // shrink axis mask
  );

  assert_not_converted();
}

TEST_F(SubstituteStridedSliceToReshapeTest, not_matching_end_index_NEG)
{
  buildGraph({1, 3, 5, 7}, // input shape
             {0, 0, 0, 0}, // begin
             {1, 3, 3, 7}, // end[2] does not meet condition
             {1, 1, 1, 1}, // strides
             0,            // begin mask
             0,            // end mask
             0,            // ellipsis axis mask
             0,            // new axis mask
             0b0001        // shrink axis mask
  );

  assert_not_converted();
}

TEST_F(SubstituteStridedSliceToReshapeTest, not_matching_strides_NEG)
{
  buildGraph({1, 3, 5, 7}, // input shape
             {0, 0, 0, 0}, // begin
             {1, 3, 5, 7}, // end
             {1, 1, 2, 1}, // strides[2] does not meet condition
             0,            // begin mask
             0,            // end mask
             0,            // ellipsis axis mask
             0,            // new axis mask
             0b0001        // shrink axis mask
  );

  assert_not_converted();
}

TEST_F(SubstituteStridedSliceToReshapeTest, not_matching_shrink_axis_mask_NEG)
{
  buildGraph({1, 3, 5, 7}, // input shape
             {0, 0, 0, 0}, // begin
             {1, 3, 5, 7}, // end
             {1, 1, 1, 1}, // strides
             0,            // begin mask
             0,            // end mask
             0,            // ellipsis axis mask
             0,            // new axis mask
             0b0101        // shrink axis mask[1] does not meet condition
  );

  assert_not_converted();
}
