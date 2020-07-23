/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "NodeExecution.h"

#include "locomotiv/NodeData.h"
#include "NodeDataImpl.h"
#include "NodeDomain.h"

#include <nncc/core/ADT/tensor/Index.h>
#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

TEST(NodeExecution_Fixed_Reduce_Mean, f32_0)
{
  // Make pull-TensorReduce(Mean) graph
  auto g = loco::make_graph();
  auto pull_input = g->nodes()->create<loco::Pull>();
  pull_input->dtype(loco::DataType::FLOAT32);
  pull_input->shape({1, 2, 2});
  auto reduce_node = g->nodes()->create<loco::TensorReduce>();
  reduce_node->input(pull_input);
  reduce_node->axes()->insert(0);
  reduce_node->axes()->insert(1);
  reduce_node->func(loco::ReduceFunc::Mean);

  // Make and assign data to pull node
  auto pull_input_buf = make_buffer<float, LexicalLayout>({1, 2, 2});
  pull_input_buf.at(Index{0, 0, 0}) = 1.1f;
  pull_input_buf.at(Index{0, 0, 1}) = 2.2f;
  pull_input_buf.at(Index{0, 1, 0}) = 5.5f;
  pull_input_buf.at(Index{0, 1, 1}) = 6.6f;
  auto pull_input_data = locomotiv::make_data(pull_input_buf);
  locomotiv::annot_data(pull_input, std::move(pull_input_data));
  locomotiv::annot_domain(pull_input, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(reduce_node);

  auto kShape = Shape{1, 1, 2};
  auto reduce_data = locomotiv::annot_data(reduce_node);
  ASSERT_NE(reduce_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, reduce_data->dtype());
  ASSERT_EQ(kShape, *(reduce_data->shape()));
  ASSERT_FLOAT_EQ(3.3f, reduce_data->as_f32_bufptr()->at(Index{0, 0, 0}));
  ASSERT_FLOAT_EQ(4.4f, reduce_data->as_f32_bufptr()->at(Index{0, 0, 1}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(reduce_node));
}

TEST(NodeExecution_Fixed_Reduce_Mean, f32_1)
{
  // Make pull-TensorReduce(Mean) graph
  auto g = loco::make_graph();
  auto pull_input = g->nodes()->create<loco::Pull>();
  pull_input->dtype(loco::DataType::FLOAT32);
  pull_input->shape({1, 2, 2});
  auto reduce_node = g->nodes()->create<loco::TensorReduce>();
  reduce_node->input(pull_input);
  reduce_node->axes()->insert(1);
  reduce_node->axes()->insert(2);
  reduce_node->func(loco::ReduceFunc::Mean);

  // Make and assign data to pull node
  auto pull_input_buf = make_buffer<float, LexicalLayout>({1, 2, 2});
  pull_input_buf.at(Index{0, 0, 0}) = 1.1f;
  pull_input_buf.at(Index{0, 0, 1}) = 2.2f;
  pull_input_buf.at(Index{0, 1, 0}) = 5.5f;
  pull_input_buf.at(Index{0, 1, 1}) = 6.6f;
  auto pull_input_data = locomotiv::make_data(pull_input_buf);
  locomotiv::annot_data(pull_input, std::move(pull_input_data));
  locomotiv::annot_domain(pull_input, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(reduce_node);

  auto kShape = Shape{1, 1, 1};
  auto reduce_data = locomotiv::annot_data(reduce_node);
  ASSERT_NE(reduce_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, reduce_data->dtype());
  ASSERT_EQ(kShape, *(reduce_data->shape()));
  ASSERT_FLOAT_EQ(3.85f, reduce_data->as_f32_bufptr()->at(Index{0, 0, 0}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(reduce_node));
}
