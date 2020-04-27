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

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;

TEST(NodeExecution_Forward, s32)
{
  // Make pull-forward graph
  auto g = loco::make_graph();
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::S32);
  pull->shape({1});
  auto forward = g->nodes()->create<loco::Forward>();
  forward->input(pull);

  // Make and assign data to pull node
  auto pull_buf = make_buffer<int32_t, LexicalLayout>(Shape{1});
  pull_buf.at(Index{0}) = 42;
  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::annot_data(pull, std::move(pull_data));
  locomotiv::annot_domain(pull, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(forward);

  auto forward_data = locomotiv::annot_data(forward);
  ASSERT_NE(forward_data, nullptr);
  ASSERT_EQ(loco::DataType::S32, forward_data->dtype());
  ASSERT_EQ(Shape{1}, *(forward_data->shape()));
  ASSERT_EQ(pull_buf.at(Index{0}), forward_data->as_s32_bufptr()->at(Index{0}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(forward));
}

TEST(NodeExecution_Forward, f32)
{
  // Make pull-forward graph
  auto g = loco::make_graph();
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::FLOAT32);
  pull->shape({1});
  auto forward = g->nodes()->create<loco::Forward>();
  forward->input(pull);

  // Make and assign data to pull node
  auto pull_buf = make_buffer<float, LexicalLayout>(Shape{1});
  pull_buf.at(Index{0}) = 3.14f;
  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::annot_data(pull, std::move(pull_data));
  locomotiv::annot_domain(pull, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(forward);

  auto forward_data = locomotiv::annot_data(forward);
  ASSERT_NE(forward_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, forward_data->dtype());
  ASSERT_EQ(Shape{1}, *(forward_data->shape()));
  ASSERT_FLOAT_EQ(pull_buf.at(Index{0}), forward_data->as_f32_bufptr()->at(Index{0}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(forward));
}
