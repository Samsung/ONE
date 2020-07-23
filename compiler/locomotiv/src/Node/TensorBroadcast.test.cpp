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
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

TEST(NodeExecution_TensorBroadcast, f32)
{
  // Create a sample graph w/ TensorBroadcast
  auto g = loco::make_graph();
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::FLOAT32);
  pull->shape({1, 1});
  auto broadcast = g->nodes()->create<loco::TensorBroadcast>();
  broadcast->input(pull);
  broadcast->mapping()->dim(0) = 2;

  // Make and assign data to pull node
  auto pull_buf = make_buffer<float, LexicalLayout>(Shape{1, 1});
  pull_buf.at(Index{0, 0}) = -1.0f;

  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::annot_data(pull, std::move(pull_data));
  locomotiv::annot_domain(pull, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(broadcast);

  auto broadcast_data = locomotiv::annot_data(broadcast);
  ASSERT_NE(broadcast_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, broadcast_data->dtype());
  ASSERT_EQ((Shape{2, 1}), (*(broadcast_data->shape())));
  ASSERT_FLOAT_EQ(-1.0f, broadcast_data->as_f32_bufptr()->at(Index{0, 0}));
  ASSERT_FLOAT_EQ(-1.0f, broadcast_data->as_f32_bufptr()->at(Index{1, 0}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(broadcast));
}
