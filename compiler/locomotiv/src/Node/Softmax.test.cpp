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

TEST(NodeExecution_Softmax, f32)
{
  // Make pull-softmax graph
  auto g = loco::make_graph();
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::FLOAT32);
  pull->shape({2, 2});
  auto softmax = g->nodes()->create<loco::TensorSoftmax>();
  softmax->input(pull);
  softmax->axis(1);

  // Make and assign data to pull node
  auto pull_buf = make_buffer<float, LexicalLayout>({2, 2});
  pull_buf.at(Index{0, 0}) = 1.1f;
  pull_buf.at(Index{0, 1}) = 1.1f;
  pull_buf.at(Index{1, 0}) = 3.3f;
  pull_buf.at(Index{1, 1}) = 3.3f;
  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::annot_data(pull, std::move(pull_data));
  locomotiv::annot_domain(pull, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(softmax);

  auto kShape = Shape{2, 2};
  auto softmax_data = locomotiv::annot_data(softmax);
  ASSERT_NE(softmax_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, softmax_data->dtype());
  ASSERT_EQ(kShape, *(softmax_data->shape()));
  ASSERT_FLOAT_EQ(0.5f, softmax_data->as_f32_bufptr()->at(Index{0, 0}));
  ASSERT_FLOAT_EQ(0.5f, softmax_data->as_f32_bufptr()->at(Index{0, 1}));
  ASSERT_FLOAT_EQ(0.5f, softmax_data->as_f32_bufptr()->at(Index{1, 0}));
  ASSERT_FLOAT_EQ(0.5f, softmax_data->as_f32_bufptr()->at(Index{1, 1}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(softmax));
}
