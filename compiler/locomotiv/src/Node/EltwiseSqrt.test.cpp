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

#include <cmath>
#include <limits>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

TEST(NodeExecution_EltwiseSqrt, f32)
{
  // Make Pull-EltwiseSqrt graph
  auto g = loco::make_graph();
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::FLOAT32);
  pull->shape({4});
  auto sqrt = g->nodes()->create<loco::EltwiseSqrt>();
  sqrt->input(pull);

  // Make and assign data to Pull node
  auto pull_buf = make_buffer<float, LexicalLayout>(Shape{4});
  pull_buf.at(Index{0}) = 4.0f;
  pull_buf.at(Index{1}) = 9.0f;
  pull_buf.at(Index{2}) = 0.0f;
  pull_buf.at(Index{3}) = -1.0f;
  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::annot_data(pull, std::move(pull_data));
  locomotiv::annot_domain(pull, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(sqrt);

  auto sqrt_data = locomotiv::annot_data(sqrt);
  ASSERT_NE(sqrt_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, sqrt_data->dtype());
  ASSERT_EQ(Shape{4}, *(sqrt_data->shape()));
  ASSERT_FLOAT_EQ(2.0f, sqrt_data->as_f32_bufptr()->at(Index{0}));
  ASSERT_FLOAT_EQ(3.0f, sqrt_data->as_f32_bufptr()->at(Index{1}));
  ASSERT_FLOAT_EQ(0.0f, sqrt_data->as_f32_bufptr()->at(Index{2}));
  ASSERT_TRUE(std::isnan(sqrt_data->as_f32_bufptr()->at(Index{3})));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(sqrt));
}
