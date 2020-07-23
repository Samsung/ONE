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

TEST(NodeExecution_TensorConcat, f32)
{
  // Make (pull, pull)-concat graph
  auto g = loco::make_graph();
  auto pull_l = g->nodes()->create<loco::Pull>();
  pull_l->dtype(loco::DataType::FLOAT32);
  pull_l->shape({1, 2});
  auto pull_r = g->nodes()->create<loco::Pull>();
  pull_r->dtype(loco::DataType::FLOAT32);
  pull_r->shape({1, 2});
  auto tconcat = g->nodes()->create<loco::TensorConcat>();
  tconcat->lhs(pull_l);
  tconcat->rhs(pull_r);
  tconcat->axis(0);

  // Make and assign data to pull node
  auto pull_l_buf = make_buffer<float, LexicalLayout>(Shape{1, 2});
  pull_l_buf.at(Index{0, 0}) = -1.0f;
  pull_l_buf.at(Index{0, 1}) = -2.0f;
  auto pull_r_buf = make_buffer<float, LexicalLayout>(Shape{1, 2});
  pull_r_buf.at(Index{0, 0}) = 3.0f;
  pull_r_buf.at(Index{0, 1}) = 4.0f;

  auto pull_l_data = locomotiv::make_data(pull_l_buf);
  locomotiv::annot_data(pull_l, std::move(pull_l_data));
  locomotiv::annot_domain(pull_l, loco::Domain::Tensor);
  auto pull_r_data = locomotiv::make_data(pull_r_buf);
  locomotiv::annot_data(pull_r, std::move(pull_r_data));
  locomotiv::annot_domain(pull_r, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(tconcat);

  auto concat_data = locomotiv::annot_data(tconcat);
  ASSERT_NE(concat_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, concat_data->dtype());
  ASSERT_EQ((Shape{2, 2}), (*(concat_data->shape())));
  ASSERT_FLOAT_EQ(-1.0f, concat_data->as_f32_bufptr()->at(Index{0, 0}));
  ASSERT_FLOAT_EQ(-2.0f, concat_data->as_f32_bufptr()->at(Index{0, 1}));
  ASSERT_FLOAT_EQ(3.0f, concat_data->as_f32_bufptr()->at(Index{1, 0}));
  ASSERT_FLOAT_EQ(4.0f, concat_data->as_f32_bufptr()->at(Index{1, 1}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(tconcat));
}

TEST(NodeExecution_TensorConcat, f32_2)
{
  // Make (pull, pull)-concat graph
  auto g = loco::make_graph();
  auto pull_l = g->nodes()->create<loco::Pull>();
  pull_l->dtype(loco::DataType::FLOAT32);
  pull_l->shape({1, 2});
  auto pull_r = g->nodes()->create<loco::Pull>();
  pull_r->dtype(loco::DataType::FLOAT32);
  pull_r->shape({3, 2});
  auto tconcat = g->nodes()->create<loco::TensorConcat>();
  tconcat->lhs(pull_l);
  tconcat->rhs(pull_r);
  tconcat->axis(0);

  // Make and assign data to pull node
  auto pull_l_buf = make_buffer<float, LexicalLayout>(Shape{1, 2});
  pull_l_buf.at(Index{0, 0}) = -1.0f;
  pull_l_buf.at(Index{0, 1}) = -2.0f;
  auto pull_r_buf = make_buffer<float, LexicalLayout>(Shape{3, 2});
  pull_r_buf.at(Index{0, 0}) = 3.0f;
  pull_r_buf.at(Index{0, 1}) = 4.0f;
  pull_r_buf.at(Index{1, 0}) = -3.0f;
  pull_r_buf.at(Index{1, 1}) = -4.0f;
  pull_r_buf.at(Index{2, 0}) = 5.0f;
  pull_r_buf.at(Index{2, 1}) = 6.0f;

  auto pull_l_data = locomotiv::make_data(pull_l_buf);
  locomotiv::annot_data(pull_l, std::move(pull_l_data));
  locomotiv::annot_domain(pull_l, loco::Domain::Tensor);
  auto pull_r_data = locomotiv::make_data(pull_r_buf);
  locomotiv::annot_data(pull_r, std::move(pull_r_data));
  locomotiv::annot_domain(pull_r, loco::Domain::Tensor);

  locomotiv::NodeExecution::get().run(tconcat);

  auto concat_data = locomotiv::annot_data(tconcat);
  ASSERT_NE(concat_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, concat_data->dtype());
  ASSERT_EQ((Shape{4, 2}), (*(concat_data->shape())));
  ASSERT_FLOAT_EQ(-1.0f, concat_data->as_f32_bufptr()->at(Index{0, 0}));
  ASSERT_FLOAT_EQ(-2.0f, concat_data->as_f32_bufptr()->at(Index{0, 1}));
  ASSERT_FLOAT_EQ(3.0f, concat_data->as_f32_bufptr()->at(Index{1, 0}));
  ASSERT_FLOAT_EQ(4.0f, concat_data->as_f32_bufptr()->at(Index{1, 1}));
  ASSERT_FLOAT_EQ(-3.0f, concat_data->as_f32_bufptr()->at(Index{2, 0}));
  ASSERT_FLOAT_EQ(-4.0f, concat_data->as_f32_bufptr()->at(Index{2, 1}));
  ASSERT_FLOAT_EQ(5.0f, concat_data->as_f32_bufptr()->at(Index{3, 0}));
  ASSERT_FLOAT_EQ(6.0f, concat_data->as_f32_bufptr()->at(Index{3, 1}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(tconcat));
}
