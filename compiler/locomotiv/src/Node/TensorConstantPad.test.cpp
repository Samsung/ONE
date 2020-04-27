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
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

TEST(NodeExecution_Pad, tensor_constant_pad_4_dim)
{
  auto g = loco::make_graph();

  auto inputTensor = g->nodes()->create<loco::Pull>();
  inputTensor->dtype(loco::DataType::FLOAT32);
  inputTensor->shape({1, 2, 2, 1});
  auto inputTensor_buf = make_buffer<float, LexicalLayout>(Shape{1, 2, 2, 1});
  inputTensor_buf.at(Index{0, 0, 0, 0}) = 1.0f;
  inputTensor_buf.at(Index{0, 0, 1, 0}) = 2.0f;
  inputTensor_buf.at(Index{0, 1, 0, 0}) = 3.0f;
  inputTensor_buf.at(Index{0, 1, 1, 0}) = 4.0f;
  auto inputTensor_data = locomotiv::make_data(inputTensor_buf);
  locomotiv::annot_data(inputTensor, std::move(inputTensor_data));
  locomotiv::annot_domain(inputTensor, loco::Domain::Tensor);

  auto constant = g->nodes()->create<loco::ConstGen>();
  constant->dtype(loco::DataType::FLOAT32);
  constant->shape({1});
  auto constant_buf = make_buffer<float, LexicalLayout>(Shape{1});
  constant_buf.at(Index{0}) = 0.0f;
  auto constant_data = locomotiv::make_data(constant_buf);
  locomotiv::annot_data(constant, std::move(constant_data));
  locomotiv::annot_domain(constant, loco::Domain::Tensor);

  auto pad = g->nodes()->create<loco::TensorConstantPad>();
  pad->input(inputTensor);
  pad->constant(constant);

  auto padding = pad->padding();
  padding->rank(4);
  padding->front(0) = 0;
  padding->back(0) = 0;
  padding->front(1) = 3;
  padding->back(1) = 1;
  padding->front(2) = 1;
  padding->back(2) = 1;
  padding->front(3) = 0;
  padding->back(3) = 0;

  locomotiv::NodeExecution::get().run(pad);

  auto pad_data = locomotiv::annot_data(pad);
  ASSERT_NE(pad_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, pad_data->dtype());
  ASSERT_EQ(Shape({1, 6, 4, 1}), *(pad_data->shape()));

  ASSERT_FLOAT_EQ(1.0f, pad_data->as_f32_bufptr()->at(Index{0, 3, 1, 0}));
  ASSERT_FLOAT_EQ(2.0f, pad_data->as_f32_bufptr()->at(Index{0, 3, 2, 0}));
  ASSERT_FLOAT_EQ(3.0f, pad_data->as_f32_bufptr()->at(Index{0, 4, 1, 0}));
  ASSERT_FLOAT_EQ(4.0f, pad_data->as_f32_bufptr()->at(Index{0, 4, 2, 0}));
  ASSERT_FLOAT_EQ(0.0f, pad_data->as_f32_bufptr()->at(Index{0, 0, 0, 0}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(pad));
}

TEST(NodeExecution_Pad, tensor_constant_pad_1_dim)
{
  auto g = loco::make_graph();

  auto inputTensor = g->nodes()->create<loco::Pull>();
  inputTensor->dtype(loco::DataType::FLOAT32);
  inputTensor->shape({3});
  auto inputTensor_buf = make_buffer<float, LexicalLayout>(Shape{3});
  inputTensor_buf.at(Index{0}) = 1.0f;
  inputTensor_buf.at(Index{1}) = 5.0f;
  inputTensor_buf.at(Index{2}) = 3.0f;
  auto inputTensor_data = locomotiv::make_data(inputTensor_buf);
  locomotiv::annot_data(inputTensor, std::move(inputTensor_data));
  locomotiv::annot_domain(inputTensor, loco::Domain::Tensor);

  auto constant = g->nodes()->create<loco::ConstGen>();
  constant->dtype(loco::DataType::FLOAT32);
  constant->shape({1});
  auto constant_buf = make_buffer<float, LexicalLayout>(Shape{1});
  constant_buf.at(Index{0}) = 0.0f;
  auto constant_data = locomotiv::make_data(constant_buf);
  locomotiv::annot_data(constant, std::move(constant_data));
  locomotiv::annot_domain(constant, loco::Domain::Tensor);

  auto pad = g->nodes()->create<loco::TensorConstantPad>();
  pad->input(inputTensor);
  pad->constant(constant);
  auto padding = pad->padding();
  padding->rank(1);
  padding->front(0) = 2;
  padding->back(0) = 1;

  locomotiv::NodeExecution::get().run(pad);

  auto pad_data = locomotiv::annot_data(pad);
  ASSERT_NE(pad_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, pad_data->dtype());
  ASSERT_EQ(Shape({6}), *(pad_data->shape()));

  ASSERT_FLOAT_EQ(0.0f, pad_data->as_f32_bufptr()->at(Index{0}));
  ASSERT_FLOAT_EQ(0.0f, pad_data->as_f32_bufptr()->at(Index{1}));
  ASSERT_FLOAT_EQ(1.0f, pad_data->as_f32_bufptr()->at(Index{2}));
  ASSERT_FLOAT_EQ(5.0f, pad_data->as_f32_bufptr()->at(Index{3}));
  ASSERT_FLOAT_EQ(3.0f, pad_data->as_f32_bufptr()->at(Index{4}));
  ASSERT_FLOAT_EQ(0.0f, pad_data->as_f32_bufptr()->at(Index{5}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(pad));
}

TEST(NodeExecution_Pad, tensor_constant_pad_6_dim)
{
  auto g = loco::make_graph();

  auto inputTensor = g->nodes()->create<loco::Pull>();
  inputTensor->dtype(loco::DataType::FLOAT32);
  inputTensor->shape({2, 1, 3, 2, 1, 2});
  auto inputTensor_buf = make_buffer<float, LexicalLayout>(Shape{2, 1, 3, 2, 1, 2});
  int a, b, c, d, e, f;
  float dummy = 1.0f;
  for (uint32_t a = 0; a < 2; a++)
  {
    for (uint32_t b = 0; b < 1; b++)
    {
      for (uint32_t c = 0; c < 3; c++)
      {
        for (uint32_t d = 0; d < 2; d++)
        {
          for (uint32_t e = 0; e < 1; e++)
          {
            for (uint32_t f = 0; f < 2; f++)
            {
              inputTensor_buf.at(Index{a, b, c, d, e, f}) = dummy++;
            }
          }
        }
      }
    }
  }
  auto inputTensor_data = locomotiv::make_data(inputTensor_buf);
  locomotiv::annot_data(inputTensor, std::move(inputTensor_data));
  locomotiv::annot_domain(inputTensor, loco::Domain::Tensor);

  auto constant = g->nodes()->create<loco::ConstGen>();
  constant->dtype(loco::DataType::FLOAT32);
  constant->shape({1});
  auto constant_buf = make_buffer<float, LexicalLayout>(Shape{1});
  constant_buf.at(Index{0}) = 0.0f;
  auto constant_data = locomotiv::make_data(constant_buf);
  locomotiv::annot_data(constant, std::move(constant_data));
  locomotiv::annot_domain(constant, loco::Domain::Tensor);

  auto pad = g->nodes()->create<loco::TensorConstantPad>();
  pad->input(inputTensor);
  pad->constant(constant);
  auto padding = pad->padding();

  padding->rank(6);
  padding->front(0) = 1;
  padding->back(0) = 1;
  padding->front(1) = 0;
  padding->back(1) = 0;
  padding->front(2) = 1;
  padding->back(2) = 2;
  padding->front(3) = 2;
  padding->back(3) = 1;
  padding->front(4) = 0;
  padding->back(4) = 0;
  padding->front(5) = 1;
  padding->back(5) = 2;

  locomotiv::NodeExecution::get().run(pad);

  auto pad_data = locomotiv::annot_data(pad);
  ASSERT_NE(pad_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, pad_data->dtype());
  ASSERT_EQ(Shape({4, 1, 6, 5, 1, 5}), *(pad_data->shape()));

  ASSERT_FLOAT_EQ(1.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 1, 2, 0, 1}));
  ASSERT_FLOAT_EQ(2.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 1, 2, 0, 2}));
  ASSERT_FLOAT_EQ(3.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 1, 3, 0, 1}));
  ASSERT_FLOAT_EQ(4.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 1, 3, 0, 2}));
  ASSERT_FLOAT_EQ(5.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 2, 2, 0, 1}));
  ASSERT_FLOAT_EQ(6.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 2, 2, 0, 2}));
  ASSERT_FLOAT_EQ(7.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 2, 3, 0, 1}));
  ASSERT_FLOAT_EQ(8.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 2, 3, 0, 2}));
  ASSERT_FLOAT_EQ(9.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 3, 2, 0, 1}));
  ASSERT_FLOAT_EQ(10.0f, pad_data->as_f32_bufptr()->at(Index{1, 0, 3, 2, 0, 2}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(pad));
}
