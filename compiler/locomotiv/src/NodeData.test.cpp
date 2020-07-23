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

#include "locomotiv/NodeData.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

TEST(NodeData, as_s32_buffer_wrapper)
{
  const Shape shape{1};
  auto buf = make_buffer<int32_t, LexicalLayout>(shape);
  buf.at(Index{0}) = 42;

  auto data = locomotiv::make_data(buf);

  ASSERT_EQ(loco::DataType::S32, data->dtype());
  ASSERT_EQ(shape, *(data->shape()));
  ASSERT_EQ(42, data->as_s32_bufptr()->at(Index{0}));
}

TEST(NodeData, as_f32_buffer_wrapper)
{
  const Shape shape{1};
  auto buf = make_buffer<float, LexicalLayout>(shape);
  buf.at(Index{0}) = 3.14f;

  auto data = locomotiv::make_data(buf);

  ASSERT_EQ(loco::DataType::FLOAT32, data->dtype());
  ASSERT_EQ(shape, *(data->shape()));
  ASSERT_FLOAT_EQ(3.14f, data->as_f32_bufptr()->at(Index{0}));
}
