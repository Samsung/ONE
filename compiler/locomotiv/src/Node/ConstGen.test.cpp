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

TEST(NodeExecution_ConstGen, s32)
{
  // Make ConstGen node
  loco::ConstGen constgen;

  constgen.dtype(loco::DataType::S32);
  constgen.shape({2, 3});
  constgen.size<loco::DataType::S32>(6);

  constgen.at<loco::DataType::S32>(0) = 0;  // Set 0,0
  constgen.at<loco::DataType::S32>(1) = 1;  // Set 0,1
  constgen.at<loco::DataType::S32>(2) = 2;  // Set 0,2
  constgen.at<loco::DataType::S32>(3) = -3; // Set 1,0
  constgen.at<loco::DataType::S32>(4) = -4; // Set 1,1
  constgen.at<loco::DataType::S32>(5) = -5; // Set 1,2

  // run execution
  locomotiv::NodeExecution::get().run(&constgen);

  // test
  auto data = locomotiv::annot_data(&constgen);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(loco::DataType::S32, data->dtype());
  ASSERT_EQ(Shape({2, 3}), *data->shape());
  ASSERT_EQ(0, data->as_s32_bufptr()->at(Index{0, 0}));
  ASSERT_EQ(1, data->as_s32_bufptr()->at(Index{0, 1}));
  ASSERT_EQ(2, data->as_s32_bufptr()->at(Index{0, 2}));
  ASSERT_EQ(-3, data->as_s32_bufptr()->at(Index{1, 0}));
  ASSERT_EQ(-4, data->as_s32_bufptr()->at(Index{1, 1}));
  ASSERT_EQ(-5, data->as_s32_bufptr()->at(Index{1, 2}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(&constgen));
}

TEST(NodeExecution_ConstGen, f32)
{
  // Make ConstGen node
  loco::ConstGen constgen;

  constgen.dtype(loco::DataType::FLOAT32);
  constgen.shape({2, 3});
  constgen.size<loco::DataType::FLOAT32>(6);

  constgen.at<loco::DataType::FLOAT32>(0) = 0.0f; // Set 0,0
  constgen.at<loco::DataType::FLOAT32>(1) = 1.0f; // Set 0,1
  constgen.at<loco::DataType::FLOAT32>(2) = 2.0f; // Set 0,2
  constgen.at<loco::DataType::FLOAT32>(3) = 3.0f; // Set 1,0
  constgen.at<loco::DataType::FLOAT32>(4) = 4.0f; // Set 1,1
  constgen.at<loco::DataType::FLOAT32>(5) = 5.0f; // Set 1,2

  // run execution
  locomotiv::NodeExecution::get().run(&constgen);

  // test
  auto data = locomotiv::annot_data(&constgen);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, data->dtype());
  ASSERT_EQ(Shape({2, 3}), *data->shape());
  ASSERT_FLOAT_EQ(0.0f, data->as_f32_bufptr()->at(Index{0, 0}));
  ASSERT_FLOAT_EQ(1.0f, data->as_f32_bufptr()->at(Index{0, 1}));
  ASSERT_FLOAT_EQ(2.0f, data->as_f32_bufptr()->at(Index{0, 2}));
  ASSERT_FLOAT_EQ(3.0f, data->as_f32_bufptr()->at(Index{1, 0}));
  ASSERT_FLOAT_EQ(4.0f, data->as_f32_bufptr()->at(Index{1, 1}));
  ASSERT_FLOAT_EQ(5.0f, data->as_f32_bufptr()->at(Index{1, 2}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(&constgen));
}
