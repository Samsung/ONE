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
  ASSERT_EQ(data->dtype(), loco::DataType::S32);
  ASSERT_EQ(*data->shape(), Shape({2, 3}));
  ASSERT_EQ(data->as_s32_bufptr()->at(Index{0, 0}), 0);
  ASSERT_EQ(data->as_s32_bufptr()->at(Index{0, 1}), 1);
  ASSERT_EQ(data->as_s32_bufptr()->at(Index{0, 2}), 2);
  ASSERT_EQ(data->as_s32_bufptr()->at(Index{1, 0}), -3);
  ASSERT_EQ(data->as_s32_bufptr()->at(Index{1, 1}), -4);
  ASSERT_EQ(data->as_s32_bufptr()->at(Index{1, 2}), -5);

  ASSERT_EQ(locomotiv::annot_domain(&constgen), loco::Domain::Tensor);
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
  ASSERT_EQ(data->dtype(), loco::DataType::FLOAT32);
  ASSERT_EQ(*data->shape(), Shape({2, 3}));
  ASSERT_FLOAT_EQ(data->as_f32_bufptr()->at(Index{0, 0}), 0.0f);
  ASSERT_FLOAT_EQ(data->as_f32_bufptr()->at(Index{0, 1}), 1.0f);
  ASSERT_FLOAT_EQ(data->as_f32_bufptr()->at(Index{0, 2}), 2.0f);
  ASSERT_FLOAT_EQ(data->as_f32_bufptr()->at(Index{1, 0}), 3.0f);
  ASSERT_FLOAT_EQ(data->as_f32_bufptr()->at(Index{1, 1}), 4.0f);
  ASSERT_FLOAT_EQ(data->as_f32_bufptr()->at(Index{1, 2}), 5.0f);

  ASSERT_EQ(locomotiv::annot_domain(&constgen), loco::Domain::Tensor);
}
