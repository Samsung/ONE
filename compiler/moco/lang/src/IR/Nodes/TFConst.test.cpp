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

#include "moco/IR/Nodes/TFConst.h"
#include "moco/IR/TFDialect.h"

#include <gtest/gtest.h>

TEST(TFConstantTest, constructor)
{
  moco::TFConst constant;

  ASSERT_EQ(constant.dialect(), moco::TFDialect::get());
  ASSERT_EQ(constant.opcode(), moco::TFOpcode::Const);

  ASSERT_EQ(constant.dtype(), loco::DataType::Unknown);
  ASSERT_EQ(constant.rank(), 0);

  constant.dtype(loco::DataType::FLOAT32);
  ASSERT_EQ(constant.dtype(), loco::DataType::FLOAT32);

  constant.rank(2);
  ASSERT_EQ(constant.rank(), 2);

  constant.dim(0) = 2;
  constant.dim(1) = 3;

  ASSERT_TRUE(constant.dim(0).known());
  ASSERT_TRUE(constant.dim(1).known());

  ASSERT_EQ(constant.dim(0), 2);
  ASSERT_EQ(constant.dim(1), 3);

  constant.size<loco::DataType::FLOAT32>(6);

  ASSERT_EQ(constant.size<loco::DataType::FLOAT32>(), 6);

  constant.at<loco::DataType::FLOAT32>(0) = 0.0f; // Set 0,0
  constant.at<loco::DataType::FLOAT32>(1) = 1.0f; // Set 0,1
  constant.at<loco::DataType::FLOAT32>(2) = 2.0f; // Set 0,2
  constant.at<loco::DataType::FLOAT32>(3) = 3.0f; // Set 1,0
  constant.at<loco::DataType::FLOAT32>(4) = 4.0f; // Set 1,1
  constant.at<loco::DataType::FLOAT32>(5) = 5.0f; // Set 1,2

  ASSERT_EQ(constant.at<loco::DataType::FLOAT32>(0), 0.0f);
  ASSERT_EQ(constant.at<loco::DataType::FLOAT32>(1), 1.0f);
  ASSERT_EQ(constant.at<loco::DataType::FLOAT32>(2), 2.0f);
  ASSERT_EQ(constant.at<loco::DataType::FLOAT32>(3), 3.0f);
  ASSERT_EQ(constant.at<loco::DataType::FLOAT32>(4), 4.0f);
  ASSERT_EQ(constant.at<loco::DataType::FLOAT32>(5), 5.0f);
}

TEST(TFConstantTest, datatype_s8)
{
  moco::TFConst constant;

  ASSERT_EQ(constant.dialect(), moco::TFDialect::get());
  ASSERT_EQ(constant.opcode(), moco::TFOpcode::Const);

  ASSERT_EQ(constant.dtype(), loco::DataType::Unknown);
  ASSERT_EQ(constant.rank(), 0);

  constant.dtype(loco::DataType::S8);
  ASSERT_EQ(constant.dtype(), loco::DataType::S8);

  constant.rank(1);
  ASSERT_EQ(constant.rank(), 1);

  constant.dim(0) = 3;
  ASSERT_TRUE(constant.dim(0).known());
  ASSERT_EQ(constant.dim(0), 3);
  constant.size<loco::DataType::S8>(3);
  ASSERT_EQ(constant.size<loco::DataType::S8>(), 3);

  constant.at<loco::DataType::S8>(0) = -1;
  constant.at<loco::DataType::S8>(1) = 1;
  constant.at<loco::DataType::S8>(2) = 0;

  ASSERT_EQ(constant.at<loco::DataType::S8>(0), -1);
  ASSERT_EQ(constant.at<loco::DataType::S8>(1), 1);
  ASSERT_EQ(constant.at<loco::DataType::S8>(2), 0);
}
