/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Convert.h"

#include <gtest/gtest.h>

TEST(ConvertTest, as_circle_padding)
{
  ASSERT_EQ(circle::Padding_SAME, as_circle_padding(circlechef::SAME));
  ASSERT_EQ(circle::Padding_VALID, as_circle_padding(circlechef::VALID));
}

TEST(ConvertTest, as_circle_padding_NEG)
{
  EXPECT_THROW(as_circle_padding(static_cast<circlechef::Padding>(99)), std::runtime_error);
}

TEST(ConvertTest, as_circle_activation)
{
  ASSERT_EQ(circle::ActivationFunctionType_NONE, as_circle_activation(circlechef::NONE));
  ASSERT_EQ(circle::ActivationFunctionType_RELU, as_circle_activation(circlechef::RELU));
  ASSERT_EQ(circle::ActivationFunctionType_RELU6, as_circle_activation(circlechef::RELU6));
}

TEST(ConvertTest, as_circle_activation_NEG)
{
  EXPECT_THROW(as_circle_activation(static_cast<circlechef::Activation>(99)), std::runtime_error);
}

TEST(ConvertTest, as_circle_tensortype)
{
  ASSERT_EQ(circle::TensorType_FLOAT32, as_circle_tensortype(circlechef::FLOAT32));
  ASSERT_EQ(circle::TensorType_INT64, as_circle_tensortype(circlechef::INT64));
  ASSERT_EQ(circle::TensorType_INT32, as_circle_tensortype(circlechef::INT32));
  ASSERT_EQ(circle::TensorType_INT16, as_circle_tensortype(circlechef::INT16));
  ASSERT_EQ(circle::TensorType_UINT8, as_circle_tensortype(circlechef::UINT8));
  ASSERT_EQ(circle::TensorType_BOOL, as_circle_tensortype(circlechef::BOOL));
}

TEST(ConvertTest, as_circle_tensortype_NEG)
{
  EXPECT_THROW(as_circle_tensortype(static_cast<circlechef::TensorType>(99)), std::runtime_error);
}
