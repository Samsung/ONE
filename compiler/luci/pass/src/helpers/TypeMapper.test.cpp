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

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

#include "TypeMapper.h"

#include <vector>

namespace
{

template <typename T> bool fill_const_node(luci::CircleConst *node, std::vector<T> &data)
{
  if (node->dtype() != luci::TypeMapper<T>::get())
    return false;

  node->size<luci::TypeMapper<T>::get()>(data.size());
  for (uint32_t i = 0; i < data.size(); i++)
  {
    node->at<luci::TypeMapper<T>::get()>(i) = data.at(i);
  }

  return true;
}

class STRANGER
{
};

} // namespace

TEST(TypeMapperTest, simple_test)
{
  EXPECT_EQ(loco::DataType::FLOAT32, luci::TypeMapper<float>::get());
  EXPECT_EQ(loco::DataType::U8, luci::TypeMapper<uint8_t>::get());
  EXPECT_EQ(loco::DataType::U16, luci::TypeMapper<uint16_t>::get());
  EXPECT_EQ(loco::DataType::U32, luci::TypeMapper<uint32_t>::get());
  EXPECT_EQ(loco::DataType::U64, luci::TypeMapper<uint64_t>::get());
  EXPECT_EQ(loco::DataType::S8, luci::TypeMapper<int8_t>::get());
  EXPECT_EQ(loco::DataType::S16, luci::TypeMapper<int16_t>::get());
  EXPECT_EQ(loco::DataType::S32, luci::TypeMapper<int32_t>::get());
  EXPECT_EQ(loco::DataType::S64, luci::TypeMapper<int64_t>::get());
}

TEST(TypeMapperTest, with_template_test)
{
  std::vector<int32_t> int32_vec{0, 1, 2, 3, 4, 5, 6, 7};
  luci::CircleConst const_node;
  const_node.dtype(loco::DataType::S32);
  EXPECT_TRUE(fill_const_node(&const_node, int32_vec));
  EXPECT_EQ(8, const_node.size<loco::DataType::S32>());
  EXPECT_EQ(0, const_node.at<loco::DataType::S32>(0));
  EXPECT_EQ(1, const_node.at<loco::DataType::S32>(1));
  EXPECT_EQ(2, const_node.at<loco::DataType::S32>(2));
  EXPECT_EQ(3, const_node.at<loco::DataType::S32>(3));
  EXPECT_EQ(4, const_node.at<loco::DataType::S32>(4));
  EXPECT_EQ(5, const_node.at<loco::DataType::S32>(5));
  EXPECT_EQ(6, const_node.at<loco::DataType::S32>(6));
  EXPECT_EQ(7, const_node.at<loco::DataType::S32>(7));

  std::vector<float> f32_vec{0.0, 1.1, 2.2, 3.3, 4.4, 5.5};
  const_node.dtype(loco::DataType::FLOAT32);
  EXPECT_FALSE(fill_const_node(&const_node, int32_vec));
  EXPECT_TRUE(fill_const_node(&const_node, f32_vec));
  EXPECT_EQ(6, const_node.size<loco::DataType::FLOAT32>());
  EXPECT_FLOAT_EQ(0.0, const_node.at<loco::DataType::FLOAT32>(0));
  EXPECT_FLOAT_EQ(1.1, const_node.at<loco::DataType::FLOAT32>(1));
  EXPECT_FLOAT_EQ(2.2, const_node.at<loco::DataType::FLOAT32>(2));
  EXPECT_FLOAT_EQ(3.3, const_node.at<loco::DataType::FLOAT32>(3));
  EXPECT_FLOAT_EQ(4.4, const_node.at<loco::DataType::FLOAT32>(4));
  EXPECT_FLOAT_EQ(5.5, const_node.at<loco::DataType::FLOAT32>(5));
}

TEST(TypeMapperTest, wrong_condition_NEG)
{
  EXPECT_EQ(loco::DataType::Unknown, luci::TypeMapper<STRANGER>::get());
}
