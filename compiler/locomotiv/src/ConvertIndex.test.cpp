/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConvertIndex.h"

#include <gtest/gtest.h>

TEST(ConvertIndexTest, as_nncc_index_rank_0)
{
  loco::TensorIndex loco_idx;
  ASSERT_EQ(loco_idx.rank(), 0);

  auto nncc_idx = locomotiv::as_nncc_index(loco_idx);

  ASSERT_EQ(nncc_idx.rank(), 0);
}

TEST(ConvertIndexTest, as_nncc_index_rank_1)
{
  loco::TensorIndex loco_idx;
  loco_idx.resize(1);
  loco_idx.at(0) = 5;

  auto nncc_idx = locomotiv::as_nncc_index(loco_idx);

  ASSERT_EQ(nncc_idx.rank(), 1);
  ASSERT_EQ(nncc_idx.at(0), 5);
}

TEST(ConvertIndexTest, as_nncc_index_rank_4)
{
  loco::TensorIndex loco_idx;
  loco_idx.resize(4);
  loco_idx.at(0) = 1;
  loco_idx.at(1) = 2;
  loco_idx.at(2) = 3;
  loco_idx.at(3) = 4;

  auto nncc_idx = locomotiv::as_nncc_index(loco_idx);

  ASSERT_EQ(nncc_idx.rank(), 4);
  ASSERT_EQ(nncc_idx.at(0), 1);
  ASSERT_EQ(nncc_idx.at(1), 2);
  ASSERT_EQ(nncc_idx.at(2), 3);
  ASSERT_EQ(nncc_idx.at(3), 4);
}

TEST(ConvertIndexTest, as_loco_index_rank_0)
{
  nncc::core::ADT::tensor::Index nncc_idx;
  ASSERT_EQ(nncc_idx.rank(), 0);

  auto loco_idx = locomotiv::as_loco_index(nncc_idx);

  ASSERT_EQ(loco_idx.rank(), 0);
}

TEST(ConvertIndexTest, as_loco_index_rank_1)
{
  nncc::core::ADT::tensor::Index nncc_idx;
  nncc_idx.resize(1);
  nncc_idx.at(0) = 10;

  auto loco_idx = locomotiv::as_loco_index(nncc_idx);

  ASSERT_EQ(loco_idx.rank(), 1);
  ASSERT_EQ(loco_idx.at(0), 10);
}

TEST(ConvertIndexTest, as_loco_index_rank_4)
{
  nncc::core::ADT::tensor::Index nncc_idx;
  nncc_idx.resize(4);
  nncc_idx.at(0) = 4;
  nncc_idx.at(1) = 3;
  nncc_idx.at(2) = 2;
  nncc_idx.at(3) = 1;

  auto loco_idx = locomotiv::as_loco_index(nncc_idx);

  ASSERT_EQ(loco_idx.rank(), 4);
  ASSERT_EQ(loco_idx.at(0), 4);
  ASSERT_EQ(loco_idx.at(1), 3);
  ASSERT_EQ(loco_idx.at(2), 2);
  ASSERT_EQ(loco_idx.at(3), 1);
}

TEST(ConvertIndexTest, roundtrip_loco_to_nncc_to_loco)
{
  loco::TensorIndex original_loco_idx;
  original_loco_idx.resize(3);
  original_loco_idx.at(0) = 100;
  original_loco_idx.at(1) = 200;
  original_loco_idx.at(2) = 300;

  auto nncc_idx = locomotiv::as_nncc_index(original_loco_idx);
  auto converted_loco_idx = locomotiv::as_loco_index(nncc_idx);

  ASSERT_EQ(converted_loco_idx.rank(), original_loco_idx.rank());
  for (uint32_t i = 0; i < original_loco_idx.rank(); ++i)
  {
    ASSERT_EQ(converted_loco_idx.at(i), original_loco_idx.at(i));
  }
}

TEST(ConvertIndexTest, roundtrip_nncc_to_loco_to_nncc)
{
  nncc::core::ADT::tensor::Index original_nncc_idx;
  original_nncc_idx.resize(2);
  original_nncc_idx.at(0) = 50;
  original_nncc_idx.at(1) = 60;

  auto loco_idx = locomotiv::as_loco_index(original_nncc_idx);
  auto converted_nncc_idx = locomotiv::as_nncc_index(loco_idx);

  ASSERT_EQ(converted_nncc_idx.rank(), original_nncc_idx.rank());
  for (uint32_t i = 0; i < original_nncc_idx.rank(); ++i)
  {
    ASSERT_EQ(converted_nncc_idx.at(i), original_nncc_idx.at(i));
  }
}
