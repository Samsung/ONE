/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cker/operation/FloorMod.h>

#include <gtest/gtest.h>
#include <vector>

TEST(CKer_Operation, FloorMod)
{
  // Simple
  {
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input1 = {10, 9, 11, 3};
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input2 = {2, 2, 3, 4};
    std::vector<int32_t> expected_output = {0, 1, 2, 3};
    std::vector<int32_t> output(4);

    nnfw::cker::FloorModElementwise(nnfw::cker::Shape{1, 2, 2, 1}, input1.data(), input2.data(),
                                    output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      ASSERT_EQ(output[i], expected_output[i]);
  }

  // Negative Value
  {
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input1 = {10, -9, -11, 7};
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input2 = {2, 2, -3, -4};
    std::vector<int32_t> expected_output = {0, 1, -2, -1};
    std::vector<int32_t> output(4);

    nnfw::cker::FloorModElementwise(nnfw::cker::Shape{1, 2, 2, 1}, input1.data(), input2.data(),
                                    output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      ASSERT_EQ(output[i], expected_output[i]);
  }

  // Broadcast
  {
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input1 = {10, -9, -11, 7};
    // Shape: {1}
    std::vector<int32_t> input2 = {-3};
    std::vector<int32_t> expected_output = {-2, 0, -2, -2};
    std::vector<int32_t> output(4);

    nnfw::cker::FloorModBroadcast(nnfw::cker::Shape{1, 2, 2, 1}, input1.data(),
                                  nnfw::cker::Shape{1}, input2.data(),
                                  nnfw::cker::Shape{1, 2, 2, 1}, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      ASSERT_EQ(output[i], expected_output[i]);
  }

  // Broadcast Int64
  {
    // Shape: {1, 2, 2, 1}
    std::vector<int64_t> input1 = {10, -9, -11, (1LL << 34) + 9};
    // Shape: {1}
    std::vector<int64_t> input2 = {-(1LL << 33)};
    std::vector<int64_t> expected_output = {-8589934582, -9, -11, -8589934583};
    std::vector<int64_t> output(4);

    nnfw::cker::FloorModBroadcast(nnfw::cker::Shape{1, 2, 2, 1}, input1.data(),
                                  nnfw::cker::Shape{1}, input2.data(),
                                  nnfw::cker::Shape{1, 2, 2, 1}, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      ASSERT_EQ(output[i], expected_output[i]);
  }

  // Simple Float
  {
    // Shape: {1, 2, 2, 1}
    std::vector<float> input1 = {10, 9, 11, 3};
    // Shape: {1, 2, 2, 1}
    std::vector<float> input2 = {2, 2, 3, 4};
    std::vector<float> expected_output = {0, 1, 2, 3};
    std::vector<float> output(4);

    nnfw::cker::FloorModElementwise(nnfw::cker::Shape{1, 2, 2, 1}, input1.data(), input2.data(),
                                    output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      ASSERT_EQ(output[i], expected_output[i]);
  }

  // Negative Value Float
  {
    // Shape: {1, 2, 2, 1}
    std::vector<float> input1 = {10, -9, -11, 7};
    // Shape: {1, 2, 2, 1}
    std::vector<float> input2 = {2, 2, -3, -4};
    std::vector<float> expected_output = {0, 1, -2, -1};
    std::vector<float> output(4);

    nnfw::cker::FloorModElementwise(nnfw::cker::Shape{1, 2, 2, 1}, input1.data(), input2.data(),
                                    output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      ASSERT_EQ(output[i], expected_output[i]);
  }

  // Broadcast
  {
    // Shape: {1, 2, 2, 1}
    std::vector<float> input1 = {10, -9, -11, 7};
    // Shape: {1}
    std::vector<float> input2 = {-3};
    std::vector<float> expected_output = {-2, 0, -2, -2};
    std::vector<float> output(4);

    nnfw::cker::FloorModBroadcast(nnfw::cker::Shape{1, 2, 2, 1}, input1.data(),
                                  nnfw::cker::Shape{1}, input2.data(),
                                  nnfw::cker::Shape{1, 2, 2, 1}, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      ASSERT_EQ(output[i], expected_output[i]);
  }
}

TEST(CKer_Operation, neg_FloorModUnsupportedBroadcastRank)
{
  // Unsupported rank
  {
    // Shape: {1, 2, 2, 1, 1}
    std::vector<float> input1 = {10, -9, -11, 7};
    // Shape: {1}
    std::vector<float> input2 = {-3};
    std::vector<float> output(4);

    EXPECT_ANY_THROW(nnfw::cker::FloorModBroadcast(
      nnfw::cker::Shape{1, 2, 2, 1, 1}, input1.data(), nnfw::cker::Shape{1}, input2.data(),
      nnfw::cker::Shape{1, 2, 2, 1, 1}, output.data()));
  }
}
