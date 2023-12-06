/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/operation/BinaryArithmeticOps.h>

#include <gtest/gtest.h>
#include <vector>

TEST(CKer_Operation, Mul)
{
  // Simple
  {
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input1 = {10, 9, 11, 3};
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input2 = {2, 2, 3, 4};
    std::vector<int32_t> expected_output = {20, 18, 33, 12};
    std::vector<int32_t> output(4);

    nnfw::cker::BinaryArithmeticOpParam param;
    param.quantized_activation_min = std::numeric_limits<int32_t>::lowest();
    param.quantized_activation_max = std::numeric_limits<int32_t>::max();
    nnfw::cker::Shape shape{1, 2, 2, 1};

    nnfw::cker::BinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL>(
      param, shape, input1.data(), shape, input2.data(), shape, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_EQ(output[i], expected_output[i]);
  }

  // Negative Value
  {
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input1 = {10, -9, -11, 7};
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input2 = {2, 2, -3, -4};
    std::vector<int32_t> expected_output = {20, -18, 33, -28};
    std::vector<int32_t> output(4);

    nnfw::cker::BinaryArithmeticOpParam param;
    param.quantized_activation_min = std::numeric_limits<int32_t>::lowest();
    param.quantized_activation_max = std::numeric_limits<int32_t>::max();
    nnfw::cker::Shape shape{1, 2, 2, 1};

    nnfw::cker::BinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL>(
      param, shape, input1.data(), shape, input2.data(), shape, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_EQ(output[i], expected_output[i]);
  }

  // Broadcast
  {
    // Shape: {1, 2, 2, 1}
    std::vector<int32_t> input1 = {10, -9, -11, 7};
    // Shape: {1}
    std::vector<int32_t> input2 = {-3};
    std::vector<int32_t> expected_output = {-30, 27, 33, -21};
    std::vector<int32_t> output(4);

    nnfw::cker::BinaryArithmeticOpParam param;
    param.broadcast_category = nnfw::cker::BroadcastableOpCategory::kGenericBroadcast;
    param.quantized_activation_min = std::numeric_limits<int32_t>::lowest();
    param.quantized_activation_max = std::numeric_limits<int32_t>::max();

    nnfw::cker::BroadcastBinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL>(
      param, nnfw::cker::Shape{1, 2, 2, 1}, input1.data(), nnfw::cker::Shape{1}, input2.data(),
      nnfw::cker::Shape{1, 2, 2, 1}, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_EQ(output[i], expected_output[i]);
  }

  // Simple Float
  {
    // Shape: {1, 2, 2, 1}
    std::vector<float> input1 = {10, 9, -11.1, 3};
    // Shape: {1, 2, 2, 1}
    std::vector<float> input2 = {2, -2.2, -3.3, 4};
    std::vector<float> expected_output = {20, -19.8, 36.63, 12};
    std::vector<float> output(4);

    nnfw::cker::BinaryArithmeticOpParam param;
    param.float_activation_min = std::numeric_limits<float>::lowest();
    param.float_activation_max = std::numeric_limits<float>::max();
    nnfw::cker::Shape shape{1, 2, 2, 1};

    nnfw::cker::BinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL>(
      param, shape, input1.data(), shape, input2.data(), shape, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_NEAR(output[i], expected_output[i], 1e-5f);
  }

  // Float Relu
  {
    // Shape: {1, 2, 2, 1}
    std::vector<float> input1 = {10, 9, -11.1, 3};
    // Shape: {1, 2, 2, 1}
    std::vector<float> input2 = {2, -2.2, -3.3, 4};
    std::vector<float> expected_output = {20, 0, 36.63, 12};
    std::vector<float> output(4);

    nnfw::cker::BinaryArithmeticOpParam param;
    param.float_activation_min = 0;
    param.float_activation_max = std::numeric_limits<float>::max();
    nnfw::cker::Shape shape{1, 2, 2, 1};

    nnfw::cker::BinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL>(
      param, shape, input1.data(), shape, input2.data(), shape, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_NEAR(output[i], expected_output[i], 1e-5f);
  }

  // Broadcast
  {
    // Shape: {1, 2, 2, 1}
    std::vector<float> input1 = {10, 9, -11.1, 3};
    // Shape: {1}
    std::vector<float> input2 = {-3};
    std::vector<float> expected_output = {-30, -27, 33.3, -9};
    std::vector<float> output(4);

    nnfw::cker::BinaryArithmeticOpParam param;
    param.broadcast_category = nnfw::cker::BroadcastableOpCategory::kGenericBroadcast;
    param.float_activation_min = std::numeric_limits<float>::lowest();
    param.float_activation_max = std::numeric_limits<float>::max();

    nnfw::cker::BroadcastBinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL>(
      param, nnfw::cker::Shape{1, 2, 2, 1}, input1.data(), nnfw::cker::Shape{1}, input2.data(),
      nnfw::cker::Shape{1, 2, 2, 1}, output.data());

    for (size_t i = 0; i < expected_output.size(); ++i)
      EXPECT_NEAR(output[i], expected_output[i], 1e-5f);
  }

  // Bool8
  {
    // Shape: {1, 2, 2, 1}
    bool input1[4] = {true, true, false, false};
    // Shape: {1, 2, 2, 1}
    bool input2[4] = {true, false, true, false};
    bool expected_output[4] = {true, false, false, false};
    bool output[4];

    nnfw::cker::BinaryArithmeticOpParam param;
    nnfw::cker::Shape shape{1, 2, 2, 1};

    nnfw::cker::BinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL, bool>(
      param, shape, input1, shape, input2, shape, output);

    for (size_t i = 0; i < 4; ++i)
      EXPECT_EQ(output[i], expected_output[i]);
  }

  // Broadcast Bool8
  {
    // Shape: {1, 2, 2, 1}
    bool input1[4] = {true, true, false, false};
    // Shape: {1, 2, 2, 1}
    bool input2[1] = {true};
    bool expected_output[4] = {true, true, false, false};
    bool output[4];

    nnfw::cker::BinaryArithmeticOpParam param;

    nnfw::cker::BroadcastBinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL, bool>(
      param, nnfw::cker::Shape{1, 2, 2, 1}, input1, nnfw::cker::Shape{1}, input2,
      nnfw::cker::Shape{1, 2, 2, 1}, output);

    for (size_t i = 0; i < 4; ++i)
      EXPECT_EQ(output[i], expected_output[i]);
  }

  // TODO Add other types
}

TEST(CKer_Operation, neg_MulUnsupportedBroadcastRank)
{
  // Unsupported rank
  {
    // Shape: {1, 2, 2, 1, 1}
    std::vector<float> input1 = {10, -9, -11, 7};
    // Shape: {1}
    std::vector<float> input2 = {-3};
    std::vector<float> output(4);

    nnfw::cker::BinaryArithmeticOpParam param;

    EXPECT_ANY_THROW(
      nnfw::cker::BroadcastBinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::MUL>(
        param, nnfw::cker::Shape{1, 2, 2, 1, 1}, input1.data(), nnfw::cker::Shape{1}, input2.data(),
        nnfw::cker::Shape{1, 2, 2, 1, 1}, output.data()));
  }
}
