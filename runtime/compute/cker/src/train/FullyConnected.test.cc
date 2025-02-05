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

#include <cker/train/operation/FullyConnected.h>

#include <gtest/gtest.h>
#include <vector>

TEST(CKer_Operation, FullyConnectedBiasGrad)
{
  {
    // Shape: {2, 4}
    std::vector<float> incoming_backward = {-1, 2, -3, 4, 5, -6, -7, 8};
    // Shape: {4}
    std::vector<float> expected_bias_backward = {4, -4, -10, 12};
    std::vector<float> bias_backward(4);

    nnfw::cker::train::FullyConnectedBiasGrad(
      nnfw::cker::Shape{2, 4}, incoming_backward.data(),
      nnfw::cker::Shape{static_cast<int>(bias_backward.size())}, bias_backward.data());

    for (size_t i = 0; i < bias_backward.size(); ++i)
      ASSERT_EQ(bias_backward[i], expected_bias_backward[i]);
  }

  {
    // Shape: {3, 3}
    std::vector<float> incoming_backward = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    // Shape: {3}
    std::vector<float> expected_bias_backward = {-4, 15, 0};
    std::vector<float> bias_backward(3);

    nnfw::cker::train::FullyConnectedBiasGrad(
      nnfw::cker::Shape{3, 3}, incoming_backward.data(),
      nnfw::cker::Shape{static_cast<int>(bias_backward.size())}, bias_backward.data());

    for (size_t i = 0; i < bias_backward.size(); ++i)
      ASSERT_EQ(bias_backward[i], expected_bias_backward[i]);
  }

  {
    // Shape: {1, 2, 2, 3}
    std::vector<float> incoming_backward = {-1, 2, -3, 4, 5, -6, -7, 8, 9, -10, -11, 12};
    // Shape: {3}
    std::vector<float> expected_bias_backward = {-14, 4, 12};
    std::vector<float> bias_backward(3);

    nnfw::cker::train::FullyConnectedBiasGrad(
      nnfw::cker::Shape{1, 2, 2, 3}, incoming_backward.data(),
      nnfw::cker::Shape{static_cast<int>(bias_backward.size())}, bias_backward.data());

    for (size_t i = 0; i < bias_backward.size(); ++i)
      ASSERT_EQ(bias_backward[i], expected_bias_backward[i]);
  }
}

TEST(CKer_Operation, neg_FullyConnectedBiasGrad)
{
  {
    // Unmatched shape
    // Shape: {2, 4}
    std::vector<float> incoming_backward = {-1, 2, -3, 4, 5, -6, -7, 8};
    // Shape: {3}
    std::vector<float> bias_backward(3);
    EXPECT_ANY_THROW(nnfw::cker::train::FullyConnectedBiasGrad(
                       nnfw::cker::Shape{2, 4}, incoming_backward.data(),
                       nnfw::cker::Shape{static_cast<int>(bias_backward.size())},
                       bias_backward.data()););
  }
}
