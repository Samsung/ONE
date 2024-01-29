/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cker/train/operation/BinaryArithmetic.h>

#include <gtest/gtest.h>
#include <vector>

TEST(CKer_Operation, AddGrad)
{
  {
    // Shape: {2, 3}
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> expected_lhs_backward = incoming_backward;
    std::vector<float> expected_rhs_backward = incoming_backward;
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(6);

    nnfw::cker::train::BinaryArithmeticGrad(nnfw::cker::Shape{2, 3}, incoming_backward.data(),
                                            nnfw::cker::Shape{2, 3}, lhs_backward.data(),
                                            nnfw::cker::Shape{2, 3}, rhs_backward.data(),
                                            nnfw::cker::train::ArithmeticType::kAdd);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], expected_rhs_backward[i]);
  }
  {
    // Shape: {4, 3}
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13};
    std::vector<float> expected_lhs_backward = incoming_backward;
    std::vector<float> expected_rhs_backward = incoming_backward;
    std::vector<float> lhs_backward(12);
    std::vector<float> rhs_backward(12);

    nnfw::cker::train::BinaryArithmeticGrad(nnfw::cker::Shape{4, 3}, incoming_backward.data(),
                                            nnfw::cker::Shape{4, 3}, lhs_backward.data(),
                                            nnfw::cker::Shape{4, 3}, rhs_backward.data(),
                                            nnfw::cker::train::ArithmeticType::kAdd);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], expected_rhs_backward[i]);
  }
}
