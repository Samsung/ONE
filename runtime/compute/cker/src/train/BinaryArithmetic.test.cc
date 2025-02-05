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
    std::vector<float> lhs(6); // Does not matter
    std::vector<float> rhs(6); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> expected_lhs_backward = incoming_backward;
    std::vector<float> expected_rhs_backward = incoming_backward;
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(6);

    nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 3}, lhs.data(), nnfw::cker::Shape{2, 3}, rhs.data(),
      nnfw::cker::Shape{2, 3}, incoming_backward.data(), nnfw::cker::Shape{2, 3},
      lhs_backward.data(), nnfw::cker::Shape{2, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kAdd);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], expected_rhs_backward[i]);
  }
  {
    // Shape: {4, 3}
    std::vector<float> lhs(12); // Does not matter
    std::vector<float> rhs(12); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13};
    std::vector<float> expected_lhs_backward = incoming_backward;
    std::vector<float> expected_rhs_backward = incoming_backward;
    std::vector<float> lhs_backward(12);
    std::vector<float> rhs_backward(12);

    nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{4, 3}, lhs.data(), nnfw::cker::Shape{4, 3}, rhs.data(),
      nnfw::cker::Shape{4, 3}, incoming_backward.data(), nnfw::cker::Shape{4, 3},
      lhs_backward.data(), nnfw::cker::Shape{4, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kAdd);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], expected_rhs_backward[i]);
  }
}

TEST(CKer_Operation, SubGrad)
{
  {
    // Shape: {2, 3}
    std::vector<float> lhs(6); // Does not matter
    std::vector<float> rhs(6); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> expected_lhs_backward = incoming_backward;
    std::vector<float> expected_rhs_backward = incoming_backward;
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(6);

    nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 3}, lhs.data(), nnfw::cker::Shape{2, 3}, rhs.data(),
      nnfw::cker::Shape{2, 3}, incoming_backward.data(), nnfw::cker::Shape{2, 3},
      lhs_backward.data(), nnfw::cker::Shape{2, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kSub);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], -expected_rhs_backward[i]);
  }
  {
    // Shape: {4, 3}
    std::vector<float> lhs(12); // Does not matter
    std::vector<float> rhs(12); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13};
    std::vector<float> expected_lhs_backward = incoming_backward;
    std::vector<float> expected_rhs_backward = incoming_backward;
    std::vector<float> lhs_backward(12);
    std::vector<float> rhs_backward(12);

    nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{4, 3}, lhs.data(), nnfw::cker::Shape{4, 3}, rhs.data(),
      nnfw::cker::Shape{4, 3}, incoming_backward.data(), nnfw::cker::Shape{4, 3},
      lhs_backward.data(), nnfw::cker::Shape{4, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kSub);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], -expected_rhs_backward[i]);
  }
  {
    // Shape: {2, 2, 2}
    std::vector<float> lhs(8); // Does not matter
    std::vector<float> rhs(8); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7, -8, 9};
    std::vector<float> expected_lhs_backward = incoming_backward;
    std::vector<float> expected_rhs_backward = incoming_backward;
    std::vector<float> lhs_backward(8);
    std::vector<float> rhs_backward(8);

    nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 2, 2}, lhs.data(), nnfw::cker::Shape{2, 2, 2}, rhs.data(),
      nnfw::cker::Shape{2, 2, 2}, incoming_backward.data(), nnfw::cker::Shape{2, 2, 2},
      lhs_backward.data(), nnfw::cker::Shape{2, 2, 2}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kSub);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], -expected_rhs_backward[i]);
  }
}

TEST(CKer_Operation, MulGrad)
{
  {
    // Shape: {2, 3}
    std::vector<float> lhs = {-8, 9, -10, 11, -12, 13};
    std::vector<float> rhs = {-14, 15, -16, 17, -18, 19};
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> expected_lhs_backward(6);
    std::vector<float> expected_rhs_backward(6);
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(6);

    for (uint32_t i = 0; i < 6; ++i)
    {
      expected_lhs_backward[i] = incoming_backward[i] * rhs[i];
      expected_rhs_backward[i] = incoming_backward[i] * lhs[i];
    }

    nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 3}, lhs.data(), nnfw::cker::Shape{2, 3}, rhs.data(),
      nnfw::cker::Shape{2, 3}, incoming_backward.data(), nnfw::cker::Shape{2, 3},
      lhs_backward.data(), nnfw::cker::Shape{2, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kMul);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], expected_rhs_backward[i]);
  }
  {
    // Shape: {4, 3}
    std::vector<float> lhs = {-14, 15, -16, 17, -18, 19, -20, 21, -22, 23, -24, 25};
    std::vector<float> rhs = {-26, 27, -28, 29, -30, 31, -32, 33, -34, 35, -36, 37};
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13};
    std::vector<float> expected_lhs_backward(12);
    std::vector<float> expected_rhs_backward(12);
    std::vector<float> lhs_backward(12);
    std::vector<float> rhs_backward(12);

    for (uint32_t i = 0; i < 12; ++i)
    {
      expected_lhs_backward[i] = incoming_backward[i] * rhs[i];
      expected_rhs_backward[i] = incoming_backward[i] * lhs[i];
    }

    nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{4, 3}, lhs.data(), nnfw::cker::Shape{4, 3}, rhs.data(),
      nnfw::cker::Shape{4, 3}, incoming_backward.data(), nnfw::cker::Shape{4, 3},
      lhs_backward.data(), nnfw::cker::Shape{4, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kMul);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], expected_rhs_backward[i]);
  }
  {
    // Shape: {2, 2, 2}
    std::vector<float> lhs = {-10, 11, -12, 13, -14, 15, -16, 17};
    std::vector<float> rhs = {-18, 19, -20, 21, -22, 23, -24, 25};
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7, -8, 9};
    std::vector<float> expected_lhs_backward = incoming_backward;
    std::vector<float> expected_rhs_backward = incoming_backward;
    std::vector<float> lhs_backward(8);
    std::vector<float> rhs_backward(8);

    for (uint32_t i = 0; i < 8; ++i)
    {
      expected_lhs_backward[i] = incoming_backward[i] * rhs[i];
      expected_rhs_backward[i] = incoming_backward[i] * lhs[i];
    }

    nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 2, 2}, lhs.data(), nnfw::cker::Shape{2, 2, 2}, rhs.data(),
      nnfw::cker::Shape{2, 2, 2}, incoming_backward.data(), nnfw::cker::Shape{2, 2, 2},
      lhs_backward.data(), nnfw::cker::Shape{2, 2, 2}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kMul);

    for (size_t i = 0; i < lhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(lhs_backward[i], expected_lhs_backward[i]);
    for (size_t i = 0; i < rhs_backward.size(); ++i)
      EXPECT_FLOAT_EQ(rhs_backward[i], expected_rhs_backward[i]);
  }
}

TEST(CKer_Operation, neg_BinaryArithmeticDistinctShape)
{
  {
    // all but lhs have the same shape
    std::vector<float> lhs(8); // Does not matter
    std::vector<float> rhs(6); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(6);

    EXPECT_ANY_THROW(nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 2, 2}, lhs.data(), nnfw::cker::Shape{2, 3}, rhs.data(),
      nnfw::cker::Shape{2, 3}, incoming_backward.data(), nnfw::cker::Shape{2, 3},
      lhs_backward.data(), nnfw::cker::Shape{2, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kAdd));
  }
  {
    // all but rhs have the same shape
    std::vector<float> lhs(6); // Does not matter
    std::vector<float> rhs(8); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(6);

    EXPECT_ANY_THROW(nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 3}, lhs.data(), nnfw::cker::Shape{2, 2, 2}, rhs.data(),
      nnfw::cker::Shape{2, 3}, incoming_backward.data(), nnfw::cker::Shape{2, 3},
      lhs_backward.data(), nnfw::cker::Shape{2, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kAdd));
  }
  {
    // all but incoming_backward have the same shape
    std::vector<float> lhs(6); // Does not matter
    std::vector<float> rhs(6); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7, -8, 9};
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(6);

    EXPECT_ANY_THROW(nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 3}, lhs.data(), nnfw::cker::Shape{2, 3}, rhs.data(),
      nnfw::cker::Shape{2, 2, 2}, incoming_backward.data(), nnfw::cker::Shape{2, 3},
      lhs_backward.data(), nnfw::cker::Shape{2, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kAdd));
  }
  {
    // all but lhs_backward have the same shape
    std::vector<float> lhs(6); // Does not matter
    std::vector<float> rhs(6); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> lhs_backward(8);
    std::vector<float> rhs_backward(6);

    EXPECT_ANY_THROW(nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 3}, lhs.data(), nnfw::cker::Shape{2, 3}, rhs.data(),
      nnfw::cker::Shape{2, 3}, incoming_backward.data(), nnfw::cker::Shape{2, 2, 2},
      lhs_backward.data(), nnfw::cker::Shape{2, 3}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kAdd));
  }
  {
    // all but rhs_backward have the same shape
    std::vector<float> lhs(6); // Does not matter
    std::vector<float> rhs(6); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(8);

    EXPECT_ANY_THROW(nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 3}, lhs.data(), nnfw::cker::Shape{2, 3}, rhs.data(),
      nnfw::cker::Shape{2, 3}, incoming_backward.data(), nnfw::cker::Shape{2, 3},
      lhs_backward.data(), nnfw::cker::Shape{2, 2, 2}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kAdd));
  }
  {
    std::vector<float> lhs(8); // Does not matter
    std::vector<float> rhs(8); // Does not matter
    std::vector<float> incoming_backward = {-2, 3, -4, 5, -6, 7};
    std::vector<float> lhs_backward(6);
    std::vector<float> rhs_backward(8);

    EXPECT_ANY_THROW(nnfw::cker::train::BinaryArithmeticGrad(
      nnfw::cker::Shape{2, 2, 2}, lhs.data(), nnfw::cker::Shape{2, 2, 2}, rhs.data(),
      nnfw::cker::Shape{2, 3}, incoming_backward.data(), nnfw::cker::Shape{2, 3},
      lhs_backward.data(), nnfw::cker::Shape{2, 2, 2}, rhs_backward.data(),
      nnfw::cker::train::ArithmeticType::kAdd));
  }
}
