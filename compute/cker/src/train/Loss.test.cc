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

#include <cker/train/operation/Loss.h>

#include <gtest/gtest.h>
#include <vector>

TEST(CKer_Operation, LossMSE)
{
  {
    // Shape: {1, 10} -> m_rows:10, m_cols:1
    std::vector<int> y_pred = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> y_true = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> output(1);
    std::vector<int> expected = {1};

    nnfw::cker::train::MSE(nnfw::cker::Shape{1, 10}, y_pred.data(), nnfw::cker::Shape{1, 10},
                           y_true.data(), nnfw::cker::Shape{1}, output.data());

    ASSERT_EQ(output[0], expected[0]);
  }

  {
    // Shape: {1, 10} -> m_rows:10, m_cols:1
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<float> output(1);
    std::vector<float> expected = {1.0};

    nnfw::cker::train::MSE(nnfw::cker::Shape{1, 10}, y_pred.data(), nnfw::cker::Shape{1, 10},
                           y_true.data(), nnfw::cker::Shape{1}, output.data());

    ASSERT_FLOAT_EQ(output[0], expected[0]);
  }

  {
    // Shape: {2, 3} -> m_rows:3, m_cols:2
    std::vector<float> y_pred = {27.2, 31.8, 51.9, 10.2, 34.2, 12.4};
    std::vector<float> y_true = {31.3, 40.3, 29.7, 12.9, 25.8, 11.9};
    std::vector<float> output(1);
    std::vector<float> expected = {110.0};

    nnfw::cker::train::MSE(nnfw::cker::Shape{2, 3}, y_pred.data(), nnfw::cker::Shape{2, 3},
                           y_true.data(), nnfw::cker::Shape{1}, output.data());

    ASSERT_FLOAT_EQ(output[0], expected[0]);
  }

  {
    // Shape: {2, 3, 4} -> m_rows:4, m_cols:6
    std::vector<float> y_pred = {1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
                                 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.};
    std::vector<float> y_true = {1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.,
                                 1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.};
    std::vector<float> output(1);
    std::vector<float> expected = {2.1666667};

    nnfw::cker::train::MSE(nnfw::cker::Shape{2, 3, 4}, y_pred.data(), nnfw::cker::Shape{2, 3, 4},
                           y_true.data(), nnfw::cker::Shape{1}, output.data());

    ASSERT_FLOAT_EQ(output[0], expected[0]);
  }
}

TEST(CKer_Operation, neg_LossMSE)
{
  {
    // Invalid expected value
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<float> output(1);
    std::vector<float> expected = {-1.0};

    nnfw::cker::train::MSE(nnfw::cker::Shape{2, 3, 4}, y_pred.data(), nnfw::cker::Shape{2, 3, 4},
                           y_true.data(), nnfw::cker::Shape{1}, output.data());

    ASSERT_NE(output[0], expected[0]);
  }

  {
    // Invalid output shape
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<float> output(3);
    std::vector<float> expected = {1.0};

    EXPECT_ANY_THROW(nnfw::cker::train::MSE(nnfw::cker::Shape{2, 3, 4}, y_pred.data(),
                                            nnfw::cker::Shape{2, 3, 4}, y_true.data(),
                                            nnfw::cker::Shape{3}, output.data()));
  }

  {
    // Different y_pread and y_true shape
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5.};
    std::vector<float> output(1);
    std::vector<float> expected = {1.0};

    EXPECT_ANY_THROW(nnfw::cker::train::MSE(nnfw::cker::Shape{2, 3, 4}, y_pred.data(),
                                            nnfw::cker::Shape{2, 3}, y_true.data(),
                                            nnfw::cker::Shape{1}, output.data()));
  }
}
