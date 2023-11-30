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

namespace
{
using namespace nnfw::cker;

template <typename T> class LossCCEVerifier
{
public:
  LossCCEVerifier(const Shape &in_shape, const Shape &out_shape)
    : _in_shape{in_shape}, _out_shape{out_shape}
  {
  }

  void verifyForward(const std::vector<T> &y_pred, const std::vector<T> &y_true,
                     const std::vector<T> &expected)
  {
    assert(y_pred.size() == y_true.size());

    std::vector<T> output(_out_shape.FlatSize());
    const int N = _in_shape.Dims(0);
    const int D = _in_shape.FlatSize() / N;

    nnfw::cker::train::CategoricalCrossEntropy(_in_shape, y_pred.data(), _in_shape, y_true.data(),
                                               _out_shape, output.data());

    // Don't be panic when it fails after kernel implementation or input is changed.
    // CrossEntropy formula can be calculated slightly differently depending on the environment
    // because it involes calculations such as log or exp.
    for (int i = 0; i < output.size(); ++i)
    {
      EXPECT_NEAR(output[i], expected[i], 1e-3f);
    }
  }

  void throwForward(const std::vector<T> &y_pred, const std::vector<T> &y_true,
                    const std::vector<T> &expected)
  {
    assert(y_pred.size() == y_true.size());

    std::vector<T> output(_out_shape.FlatSize());
    const int N = _in_shape.Dims(0);
    const int D = _in_shape.FlatSize() / N;

    EXPECT_ANY_THROW(nnfw::cker::train::CategoricalCrossEntropy(
      _in_shape, y_pred.data(), _in_shape, y_true.data(), _out_shape, output.data()));
  }

  void verifyBackward(const std::vector<T> &y_pred, const std::vector<T> &y_true,
                      const std::vector<T> &expected)
  {
    assert(y_pred.size() == y_true.size());

    std::vector<T> output(_in_shape.FlatSize());
    const int N = _in_shape.Dims(0);
    const int D = _in_shape.FlatSize() / N;

    nnfw::cker::train::CategoricalCrossEntropyGrad(_in_shape, y_pred.data(), _in_shape,
                                                   y_true.data(), _out_shape, output.data());

    // Don't be panic when it fails after kernel implementation or input is changed.
    // CrossEntropy Gradient formula can be calculated slightly differently depending on the
    // environment because it involes calculations such as log or exp.
    for (int i = 0; i < output.size(); ++i)
    {
      EXPECT_NEAR(output[i], expected[i], 1e-3f);
    }
  }

  void throwBackward(const std::vector<T> &y_pred, const std::vector<T> &y_true,
                     const std::vector<T> &expected)
  {
    assert(y_pred.size() == y_true.size());

    std::vector<T> output(_out_shape.FlatSize());
    const int N = _in_shape.Dims(0);
    const int D = _in_shape.FlatSize() / N;

    EXPECT_ANY_THROW(nnfw::cker::train::CategoricalCrossEntropyGrad(
      _in_shape, y_pred.data(), _in_shape, y_true.data(), _out_shape, output.data()));
  }

private:
  const Shape _in_shape;
  const Shape _out_shape;
};
} // namespace

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

    EXPECT_EQ(output[0], expected[0]);
  }

  {
    // Shape: {1, 10} -> m_rows:10, m_cols:1
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<float> output(1);
    std::vector<float> expected = {1.0};

    nnfw::cker::train::MSE(nnfw::cker::Shape{1, 10}, y_pred.data(), nnfw::cker::Shape{1, 10},
                           y_true.data(), nnfw::cker::Shape{1}, output.data());

    EXPECT_FLOAT_EQ(output[0], expected[0]);
  }

  {
    // Shape: {2, 3} -> m_rows:3, m_cols:2
    std::vector<float> y_pred = {27.2, 31.8, 51.9, 10.2, 34.2, 12.4};
    std::vector<float> y_true = {31.3, 40.3, 29.7, 12.9, 25.8, 11.9};
    std::vector<float> output(2);
    std::vector<float> expected = {193.9667, 26.033342};

    nnfw::cker::train::MSE(nnfw::cker::Shape{2, 3}, y_pred.data(), nnfw::cker::Shape{2, 3},
                           y_true.data(), nnfw::cker::Shape{2}, output.data());

    for (int i = 0; i < output.size(); ++i)
    {
      EXPECT_FLOAT_EQ(output[i], expected[i]);
    }
  }

  {
    // Shape: {2, 3, 4} -> m_rows:4, m_cols:6
    std::vector<float> y_pred = {1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.,
                                 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.};
    std::vector<float> y_true = {1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.,
                                 1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.};
    std::vector<float> output(2);
    std::vector<float> expected = {2.1666667, 2.1666667};

    nnfw::cker::train::MSE(nnfw::cker::Shape{2, 3, 4}, y_pred.data(), nnfw::cker::Shape{2, 3, 4},
                           y_true.data(), nnfw::cker::Shape{2}, output.data());

    for (int i = 0; i < output.size(); ++i)
    {
      EXPECT_FLOAT_EQ(output[i], expected[i]);
    }
  }
}

TEST(CKer_Operation, neg_LossMSE)
{
  {
    // Invalid expected value
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<float> output(2);
    std::vector<float> expected = {0.0, 0.0};

    nnfw::cker::train::MSE(nnfw::cker::Shape{2, 5}, y_pred.data(), nnfw::cker::Shape{2, 5},
                           y_true.data(), nnfw::cker::Shape{2}, output.data());

    for (int i = 0; i < output.size(); ++i)
    {
      EXPECT_NE(output[i], expected[i]);
    }
  }

  {
    // Invalid output shape
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<float> output(3);
    std::vector<float> expected = {1.0, 1.0};

    EXPECT_ANY_THROW(nnfw::cker::train::MSE(nnfw::cker::Shape{2, 5}, y_pred.data(),
                                            nnfw::cker::Shape{2, 5}, y_true.data(),
                                            nnfw::cker::Shape{3}, output.data()));
  }

  {
    // Different y_pread and y_true shape
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5.};
    std::vector<float> output(2);
    std::vector<float> expected = {1.0, 1.0};

    EXPECT_ANY_THROW(nnfw::cker::train::MSE(nnfw::cker::Shape{2, 5}, y_pred.data(),
                                            nnfw::cker::Shape{2, 3}, y_true.data(),
                                            nnfw::cker::Shape{2}, output.data()));
  }
}

TEST(CKer_Operation, LossMSEGrad)
{
  {
    // Shape: {1, 10} -> m_rows:10, m_cols:1
    std::vector<int> y_pred = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> y_true = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> deriv_y_pred(10);
    std::vector<int> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    nnfw::cker::train::MSEGrad(nnfw::cker::Shape{1, 10}, y_pred.data(), nnfw::cker::Shape{1, 10},
                               y_true.data(), nnfw::cker::Shape{1, 10}, deriv_y_pred.data());

    for (size_t i = 0; i < deriv_y_pred.size(); ++i)
      EXPECT_EQ(deriv_y_pred[i], expected[i]);
  }

  {
    // Shape: {1, 10} -> m_rows:10, m_cols:1
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<float> deriv_y_pred(10);
    std::vector<float> expected = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

    nnfw::cker::train::MSEGrad(nnfw::cker::Shape{1, 10}, y_pred.data(), nnfw::cker::Shape{1, 10},
                               y_true.data(), nnfw::cker::Shape{1, 10}, deriv_y_pred.data());

    for (size_t i = 0; i < deriv_y_pred.size(); ++i)
      EXPECT_FLOAT_EQ(deriv_y_pred[i], expected[i]);
  }

  {
    // Shape: {2, 3} -> m_rows:3, m_cols:2
    std::vector<float> y_pred = {27.2, 31.8, 51.9, 10.2, 34.2, 12.4};
    std::vector<float> y_true = {31.3, 40.3, 29.7, 12.9, 25.8, 11.9};
    std::vector<float> deriv_y_pred(6);
    std::vector<float> expected = {-1.3666667, -2.8333333, 7.4, -0.9, 2.8, 0.1666667};

    nnfw::cker::train::MSEGrad(nnfw::cker::Shape{2, 3}, y_pred.data(), nnfw::cker::Shape{2, 3},
                               y_true.data(), nnfw::cker::Shape{2, 3}, deriv_y_pred.data());

    for (size_t i = 0; i < deriv_y_pred.size(); ++i)
      EXPECT_FLOAT_EQ(deriv_y_pred[i], expected[i]);
  }
}

TEST(CKer_Operation, neg_LossMSEGrad)
{
  {
    // Invalid expected value
    std::vector<float> y_pred = {27.2, 31.8, 51.9, 10.2, 34.2, 12.4};
    std::vector<float> y_true = {31.3, 40.3, 29.7, 12.9, 25.8, 11.9};
    std::vector<float> deriv_y_pred(6);
    std::vector<float> expected = {1., 1., 1., 1., 1., 1.};

    nnfw::cker::train::MSEGrad(nnfw::cker::Shape{2, 3}, y_pred.data(), nnfw::cker::Shape{2, 3},
                               y_true.data(), nnfw::cker::Shape{2, 3}, deriv_y_pred.data());

    for (size_t i = 0; i < deriv_y_pred.size(); ++i)
      EXPECT_NE(deriv_y_pred[i], expected[i]);
  }

  {
    // Different y_pred and y_true shape
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5.};
    std::vector<float> deriv_y_pred(10);

    EXPECT_ANY_THROW(nnfw::cker::train::MSEGrad(nnfw::cker::Shape{1, 10}, y_pred.data(),
                                                nnfw::cker::Shape{2, 3}, y_true.data(),
                                                nnfw::cker::Shape{1, 10}, deriv_y_pred.data()));
  }

  {
    // Different y_pred and deriv_y_pred shape
    std::vector<float> y_pred = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> y_true = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<float> deriv_y_pred(6);

    EXPECT_ANY_THROW(nnfw::cker::train::MSEGrad(nnfw::cker::Shape{1, 10}, y_pred.data(),
                                                nnfw::cker::Shape{1, 10}, y_true.data(),
                                                nnfw::cker::Shape{2, 3}, deriv_y_pred.data()));
  }
}

TEST(CKer_Operation, LossCategoricalCrossEntropy)
{
  // single batch
  {
    nnfw::cker::Shape in_shape{1, 10};
    nnfw::cker::Shape out_shape{1};

    std::vector<float> y_pred = {2.86E-12, 2.82E-13, 0.99999845, 2.36E-07, 2.91E-16,
                                 2.10E-07, 1.69E-14, 1.21E-17,   1.08E-06, 6.23E-18};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {39.617155};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.verifyForward(y_pred, y_true, expected);
  }

  // multiple batch
  {
    nnfw::cker::Shape in_shape{2, 10};
    nnfw::cker::Shape out_shape{2};

    std::vector<float> y_pred = {0.01, 0.03, 0.05, 0.35,  0.04,  0.05,  0.28,  0.09,  0.04,  0.06,
                                 0.89, 0.03, 0.04, 0.005, 0.023, 0.001, 0.004, 0.005, 0.001, 0.001};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> expected = {2.813410, 0.116533};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.verifyForward(y_pred, y_true, expected);
  }
}

TEST(CKer_Operation, neg_LossCategoricalCrossEntropy)
{
  // Invalid output shape
  {
    nnfw::cker::Shape in_shape{1, 10};
    nnfw::cker::Shape out_shape{1, 1};

    std::vector<float> y_pred = {-2.86E-12, 2.82E-13, 0.99999845, 2.36E-07, 2.91E-16,
                                 2.10E-07,  1.69E-14, 1.21E-17,   1.08E-06, 6.23E-18};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {39.617155};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.throwForward(y_pred, y_true, expected);
  }
}

TEST(CKer_Operation, LossCategoricalCrossEntropyGrad)
{
  {
    nnfw::cker::Shape in_shape{1, 10};
    nnfw::cker::Shape grad_shape{1, 10};

    std::vector<float> y_pred = {0.01, 0.03, 0.05, 0.35, 0.04, 0.05, 0.28, 0.09, 0.04, 0.06};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, -16.66666667};

    LossCCEVerifier<float> verifier(in_shape, grad_shape);
    verifier.verifyBackward(y_pred, y_true, expected);
  }

  {
    nnfw::cker::Shape in_shape{2, 10};
    nnfw::cker::Shape grad_shape{2, 10};

    std::vector<float> y_pred = {0.01, 0.03, 0.05, 0.35,  0.04,  0.05,  0.28,  0.09,  0.04,  0.06,
                                 0.89, 0.03, 0.04, 0.005, 0.023, 0.001, 0.004, 0.005, 0.001, 0.001};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, -16.66666667, -1.123595506,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0};

    LossCCEVerifier<float> verifier(in_shape, grad_shape);
    verifier.verifyBackward(y_pred, y_true, expected);
  }
}

TEST(CKer_Operation, neg_LossCategoricalCrossEntropyGrad)
{
  // Invalid grad shape
  {
    nnfw::cker::Shape in_shape{1, 10};
    nnfw::cker::Shape grad_shape{1, 1};

    std::vector<float> y_pred = {0.01, 0.03, 0.05, 0.35, 0.04, 0.05, 0.28, 0.09, 0.04, 0.06};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, -16.66666667};

    LossCCEVerifier<float> verifier(in_shape, grad_shape);
    verifier.throwBackward(y_pred, y_true, expected);
  }
}
