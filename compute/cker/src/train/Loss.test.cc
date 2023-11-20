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

    // nnfw::cker::train::CategoricalCrossEntropy(y_pred.data(), y_true.data(), output.data(), N, D);

    // Don't be panic when it fails after kernel implementation or input is changed.
    // CrossEntropy formula can be calculated slightly differently depending on the environment
    // because it involes calculations such as log or exp.
    for (int i = 0; i < output.size(); ++i)
    {
      EXPECT_FLOAT_EQ(output[i], expected[i]);
    }
  }

  void throwForward(const std::vector<T> &y_pred, const std::vector<T> &y_true,
                    const std::vector<T> &expected)
  {
    assert(y_pred.size() == y_true.size());

    std::vector<T> output(_out_shape.FlatSize());
    const int N = _in_shape.Dims(0);
    const int D = _in_shape.FlatSize() / N;

    // EXPECT_ANY_THROW(nnfw::cker::train::CategoricalCrossEntropy(y_pred.data(), y_true.data(),
    //                                                             output.data(), N, D));
  }

  void verifyBackward(const std::vector<T> &y_pred, const std::vector<T> &y_true,
                      const std::vector<T> &expected)
  {
    assert(y_pred.size() == y_true.size());

    std::vector<T> output(_in_shape.FlatSize());
    const int N = _in_shape.Dims(0);
    const int D = _in_shape.FlatSize() / N;

    // nnfw::cker::train::CategoricalCrossEntropyGrad(y_pred.data(), y_true.data(), output.data(), N,
    //                                                D);

    // Don't be panic when it fails after kernel implementation or input is changed.
    // CrossEntropy Gradient formula can be calculated slightly differently depending on the
    // environment because it involes calculations such as log or exp.
    for (int i = 0; i < output.size(); ++i)
    {
      EXPECT_FLOAT_EQ(output[i], expected[i]);
    }
  }

  void throwBackward(const std::vector<T> &y_pred, const std::vector<T> &y_true,
                     const std::vector<T> &expected)
  {
    assert(y_pred.size() == y_true.size());

    std::vector<T> output(_out_shape.FlatSize());
    const int N = _in_shape.Dims(0);
    const int D = _in_shape.FlatSize() / N;

    // EXPECT_ANY_THROW(nnfw::cker::train::CategoricalCrossEntropyGrad(y_pred.data(), y_true.data(),
    //                                                                 output.data(), N, D));
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
    std::vector<float> output(1);
    std::vector<float> expected = {110.0};

    nnfw::cker::train::MSE(nnfw::cker::Shape{2, 3}, y_pred.data(), nnfw::cker::Shape{2, 3},
                           y_true.data(), nnfw::cker::Shape{1}, output.data());

    EXPECT_FLOAT_EQ(output[0], expected[0]);
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

    EXPECT_FLOAT_EQ(output[0], expected[0]);
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

    EXPECT_NE(output[0], expected[0]);
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
  {
    nnfw::cker::Shape in_shape{1, 10};
    nnfw::cker::Shape out_shape{1, 1};

    std::vector<float> y_pred = {2.86E-12, 2.82E-13, 0.99999845, 2.36E-07, 2.91E-16,
                                 2.10E-07, 1.69E-14, 1.21E-17,   1.08E-06, 6.23E-18};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {39.617155};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.verifyForward(y_pred, y_true, expected);
  }

  {
    nnfw::cker::Shape in_shape{2, 10};
    nnfw::cker::Shape out_shape{1, 1};

    std::vector<float> y_pred = {2.86E-12, 2.82E-13, 0.99999845,  2.36E-07, 2.91E-16,
                                 2.10E-07, 1.69E-14, 1.21E-17,    1.08E-06, 6.23E-18,
                                 2.75E-12, 2.71E-13, 0.999998569, 2.35E-07, 2.77E-16,
                                 2.02E-07, 1.54E-14, 1.17E-17,    1.01E-06, 5.97E-18};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {39.638470};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.verifyForward(y_pred, y_true, expected);
  }

  {
    nnfw::cker::Shape in_shape{10, 10};
    nnfw::cker::Shape out_shape{1, 1};

    std::vector<float> y_pred = {
      2.86196349e-12, 2.81945149e-13, 0.99999845,     2.35860654e-07, 2.9058794e-16,
      2.09894068e-07, 1.6895507e-14,  1.21488099e-17, 1.0800934e-06,  6.22995669e-18,
      6.80839918e-08, 2.31582132e-11, 5.66484014e-05, 0.450473011,    7.77803386e-13,
      0.000452301028, 0.541836798,    1.54490024e-07, 0.00718099857,  4.84658429e-08,
      0.00148730644,  0.000732639397, 0.0529860929,   0.104999296,    6.03449457e-10,
      0.000499041693, 0.834561646,    2.97201696e-05, 0.0041414029,   0.000562905625,
      7.88594662e-06, 1.72171513e-05, 0.00883230194,  0.493081957,    7.07157586e-13,
      0.0145716211,   0.482783914,    2.76635546e-07, 0.00042539509,  0.000279496337,
      1.0276257e-07,  3.83042126e-10, 0.0112575656,   0.983353972,    2.16322504e-17,
      3.18806633e-06, 3.15719735e-05, 1.65277847e-10, 0.00535304006,  5.68216365e-07,
      4.83230951e-06, 1.0492589e-09,  0.000612914097, 0.00494865049,  1.38217895e-10,
      0.00836936105,  0.336497307,    8.71851896e-12, 0.649567008,    2.37553195e-08,
      7.90850857e-11, 2.31540633e-12, 0.999991655,    1.34670612e-08, 6.38619986e-06,
      1.88032658e-07, 1.6713152e-06,  1.14870158e-09, 8.11773879e-13, 4.81475465e-15,
      4.71250632e-06, 4.9459074e-12,  0.915183961,    0.0051834262,   2.34058541e-12,
      0.00018049599,  0.0348859429,   2.23697881e-16, 0.0445616208,   2.25251653e-12,
      0.00105480896,  1.76535832e-06, 0.00727920234,  0.036443647,    1.75241883e-06,
      0.954570055,    1.01299993e-05, 0.000419652439, 0.000141307348, 7.75765366e-05,
      0.000144087317, 4.25886909e-10, 0.127607137,    0.87181282,     1.07257775e-14,
      0.000421706965, 2.60532915e-12, 7.31942293e-08, 1.40476577e-05, 9.71598766e-08};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    std::vector<float> expected = {11.531664};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.verifyForward(y_pred, y_true, expected);
  }
}

TEST(CKer_Operation, neg_LossCategoricalCrossEntropy)
{
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
    nnfw::cker::Shape out_shape{1, 1};

    std::vector<float> y_pred = {2.86E-12, 2.82E-13, 0.99999845, 2.36E-07, 2.91E-16,
                                 2.10E-07, 1.69E-14, 1.21E-17,   1.08E-06, 6.23E-18};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, -1.60513648e+17};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.verifyBackward(y_pred, y_true, expected);
  }

  {
    nnfw::cker::Shape in_shape{2, 10};
    nnfw::cker::Shape out_shape{1, 1};

    std::vector<float> y_pred = {2.86E-12, 2.82E-13, 0.99999845,  2.36E-07, 2.91E-16,
                                 2.10E-07, 1.69E-14, 1.21E-17,    1.08E-06, 6.23E-18,
                                 2.75E-12, 2.71E-13, 0.999998569, 2.35E-07, 2.77E-16,
                                 2.02E-07, 1.54E-14, 1.17E-17,    1.01E-06, 5.97E-18};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, -1.60513648e+17,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, -1.67504188e+17};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.verifyBackward(y_pred, y_true, expected);
  }

  {
    nnfw::cker::Shape in_shape{10, 10};
    nnfw::cker::Shape out_shape{1, 1};

    std::vector<float> y_pred = {
      2.86196349e-12, 2.81945149e-13, 0.99999845,     2.35860654e-07, 2.9058794e-16,
      2.09894068e-07, 1.6895507e-14,  1.21488099e-17, 1.0800934e-06,  6.22995669e-18,
      6.80839918e-08, 2.31582132e-11, 5.66484014e-05, 0.450473011,    7.77803386e-13,
      0.000452301028, 0.541836798,    1.54490024e-07, 0.00718099857,  4.84658429e-08,
      0.00148730644,  0.000732639397, 0.0529860929,   0.104999296,    6.03449457e-10,
      0.000499041693, 0.834561646,    2.97201696e-05, 0.0041414029,   0.000562905625,
      7.88594662e-06, 1.72171513e-05, 0.00883230194,  0.493081957,    7.07157586e-13,
      0.0145716211,   0.482783914,    2.76635546e-07, 0.00042539509,  0.000279496337,
      1.0276257e-07,  3.83042126e-10, 0.0112575656,   0.983353972,    2.16322504e-17,
      3.18806633e-06, 3.15719735e-05, 1.65277847e-10, 0.00535304006,  5.68216365e-07,
      4.83230951e-06, 1.0492589e-09,  0.000612914097, 0.00494865049,  1.38217895e-10,
      0.00836936105,  0.336497307,    8.71851896e-12, 0.649567008,    2.37553195e-08,
      7.90850857e-11, 2.31540633e-12, 0.999991655,    1.34670612e-08, 6.38619986e-06,
      1.88032658e-07, 1.6713152e-06,  1.14870158e-09, 8.11773879e-13, 4.81475465e-15,
      4.71250632e-06, 4.9459074e-12,  0.915183961,    0.0051834262,   2.34058541e-12,
      0.00018049599,  0.0348859429,   2.23697881e-16, 0.0445616208,   2.25251653e-12,
      0.00105480896,  1.76535832e-06, 0.00727920234,  0.036443647,    1.75241883e-06,
      0.954570055,    1.01299993e-05, 0.000419652439, 0.000141307348, 7.75765366e-05,
      0.000144087317, 4.25886909e-10, 0.127607137,    0.87181282,     1.07257775e-14,
      0.000421706965, 2.60532915e-12, 7.31942293e-08, 1.40476577e-05, 9.71598766e-08};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    std::vector<float> expected = {
      0,           0, 0,           0,           0, 0,           0, 0,          0, -1.60514765e+17,
      -14687740,   0, 0,           0,           0, 0,           0, 0,          0, 0,
      -672.356384, 0, 0,           0,           0, 0,           0, 0,          0, 0,
      0,           0, 0,           -2.02806044, 0, 0,           0, 0,          0, 0,
      -9731170,    0, 0,           0,           0, 0,           0, 0,          0, 0,
      0,           0, -1631.55005, 0,           0, 0,           0, 0,          0, 0,
      0,           0, 0,           0,           0, 0,           0, -870548096, 0, 0,
      0,           0, -1.09267652, 0,           0, 0,           0, 0,          0, 0,
      0,           0, 0,           0,           0, -1.04759204, 0, 0,          0, 0,
      0,           0, 0,           0,           0, -2371.31494, 0, 0,          0, 0};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.verifyBackward(y_pred, y_true, expected);
  }
}

TEST(CKer_Operation, neg_LossCategoricalCrossEntropyGrad)
{
  {
    nnfw::cker::Shape in_shape{1, 10};
    nnfw::cker::Shape out_shape{1, 1};

    std::vector<float> y_pred = {-2.86E-12, 2.82E-13, 0.99999845, 2.36E-07, 2.91E-16,
                                 2.10E-07,  1.69E-14, 1.21E-17,   1.08E-06, 6.23E-18};
    std::vector<float> y_true = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    std::vector<float> expected = {0, 0, 0, 0, 0, 0, 0, 0, 0, -1.60513648e+17};

    LossCCEVerifier<float> verifier(in_shape, out_shape);
    verifier.throwBackward(y_pred, y_true, expected);
  }
}
