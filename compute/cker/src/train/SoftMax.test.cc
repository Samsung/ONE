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

#include <cker/train/operation/SoftMax.h>

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

// TODO Add tests that verifies result values
TEST(CKer_Operation, SoftMaxGrad)
{
  // Unmatched shape
  {
    // Dim 1x10
    nnfw::cker::Shape shape{1, 10};
    std::vector<float> softmax = {2.86195412e-12, 2.81944090e-13, 9.99998474e-01, 2.35859203e-07,
                                  2.90586864e-16, 2.09893085e-07, 1.68954109e-14, 1.21487884e-17,
                                  1.08008625e-06, 6.22994465e-18};
    std::vector<float> incoming = {5.72000000e-13, 5.64000000e-14, 1.99999690e-01, 4.72000000e-08,
                                   5.81000000e-17, 4.20000000e-08, 3.38000000e-15, 2.43000000e-18,
                                   2.16000000e-07, -2.00000003e-01};
    std::vector<float> expected = {
      -5.72389063e-13, -5.63886446e-14, 3.05167149e-07,  -4.71716844e-08, -5.81171941e-17,
      -4.19784790e-08, -3.37907178e-15, -2.42975021e-18, -2.16016353e-07, -2.49197405e-18};
    std::vector<float> grad(10);

    nnfw::cker::train::SoftMaxGrad(shape, softmax.data(), shape, incoming.data(), shape,
                                   grad.data());

    // TODO Use EXPECT_FLOAT_EQUAL
    for (size_t i = 0; i < grad.size(); ++i)
      EXPECT_NEAR(grad[i], expected[i], 0.01);
  }

  {
    // Dim 2x10
    nnfw::cker::Shape shape{2, 10};
    std::vector<float> softmax = {6.80841727e-08, 2.31582313e-11, 5.66484244e-05, 4.50472506e-01,
                                  7.77810687e-13, 4.52301134e-04, 5.41837231e-01, 1.54489112e-07,
                                  7.18104184e-03, 4.84659012e-08, 1.51924287e-03, 7.48874448e-04,
                                  5.43233175e-02, 1.10145863e-01, 6.17917826e-10, 5.09775521e-04,
                                  8.27892599e-01, 3.03146885e-05, 4.25460567e-03, 5.75406837e-04};
    std::vector<float> incoming = {
      -1.99999988e-01, 4.63000000e-12, 1.13000000e-05, 9.00945070e-02, 1.56000000e-13,
      9.05000000e-05,  1.08367443e-01, 3.09000000e-08, 1.43620800e-03, 9.69000000e-09,
      -1.99999988e-01, 4.63000000e-12, 1.13000000e-05, 9.00945070e-02, 1.56000000e-13,
      9.05000000e-05,  1.08367443e-01, 3.09000000e-08, 1.43620800e-03, 9.69000000e-09};
    std::vector<float> expected = {
      -2.03784741e-08, -2.29991239e-12, -5.62528230e-06, -4.15265741e-03, -7.72466778e-14,
      -4.48784290e-05, 4.90605867e-03,  -1.53427655e-08, -7.02857016e-04, -4.81329140e-09,
      -5.15774553e-04, -1.04579225e-04, -7.00410377e-03, -1.29717840e-02, -8.63838721e-11,
      -7.12137857e-05, 2.13432343e-02,  -4.23775872e-06, -5.91166379e-04, -8.03746891e-05};
    std::vector<float> grad(20);

    nnfw::cker::train::SoftMaxGrad(shape, softmax.data(), shape, incoming.data(), shape,
                                   grad.data());

    // TODO Use EXPECT_FLOAT_EQUAL
    for (size_t i = 0; i < grad.size(); ++i)
      EXPECT_NEAR(grad[i], expected[i], 0.1);
  }
}

TEST(CKer_Operation, neg_SoftMaxGrad)
{
  // Invalid expected value
  {
    // Dim 1x10
    nnfw::cker::Shape shape{1, 10};
    std::vector<float> softmax = {7.96183250e-06, 1.73761995e-05, 9.35646649e-03, 5.53923216e-01,
                                  7.15798787e-13, 1.46669847e-02, 4.21307124e-01, 2.77163556e-07,
                                  4.36947878e-04, 2.83646234e-04};
    std::vector<float> incoming = {1.5900000e-06, 3.4800000e-06, 1.8712930e-03, -8.9215361e-02,
                                   1.4300000e-13, 2.9333970e-03, 8.4261417e-02, 5.5400000e-08,
                                   8.7400000e-05, 5.6700000e-05};
    std::vector<float> expected = {0.001, 0.002, 0.003, 0.001, 0.002,
                                   0.003, 0.001, 0.002, 0.003, 0.001};
    std::vector<float> grad(10);

    nnfw::cker::train::SoftMaxGrad(shape, softmax.data(), shape, incoming.data(), shape,
                                   grad.data());

    for (size_t i = 0; i < grad.size(); ++i)
      EXPECT_NE(grad[i], expected[i]);
  }
}
