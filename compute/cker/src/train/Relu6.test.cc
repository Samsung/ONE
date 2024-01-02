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

#include <cker/operation/ReLU6.h>
#include <cker/train/operation/ReLU6.h>

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include <vector>

namespace
{

using namespace nnfw::cker;

template <typename T> class Relu6OpVerifier
{
public:
  void verifyForward(const std::vector<T> &input, const std::vector<T> &expected_output)
  {
    assert(input.size() == expected_output.size());

    std::vector<T> calc_output(input.size()); // calcuated output
    ReLU6(Shape{static_cast<int>(input.size())}, input.data(), calc_output.data());

    for (size_t i = 0; i < calc_output.size(); ++i)
      ASSERT_EQ(expected_output[i], calc_output[i]);
  }

  void verifyBackward(const std::vector<T> &output, const std::vector<T> &input_bwd,
                      const std::vector<T> &expected_output_bwd, bool expect_eq = true)
  {
    std::vector<T> calc_output_bwd(input_bwd.size()); // calculated output backward
    train::ReLU6Grad(Shape{static_cast<int>(output.size())}, output.data(),
                     Shape{static_cast<int>(input_bwd.size())}, input_bwd.data(),
                     Shape{static_cast<int>(calc_output_bwd.size())}, calc_output_bwd.data());

    if (expect_eq)
      EXPECT_EQ(expected_output_bwd, calc_output_bwd);
    else
      EXPECT_NE(expected_output_bwd, calc_output_bwd);
  }
};

} // namespace

TEST(CKer_Operation, ReLU6)
{
  {
    Relu6OpVerifier<float> verifier;

    // clang-format off
    // std::vector<float> input_fwd         = {-2.0,  -1.0,  2.0,  3.0,  6.0,  7.0};
    std::vector<float> output_fwd           = { 0.0,   0.0,  2.0,  3.0,  6.0,  7.0};  
    std::vector<float> input_bwd            = {-0.1,  -0.2,  0.3,  0.4, -0.1,  0.5};
    std::vector<float> expected_output_bwd  = { 0.0,   0.0,  0.3,  0.4,  0.0,  0.0};
    // clang-format on

    verifier.verifyBackward(output_fwd, input_bwd, expected_output_bwd);
  }

  {
    Relu6OpVerifier<float> verifier;

    // clang-format off
    // std::vector<float> input_fwd         = { 7.0,   8.0,  4.0, -4.0, -5.0, 10.0};
    std::vector<float> output_fwd           = { 6.0,   6.0,  4.0,  0.0,  0.0,  6.0};  
    std::vector<float> input_bwd            = {-6.1,  -3.3,  7.0,  8.4, -9.2,  0.0};
    std::vector<float> expected_output_bwd  = { 0.0,   0.0,  7.0,  0.0,  0.0,  0.0};
    // clang-format on

    verifier.verifyBackward(output_fwd, input_bwd, expected_output_bwd);
  }
}

TEST(CKer_Operation, neg_ReLU6)
{
  {
    Relu6OpVerifier<float> verifier;

    // clang-format off
    // std::vector<float> input_fwd         = { 0.0,   2.0,  4.0,  6.0,  8.0, 10.0};
    std::vector<float> output_fwd           = { 0.0,   2.0,  4.0,  6.0,  6.0,  6.0};  
    std::vector<float> input_bwd            = { 0.1,   0.2,  0.3,  0.4,  0.5,  0.6};
    std::vector<float> expected_output_bwd  = { 0.1,   0.2,  0.3,  0.4,  0.5,  0.6};  // wrong value
    // clang-format on

    verifier.verifyBackward(output_fwd, input_bwd, expected_output_bwd, false);
  }

  {
    Relu6OpVerifier<float> verifier;

    // clang-format off
    // std::vector<float> input_fwd         = { 0.0,   2.0,  4.0,  6.0,  8.0, 10.0};
    std::vector<float> output_fwd           = { 0.0,   2.0,  4.0,  6.0,  6.0,  6.0};  
    std::vector<float> input_bwd            = { 0.1,   0.2,  0.3,  0.4};  // size mismatch
    std::vector<float> expected_output_bwd  = { 0.0,   0.2,  0.3,  0.4};
    // clang-format on

    EXPECT_ANY_THROW(verifier.verifyBackward(output_fwd, input_bwd, expected_output_bwd, false));
  }
}
