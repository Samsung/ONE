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

#include <cker/operation/ReLU.h>
#include <cker/train/operation/ReLU.h>

#include <gtest/gtest.h>
#include <vector>

namespace
{

template <typename T> class ReluOpVerifier
{
public:
  ReluOpVerifier(const std::vector<T> &input, const std::vector<T> &expected_output,
                 const std::vector<T> &backprop_output,
                 const std::vector<T> &expected_backprop_input)
    : _input{input}, _expected_output{expected_output}, _backprop_output{backprop_output},
      _expected_backprop_input{expected_backprop_input}
  {
    EXPECT_TRUE(input.size() == expected_output.size());
    _output.resize(_expected_output.size());
    _backprop_input.resize(_expected_backprop_input.size());
  }

public:
  void verifyExpected()
  {
    nnfw::cker::ReLU(nnfw::cker::Shape{static_cast<int>(_input.size())}, _input.data(),
                     nnfw::cker::Shape{static_cast<int>(_output.size())}, _output.data());

    for (size_t i = 0; i < _output.size(); ++i)
      ASSERT_EQ(_output[i], _expected_output[i]);

    if (_backprop_output.size() > 0)
    {
      nnfw::cker::train::ReLUGrad(
        nnfw::cker::Shape{static_cast<int>(_output.size())}, _output.data(),
        nnfw::cker::Shape{static_cast<int>(_backprop_output.size())}, _backprop_output.data(),
        nnfw::cker::Shape{static_cast<int>(_backprop_input.size())}, _backprop_input.data());

      for (size_t i = 0; i < _backprop_input.size(); ++i)
        ASSERT_EQ(_backprop_input[i], _expected_backprop_input[i]);
    }
  }

private:
  std::vector<T> _input;
  std::vector<T> _output;
  std::vector<T> _expected_output;
  std::vector<T> _backprop_output;
  std::vector<T> _backprop_input;
  std::vector<T> _expected_backprop_input;
};

} // namespace

TEST(CKer_Operation, ReLU)
{
  {
    std::vector<float> input_forward = {-1, 2, 3, -4};
    std::vector<float> expected_forward = {0, 2, 3, 0};
    std::vector<float> incoming_backward = {-5, 6, -7, 8};
    std::vector<float> expected_backward = {0, 6, -7, 0};
    ReluOpVerifier<float> verifier{input_forward, expected_forward, incoming_backward,
                                   expected_backward};
    verifier.verifyExpected();
  }

  {
    std::vector<float> input_forward = {0, -1, 2, 3, -4, 5, 6, -7};
    std::vector<float> expected_forward = {0, 0, 2, 3, 0, 5, 6, 0};
    std::vector<float> incoming_backward = {8, -9, 10, 11, -12, -13, 14, -15};
    std::vector<float> expected_backward = {0, 0, 10, 11, 0, -13, 14, 0};
    ReluOpVerifier<float> verifier{input_forward, expected_forward, incoming_backward,
                                   expected_backward};
    verifier.verifyExpected();
  }
}

TEST(CKer_Operation, neg_ReLU)
{
  {
    // Unmatched shape
    std::vector<float> input_forward = {0, -1, 2, 3, -4};
    std::vector<float> expected_forward = {0, 0, 2, 3, 0};
    std::vector<float> incoming_backward = {-5, 6, -7, 8};
    std::vector<float> expected_backward = {0, 6, -7, 0};
    ReluOpVerifier<float> verifier{input_forward, expected_forward, incoming_backward,
                                   expected_backward};
    EXPECT_ANY_THROW(verifier.verifyExpected());
  }
}
