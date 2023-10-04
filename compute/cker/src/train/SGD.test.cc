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

#include <cker/train/optimizer/SGD.h>

#include <gtest/gtest.h>
#include <vector>

// TODO Add tests that verifies result values
template <typename T> class SGDOptimizerVerifier
{
public:
  SGDOptimizerVerifier(const std::vector<T> &trainable, const std::vector<T> &gradient,
                       float learning_rate, uint32_t nums_step)
    : _trainable{trainable}, _gradient{gradient}, _learning_rate{learning_rate}, _nums_step{
                                                                                   nums_step}
  {
    EXPECT_TRUE(trainable.size() == gradient.size());

    _expected.resize(trainable.size());
    std::copy(trainable.begin(), trainable.end(), _expected.begin());
  }

public:
  void verify()
  {
    for (uint32_t step = 0; step < _nums_step; ++step)
    {
      calculateExpected();

      nnfw::cker::train::GradientDescent(
        nnfw::cker::Shape{static_cast<int>(_trainable.size())}, _trainable.data(),
        nnfw::cker::Shape{static_cast<int>(_gradient.size())}, _gradient.data(), _learning_rate);

      for (size_t i = 0; i < _trainable.size(); ++i)
        EXPECT_NEAR(_trainable[i], _expected[i], 1e-5f);
    }
  }

private:
  void calculateExpected()
  {
    assert(_expected.size() == _gradient.size());

    for (int i = 0; i < _expected.size(); ++i)
    {
      T g = _gradient.at(i);
      T &var = _expected.at(i);

      var -= g * _learning_rate;
    }
  }

private:
  std::vector<T> _trainable;
  std::vector<T> _gradient;
  float _learning_rate;
  uint32_t _nums_step;
  std::vector<T> _expected;
};

TEST(CKer_Optimizer, SGDSimple)
{
  {
    std::vector<float> trainable = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient = {-1, 2, -3, 4, 5, -6, 7, 8, 9};
    std::vector<float> m = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float lr = 0.001;
    uint32_t training_steps = 10000;

    SGDOptimizerVerifier<float>{trainable, gradient, lr, training_steps}.verify();
  }
}

TEST(CKer_Optimizer, neg_SGDUnmatchedGradientShape)
{
  // Unmatched shape
  {
    std::vector<float> trainable = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient = {-1, 2, -3, 4, 5, -6, 7, 8};
    std::vector<float> m = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float lr = 0.001;

    EXPECT_ANY_THROW(nnfw::cker::train::GradientDescent(
      nnfw::cker::Shape{static_cast<int>(trainable.size())}, trainable.data(),
      nnfw::cker::Shape{static_cast<int>(gradient.size())}, gradient.data(), lr));
  }
}
