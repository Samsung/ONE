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

#include <cker/train/optimizer/Adam.h>

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

// TODO Add tests that verifies result values
template <typename T> class AdamOptimizerVerifier
{
public:
  AdamOptimizerVerifier(const std::vector<T> &trainable, const std::vector<T> &gradient,
                        const std::vector<T> &m, const std::vector<T> &v, float learning_rate,
                        float beta1, float beta2, float epsilon, bool use_nesterov,
                        uint32_t nums_step)
    : _trainable{trainable}, _gradient{gradient}, _m{m}, _v{v}, _learning_rate{learning_rate},
      _beta1{beta1}, _beta2{beta2}, _epsilon{epsilon}, _use_nesterov{use_nesterov},
      _nums_step{nums_step}
  {
    EXPECT_TRUE(trainable.size() == gradient.size());
    EXPECT_TRUE(trainable.size() == m.size());
    EXPECT_TRUE(trainable.size() == v.size());

    _expected.resize(trainable.size());
    std::copy(trainable.begin(), trainable.end(), _expected.begin());
  }

public:
  void verify()
  {
    for (uint32_t step = 0; step < _nums_step; ++step)
    {
      const T beta1_power = std::pow(_beta1, step + 1);
      const T beta2_power = std::pow(_beta2, step + 1);

      calculateExpected(beta1_power, beta2_power);

      nnfw::cker::train::Adam(
        nnfw::cker::Shape{static_cast<int>(_trainable.size())}, _trainable.data(),
        nnfw::cker::Shape{static_cast<int>(_gradient.size())}, _gradient.data(),
        nnfw::cker::Shape{static_cast<int>(_m.size())}, _m.data(),
        nnfw::cker::Shape{static_cast<int>(_v.size())}, _v.data(), beta1_power, beta2_power,
        _learning_rate, _beta1, _beta2, _epsilon, _use_nesterov);

      for (size_t i = 0; i < _trainable.size(); ++i)
        EXPECT_NEAR(_trainable[i], _expected[i], 1e-5f);
    }
  }

private:
  void calculateExpected(const float beta1_power, const float beta2_power)
  {
    assert(_expected.size() == _m.size());
    assert(_expected.size() == _v.size());
    assert(_expected.size() == _gradient.size());

    const T alpha = _learning_rate * std::sqrt(static_cast<T>(1) - beta2_power) /
                    (static_cast<T>(1) - beta1_power);
    for (int i = 0; i < _expected.size(); ++i)
    {
      T m = _m.at(i);
      T v = _v.at(i);
      T g = _gradient.at(i);
      T &var = _expected.at(i);

      if (_use_nesterov)
      {
        m += (g - m) * (static_cast<T>(1) - _beta1);
        v += (std::pow(g, 2) - v) * (static_cast<T>(1) - _beta2);
        var -=
          ((g * (static_cast<T>(1) - _beta1) + _beta1 * m) * alpha) / (std::sqrt(v) + _epsilon);
      }
      else
      {
        m += (g - m) * (static_cast<T>(1) - _beta1);
        v += (std::pow(g, 2) - v) * (static_cast<T>(1) - _beta2);
        var -= (m * alpha) / (std::sqrt(v) + _epsilon);
      }
    }
  }

private:
  std::vector<T> _trainable;
  std::vector<T> _gradient;
  std::vector<T> _m;
  std::vector<T> _v;
  float _learning_rate;
  float _beta1;
  float _beta2;
  float _epsilon;
  bool _use_nesterov;
  uint32_t _nums_step;
  std::vector<T> _expected;
};

TEST(CKer_Optimizer, AdamSimple)
{
  // Unmatched shape
  {
    std::vector<float> trainable = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient = {-1, 2, -3, 4, 5, -6, 7, 8, 9};
    std::vector<float> m = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-07;
    bool use_nesterov = true;
    uint32_t training_steps = 10;

    AdamOptimizerVerifier<float>{trainable,    gradient,      m, v, lr, beta1, beta2, epsilon,
                                 use_nesterov, training_steps}
      .verify();
  }
}

TEST(CKer_Optimizer, AdamSteps1e5)
{
  // Unmatched shape
  {
    std::vector<float> trainable = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> m = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-07;
    bool use_nesterov = false;
    uint32_t training_steps = 10000;

    AdamOptimizerVerifier<float>{trainable,    gradient,      m, v, lr, beta1, beta2, epsilon,
                                 use_nesterov, training_steps}
      .verify();
  }
}

TEST(CKer_Optimizer, neg_AdamUnmatchedGradientShape)
{
  // Unmatched shape
  {
    std::vector<float> trainable = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient = {-1, 2, -3, 4, 5, -6, 7, 8};
    std::vector<float> m = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-07;
    bool use_nesterov = false;
    float beta1_power = std::pow(beta1, 1);
    float beta2_power = std::pow(beta2, 1);

    EXPECT_ANY_THROW(nnfw::cker::train::Adam(
      nnfw::cker::Shape{static_cast<int>(trainable.size())}, trainable.data(),
      nnfw::cker::Shape{static_cast<int>(gradient.size())}, gradient.data(),
      nnfw::cker::Shape{static_cast<int>(m.size())}, m.data(),
      nnfw::cker::Shape{static_cast<int>(v.size())}, v.data(), beta1_power, beta2_power, lr, beta1,
      beta2, epsilon, use_nesterov));
  }
}

TEST(CKer_Optimizer, neg_AdamUnmatchedEMAShape1)
{
  // Unmatched shape
  {
    std::vector<float> trainable = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient = {-1, 2, -3, 4, 5, -6, 7, 8, 9};
    std::vector<float> m = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-07;
    bool use_nesterov = false;
    float beta1_power = std::pow(beta1, 1);
    float beta2_power = std::pow(beta2, 1);

    EXPECT_ANY_THROW(nnfw::cker::train::Adam(
      nnfw::cker::Shape{static_cast<int>(trainable.size())}, trainable.data(),
      nnfw::cker::Shape{static_cast<int>(gradient.size())}, gradient.data(),
      nnfw::cker::Shape{static_cast<int>(m.size())}, m.data(),
      nnfw::cker::Shape{static_cast<int>(v.size())}, v.data(), beta1_power, beta2_power, lr, beta1,
      beta2, epsilon, use_nesterov));
  }
}

TEST(CKer_Optimizer, neg_AdamUnmatchedEMAShape2)
{
  // Unmatched shape
  {
    std::vector<float> trainable = {-1, 2, -3, 4, 5, -6, -7, 8, 9};
    std::vector<float> gradient = {-1, 2, -3, 4, 5, -6, 7, 8, 9};
    std::vector<float> m = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v = {0, 0, 0, 0, 0, 0, 0, 0};
    float lr = 0.001;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float epsilon = 1e-07;
    bool use_nesterov = false;
    float beta1_power = std::pow(beta1, 1);
    float beta2_power = std::pow(beta2, 1);

    EXPECT_ANY_THROW(nnfw::cker::train::Adam(
      nnfw::cker::Shape{static_cast<int>(trainable.size())}, trainable.data(),
      nnfw::cker::Shape{static_cast<int>(gradient.size())}, gradient.data(),
      nnfw::cker::Shape{static_cast<int>(m.size())}, m.data(),
      nnfw::cker::Shape{static_cast<int>(v.size())}, v.data(), beta1_power, beta2_power, lr, beta1,
      beta2, epsilon, use_nesterov));
  }
}
