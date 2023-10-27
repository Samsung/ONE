/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ErrorMetric.h"

#include <gtest/gtest.h>

TEST(CircleMPQSolverMAEMetricTest, verifyResultsTest)
{
  size_t num_elements = 512;
  mpqsolver::core::WholeOutput target, source;
  // let target be zero
  {
    std::vector<float> float_buffer(num_elements, 0.f);
    auto const char_buffer = reinterpret_cast<char *>(float_buffer.data());
    auto const char_buffer_size = num_elements * sizeof(float) / sizeof(char);
    std::vector<char> buffer(char_buffer, char_buffer + char_buffer_size);

    mpqsolver::core::Output out = mpqsolver::core::Output(1, buffer);
    target = mpqsolver::core::WholeOutput(1, out);
  }

  // let source be one
  {
    std::vector<float> float_buffer(num_elements, 1.f);
    auto const char_buffer = reinterpret_cast<char *>(float_buffer.data());
    auto const char_buffer_size = num_elements * sizeof(float) / sizeof(char);
    std::vector<char> buffer(char_buffer, char_buffer + char_buffer_size);
    mpqsolver::core::Output out = mpqsolver::core::Output(1, buffer);
    source = mpqsolver::core::WholeOutput(1, out);
  }

  mpqsolver::core::MAEMetric metric;
  float value = metric.compute(target, source);
  EXPECT_FLOAT_EQ(value, 1.f);
}

TEST(CircleMPQSolverMAEMetricTest, verifyComplexResultsTest)
{
  // test for model with a pair of outputs

  size_t num_elements = 512;
  mpqsolver::core::WholeOutput target, source;
  // construct target
  {
    // let first target output be 0.0
    std::vector<float> float_buffer(num_elements, 0.f);
    auto const char_buffer = reinterpret_cast<char *>(float_buffer.data());
    auto const char_buffer_size = num_elements * sizeof(float) / sizeof(char);
    std::vector<char> buffer(char_buffer, char_buffer + char_buffer_size);

    // first target output
    mpqsolver::core::Output out = mpqsolver::core::Output(1, buffer);
    {
      // let second target output be 0.25
      std::vector<float> float_buffer(num_elements / 2, 0.25f);
      auto const char_buffer = reinterpret_cast<char *>(float_buffer.data());
      auto const char_buffer_size = num_elements / 2 * sizeof(float) / sizeof(char);
      std::vector<char> buffer(char_buffer, char_buffer + char_buffer_size);
      // second target output
      out.emplace_back(buffer);
    }

    // target consists of single sample (each sample consists of two outputs)
    target = mpqsolver::core::WholeOutput(1, out);
  }

  // construct target
  {
    // let first source output be 1.f
    std::vector<float> float_buffer(num_elements, 1.f);
    auto const char_buffer = reinterpret_cast<char *>(float_buffer.data());
    auto const char_buffer_size = num_elements * sizeof(float) / sizeof(char);
    std::vector<char> buffer(char_buffer, char_buffer + char_buffer_size);
    // first source output
    mpqsolver::core::Output out = mpqsolver::core::Output(1, buffer);
    {
      // let second source output be 1.f
      std::vector<float> float_buffer(num_elements / 2, 1.0f);
      auto const char_buffer = reinterpret_cast<char *>(float_buffer.data());
      auto const char_buffer_size = num_elements / 2 * sizeof(float) / sizeof(char);
      std::vector<char> buffer(char_buffer, char_buffer + char_buffer_size);
      // second source output
      out.emplace_back(buffer);
    }

    // source consists of single sample (each sample consists of two outputs)
    source = mpqsolver::core::WholeOutput(1, out);
  }

  mpqsolver::core::MAEMetric metric;
  float value = metric.compute(target, source);
  EXPECT_FLOAT_EQ(value, 0.875f); // (|0 - 1| + |0.25 - 1|) / 2 = 1.75 / 2 = 0.875
}

TEST(CircleMPQSolverMAEMetricTest, different_samples_NEG)
{
  mpqsolver::core::MAEMetric metric;

  size_t num_elements = 512;
  std::vector<float> float_buffer(num_elements, 0.f);
  auto const char_buffer = reinterpret_cast<char *>(float_buffer.data());
  auto const char_buffer_size = num_elements * sizeof(float) / sizeof(char);
  std::vector<char> buffer(char_buffer, char_buffer + char_buffer_size);
  mpqsolver::core::Output out = mpqsolver::core::Output(1, buffer);

  // let target be zero of two samples
  mpqsolver::core::WholeOutput target = mpqsolver::core::WholeOutput(2, out);

  // let source be zero of single sample
  mpqsolver::core::WholeOutput source;

  EXPECT_ANY_THROW(metric.compute(target, source));
}

TEST(CircleMPQSolverMAEMetricTest, verifyResultsTest_NEG)
{
  mpqsolver::core::MAEMetric metric;
  mpqsolver::core::WholeOutput target, source;
  EXPECT_ANY_THROW(metric.compute(target, source));
}
