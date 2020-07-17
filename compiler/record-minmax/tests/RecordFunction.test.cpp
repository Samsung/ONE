/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "RecordFunction.h"

#include <vector>
#include <cmath>

#include <gtest/gtest.h>

namespace record_minmax
{

#define EXPECT_FLOAT_NEAR(exp, val) EXPECT_NEAR(exp, val, 1e-5 + 1e-5 * std::abs(exp))

TEST(GetNthPercentileTest, Edge)
{
  std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  EXPECT_FLOAT_NEAR(0, getNthPercentile(input, 0));
  EXPECT_FLOAT_NEAR(9, getNthPercentile(input, 100));
}

TEST(GetNthPercentileTest, Simple)
{
  std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (float i = 1; i <= 99; i++)
  {
    EXPECT_FLOAT_NEAR(0.09 * i, getNthPercentile(input, i));
  }

  for (float i = 0.5; i <= 99.5; i++)
  {
    EXPECT_FLOAT_NEAR(0.09 * std::floor(i) + 0.045, getNthPercentile(input, i));
  }
}

TEST(GetNthPercentileTest, Float)
{
  std::vector<float> input{8.48424583,  89.39998456, 65.83323245, 87.85243858, 68.85414866,
                           98.40591775, 16.74266565, 25.09415131, 74.54084952, 29.70536481,
                           49.26803928, 79.49602425, 53.69395631, 73.73140271, 99.81245733,
                           46.76997646, 78.37688474, 10.43076744, 30.39480496, 14.30875609,
                           86.72073486, 17.97364969, 14.66724564, 0.47818459,  17.77138025,
                           85.68981239, 22.18322696, 78.81541331, 93.04085581, 40.2147895};

  EXPECT_FLOAT_NEAR(2.799942346802177, getNthPercentile(input, 1));
  EXPECT_FLOAT_NEAR(7.768503955476342, getNthPercentile(input, 3.14));
  EXPECT_FLOAT_NEAR(99.40456084968194, getNthPercentile(input, 99));
}

TEST(GetNthPercentileTest, FloatWithNegative)
{
  std::vector<float> input{-41.51575417, 39.39998456,  15.83323245,  37.85243858,  18.85414866,
                           48.40591775,  -33.25733435, -24.90584869, 24.54084952,  -20.29463519,
                           -0.73196072,  29.49602425,  3.69395631,   23.73140271,  49.81245733,
                           -3.23002354,  28.37688474,  -39.56923256, -19.60519504, -35.69124391,
                           36.72073486,  -32.02635031, -35.33275436, -49.52181541, -32.22861975,
                           35.68981239,  -27.81677304, 28.81541331,  43.04085581,  -9.7852105};

  EXPECT_FLOAT_NEAR(-47.20005765319782, getNthPercentile(input, 1));
  EXPECT_FLOAT_NEAR(-42.23149604452366, getNthPercentile(input, 3.14));
  EXPECT_FLOAT_NEAR(49.40456084968194, getNthPercentile(input, 99));
}

TEST(GetNthPercentileTest, SigleElement)
{
  std::vector<float> input{33};

  EXPECT_FLOAT_NEAR(33, getNthPercentile(input, 0));
  EXPECT_FLOAT_NEAR(33, getNthPercentile(input, 50));
  EXPECT_FLOAT_NEAR(33, getNthPercentile(input, 100));
}

TEST(GetNthPercentileTest, OutOfBoundary_NEG)
{
  std::vector<float> input{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  EXPECT_THROW(getNthPercentile(input, -1), std::runtime_error);
  EXPECT_THROW(getNthPercentile(input, 101), std::runtime_error);
}

TEST(GetNthPercentileTest, EmptyVector_NEG)
{
  std::vector<float> input;

  EXPECT_THROW(getNthPercentile(input, 10), std::runtime_error);
}

} // namespace record_minmax
