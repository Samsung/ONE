/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "KernelShapeUtil.h"
#include "Errors.h"

#include <gtest/gtest.h>

using namespace mlir::Circle;

TEST(KernelShapeUtilTest, GetWindowedOutputSizeVerboseV2_NEG)
{
  // negative test for invalid stride
  int64_t stride = 0; // stride <= 0 should return not ok
  auto padding = Padding::VALID;
  auto status = GetWindowedOutputSizeVerboseV2(0, 0, 0, stride, padding, nullptr, nullptr, nullptr);
  ASSERT_FALSE(status.ok());

  // negative test for invalid dialation_rate
  stride = 1;                 // valid stride
  int64_t dilation_rate = -1; // dialation_rate < 0 should return not ok
  status =
    GetWindowedOutputSizeVerboseV2(0, 0, dilation_rate, stride, padding, nullptr, nullptr, nullptr);
  ASSERT_FALSE(status.ok());

  // negative test EXPLICIT, *output_size < 0
  stride = 1;        // valid stride
  dilation_rate = 1; // valid dilation_rate
  padding = Padding::EXPLICIT;
  int64_t output_size, padding_before = -2, padding_after = -2;
  status = GetWindowedOutputSizeVerboseV2(0, 0, dilation_rate, stride, padding, &output_size,
                                          &padding_before, &padding_after);
  ASSERT_FALSE(status.ok());

  // negative test SAME, *output_size < 0
  stride = 1;        // valid stride
  dilation_rate = 1; // valid dilation_rate
  padding = Padding::SAME;
  output_size = 0, padding_before = 0, padding_after = 0;
  status = GetWindowedOutputSizeVerboseV2(-100, 0, dilation_rate, stride, padding, &output_size,
                                          &padding_before, &padding_after);
  ASSERT_FALSE(status.ok());
}

TEST(KernelShapeUtilTest, GetWindowedOutputSizeVerbose_NEG)
{
  // negative test for invalid stride
  int64_t stride = 0; // stride <= 0 should return not ok
  auto padding = Padding::VALID;
  auto status = GetWindowedOutputSizeVerbose(-100, 0, stride, padding, nullptr, nullptr, nullptr);
  ASSERT_FALSE(status.ok());
}

TEST(KernelShapeUtilTest, GetWindowedOutputSize_NEG)
{
  // negative test for EXPLICIT
  auto padding = Padding::EXPLICIT;
  int64_t output_size, padding_size = 0;
  auto status = GetWindowedOutputSize(-100, 0, 0, padding, &output_size, &padding_size);
  ASSERT_FALSE(status.ok());

  // negative test for invalid input size
  padding = Padding::VALID;
  status = GetWindowedOutputSize(-100, 0, 0, padding, &output_size, &padding_size);
  ASSERT_FALSE(status.ok());
}

TEST(KernelShapeUtilTest, GetWindowedOutputSizeV2_NEG)
{
  // negative test for EXPLICIT
  auto padding = Padding::EXPLICIT;
  int64_t dilation_rate = 1; // valid dilation_rate
  int64_t output_size, padding_size = 0;
  auto status =
    GetWindowedOutputSizeV2(-100, 0, 0, dilation_rate, padding, &output_size, &padding_size);
  ASSERT_FALSE(status.ok());

  // negative test for invalid input size
  padding = Padding::VALID;
  status = GetWindowedOutputSizeV2(-100, 0, 0, dilation_rate, padding, &output_size, &padding_size);
  ASSERT_FALSE(status.ok());
}

TEST(KernelShapeUtilTest, Get3dOutputSize_NEG)
{
  // negative test with EXPLICIT
  std::array<int64_t, 3> input = {0, 0, 0};
  std::array<int64_t, 3> window = {0, 0, 0};
  std::array<int64_t, 3> strides = {1, 1, 1};
  auto padding = Padding::EXPLICIT;
  std::array<int64_t, 3> outputs = {0, 0, 0};
  std::array<int64_t, 3> padding_out = {0, 0, 0};

  auto status = Get3dOutputSize(input, window, strides, padding, &outputs, &padding_out);
  ASSERT_FALSE(status.ok());
}

TEST(KernelShapeUtilTest, Get3dOutputSize)
{
  // positive test
  std::array<int64_t, 3> input = {10, 10, 10};
  std::array<int64_t, 3> window = {1, 1, 1};
  std::array<int64_t, 3> strides = {1, 1, 1};
  auto padding = Padding::VALID;
  std::array<int64_t, 3> outputs = {0, 0, 0};
  std::array<int64_t, 3> padding_out = {0, 0, 0};

  auto status = Get3dOutputSize(input, window, strides, padding, &outputs, &padding_out);
  ASSERT_TRUE(status.ok());
}

TEST(KernelShapeUtilTest, Get3dOutputSizeV2_NEG)
{
  // negative test with EXPLICIT
  std::array<int64_t, 3> input = {0, 0, 0};
  std::array<int64_t, 3> window = {0, 0, 0};
  std::array<int64_t, 3> dilations = {1, 1, 1};
  std::array<int64_t, 3> strides = {1, 1, 1};
  auto padding = Padding::EXPLICIT;
  std::array<int64_t, 3> outputs = {0, 0, 0};
  std::array<int64_t, 3> padding_out = {0, 0, 0};

  auto status =
    Get3dOutputSizeV2(input, window, dilations, strides, padding, &outputs, &padding_out);
  ASSERT_FALSE(status.ok());
}

TEST(KernelShapeUtilTest, Get3dOutputSizeV2)
{
  // positive test
  std::array<int64_t, 3> input = {10, 10, 10};
  std::array<int64_t, 3> window = {1, 1, 1};
  std::array<int64_t, 3> dilations = {1, 1, 1};
  std::array<int64_t, 3> strides = {1, 1, 1};
  auto padding = Padding::VALID;
  std::array<int64_t, 3> outputs = {0, 0, 0};
  std::array<int64_t, 3> padding_out = {0, 0, 0};

  auto status =
    Get3dOutputSizeV2(input, window, dilations, strides, padding, &outputs, &padding_out);
  ASSERT_TRUE(status.ok());
}
