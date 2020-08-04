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

#include <gtest/gtest.h>
#include <nnfw_internal.h>

#include "fixtures.h"
#include "NNPackages.h"
#include "common.h"

using TestInputReshapingAddModelLoaded = ValidationTestModelLoaded<NNPackages::INPUT_RESHAPING_ADD>;

/**
 * @brief Testing the following model:
 *       #1 = placeholder (shape = [2, 2], dtype=float)
 *       #2 = placeholder (shape = [2], dtype=float)
 *       #3 = add(#1, #2)
 *
 * @note Run this test with "cpu" backend and "linear" executor
 */
TEST_F(TestInputReshapingAddModelLoaded, reshaping_2x2_to_4x2)
{
  NNFW_STATUS res = NNFW_STATUS_ERROR;

  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_set_config(_session, "EXECUTOR", "Linear"), NNFW_STATUS_NO_ERROR);

  // input and output values
  const std::vector<float> input1 = {0, 1, 2, 3, 4, 5, 6, 7}; // of changed shape [4, 2]
  const std::vector<float> input2 = {-10, -10};
  const std::vector<float> expected = {-10, -9, -8, -7, -6, -5, -4, -3}; // of shape [4, 2]

  /*
  testing sequence and what's been done:
    1. nnfw_set_input_tensorinfo : set input shape to different shape (static inference)
    2. nnfw_prepare
    3. nnfw_set_input
    4. nnfw_run
  */

  // input reshaping from [2, 2] to [4, 2]
  nnfw_tensorinfo ti = {NNFW_TYPE_TENSOR_FLOAT32, 2, {4, 2}};
  res = nnfw_set_input_tensorinfo(_session, 0, &ti);

  res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  nnfw_tensorinfo ti_input = {}; // Static inference result will be stored
  nnfw_input_tensorinfo(_session, 0, &ti_input);
  ASSERT_TRUE(tensorInfoEqual(ti, ti_input));

  nnfw_tensorinfo ti_output = {}; // Static inference result will be stored
  nnfw_output_tensorinfo(_session, 0, &ti_output);
  ASSERT_TRUE(tensorInfoEqual(ti, ti_output)); // input/output shapes are same with for this model

  res = nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input1.data(),
                       sizeof(float) * input1.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);
  res = nnfw_set_input(_session, 1, NNFW_TYPE_TENSOR_FLOAT32, input2.data(),
                       sizeof(float) * input2.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  uint64_t output_num_elements = tensorInfoNumElements(ti_output);
  ASSERT_EQ(output_num_elements, expected.size());
  std::vector<float> actual_output(output_num_elements);
  res = nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output.data(),
                        sizeof(float) * actual_output.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // Do inference
  res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // compare
  for (int i = 0; i < expected.size(); ++i)
    ASSERT_EQ(expected[i], actual_output[i]);
}
