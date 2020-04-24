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
#include <nnfw_debug.h>

#include "fixtures.h"
#include "NNPackages.h"

class TestInputReshapingAddModelLoaded
    : public ValidationTestModelLoaded<NNPackages::INPUT_RESHAPING_ADD>
{
protected:
  void set_input_output(const std::vector<float> &input1, const std::vector<float> &input2,
                        std::vector<float> *actual_output)
  {
    ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input1.data(),
                             sizeof(float) * input1.size()),
              NNFW_STATUS_NO_ERROR);
    ASSERT_EQ(nnfw_set_input(_session, 1, NNFW_TYPE_TENSOR_FLOAT32, input2.data(),
                             sizeof(float) * input2.size()),
              NNFW_STATUS_NO_ERROR);

    ASSERT_EQ(nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output->data(),
                              sizeof(float) * actual_output->size()),
              NNFW_STATUS_NO_ERROR);
  }
};

/**
 * @brief Testing the following model:
 *       #1 = placeholder (shape = [2, 2], dtype=float)
 *       #2 = placeholder (shape = [2], dtype=float)
 *       #3 = add(#1, #2)
 *
 * @note Run this test with "cpu" backend and "linear" executor
 */
TEST_F(TestInputReshapingAddModelLoaded, reshaping_2x2_to_4x2_before_prepare)
{
  NNFW_STATUS res = NNFW_STATUS_ERROR;

  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

  // input and output values
  const std::vector<float> input1 = {0, 1, 2, 3, 4, 5, 6, 7}; // of changed shape [4, 2]
  const std::vector<float> input2 = {-10, -10};
  const std::vector<float> expected = {-10, -9, -8, -7, -6, -5, -4, -3}; // of shape [4, 2]
  std::vector<float> actual_output(expected.size());

  /*
  testing sequence and what's been done:
    1. nnfw_apply_tensorinfo : set input shape to different shape
    2. nnfw_prepare
    3. nnfw_set_input
    4. nnfw_run
  */

  // input reshaping from [2, 2] to [4, 2]
  nnfw_tensorinfo ti;
  {
    ti.dtype = NNFW_TYPE_TENSOR_FLOAT32;
    ti.rank = 2;
    ti.dims[0] = 4;
    ti.dims[1] = 2;
  }
  res = nnfw_apply_tensorinfo(_session, 0, ti);

  res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  set_input_output(input1, input2, &actual_output);

  // Do inference
  res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // compare
  for (int i = 0; i < expected.size(); ++i)
    ASSERT_EQ(expected[i], actual_output[i]);
}

TEST_F(TestInputReshapingAddModelLoaded, reshaping_2x2_to_4x2_after_prepare)
{
  NNFW_STATUS res = NNFW_STATUS_ERROR;

  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);

  // input and output values
  const std::vector<float> input1 = {0, 1, 2, 3, 4, 5, 6, 7}; // of changed shape [4, 2]
  const std::vector<float> input2 = {-10, -10};
  const std::vector<float> expected = {-10, -9, -8, -7, -6, -5, -4, -3}; // of shape [4, 2]
  std::vector<float> actual_output(expected.size());

  /*
  testing sequence and what's been done:
    1. nnfw_apply_tensorinfo : set input shape to different shape
    2. nnfw_set_input
    3. nnfw_prepare
    4. nnfw_run
  */

  // input reshaping from [2, 2] to [4, 2]
  nnfw_tensorinfo ti;
  {
    ti.dtype = NNFW_TYPE_TENSOR_FLOAT32;
    ti.rank = 2;
    ti.dims[0] = 4;
    ti.dims[1] = 2;
  }

  res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  res = nnfw_apply_tensorinfo(_session, 0, ti);

  set_input_output(input1, input2, &actual_output);

  // Do inference
  res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // compare
  for (int i = 0; i < expected.size(); ++i)
    ASSERT_EQ(expected[i], actual_output[i]);
}
