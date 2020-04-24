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
#include "ModelTestHelper.h"

/**
 * @brief Testing the following model:
 *
 * Testing the following model:
 *       #1 = const(value = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5], shape=[2, 3])
 *       #2 = placeholder (shape = [2])      <-------- this is an input
 *       #3 = reshape(#1, #2)
 *
 * @note Run this test with "cpu" backend and "linear" executor
 */
using TestDynamicTensorReshapeModelLoaded =
    ValidationTestModelLoaded<NNPackages::DYNAMIC_TENSOR_RESHAPE>;

TEST_F(TestDynamicTensorReshapeModelLoaded, reshape_to_3x2)
{
  if (!(onlyForCpuBackend(_session))) // && onlyForLinearExecutor(_session)))
  {
    // let's skip this test
    SUCCEED();
    return;
  }
  const std::vector<int> new_shape = {3, 2};
  const std::vector<float> expected = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};

  NNFW_STATUS res = NNFW_STATUS_ERROR;

  res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  res = nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_INT32, new_shape.data(),
                       sizeof(int) * new_shape.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  const int output_element_num = expected.size();
  // TODO fix output setting in dynamic way
  std::vector<float> actual_output(output_element_num);
  res = nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output.data(),
                        sizeof(float) * output_element_num);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // Do inference
  res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // output value check
  for (int i = 0; i < expected.size(); ++i)
    ASSERT_EQ(expected[i], actual_output[i]);
}

/**
 * @brief Negative test. running Reshape with non-compatible new_shape input will fail.
 */
TEST_F(TestDynamicTensorReshapeModelLoaded, neg_reshape_to_wrong_3x3)
{
  if (!(onlyForCpuBackend(_session))) // && onlyForLinearExecutor(_session)))
  {
    // let's skip this test
    SUCCEED();
    return;
  }

  NNFW_STATUS res = NNFW_STATUS_ERROR;

  res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  const std::vector<int> wrong_shape = {3, 3}; // wrong shape input
  const int output_element_num = 9;

  res = nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_INT32, wrong_shape.data(),
                       sizeof(int) * wrong_shape.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // TODO fix output setting in dynamic way
  std::vector<float> actual_output(output_element_num);
  res = nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output.data(),
                        sizeof(float) * output_element_num);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // Do inference
  res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_ERROR); // run should fail
}

TEST_F(TestDynamicTensorReshapeModelLoaded, reshape_to_3x2_multiple_executions)
{
  if (!(onlyForCpuBackend(_session))) // && onlyForLinearExecutor(_session)))
  {
    // let's skip this test
    SUCCEED();
    return;
  }

  NNFW_STATUS res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  std::vector<int> new_shape;
  std::vector<float> expected = {-1.5, -1.0, -0.5, 0.5, 1.0, 1.5};

  auto execute = [](decltype(_session) session, const decltype(new_shape) &new_shape,
                    const decltype(expected) &expected) {
    NNFW_STATUS res = NNFW_STATUS_ERROR;

    res = nnfw_set_input(session, 0, NNFW_TYPE_TENSOR_INT32, new_shape.data(),
                         sizeof(int) * new_shape.size());
    ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

    int output_element_num = expected.size();

    // TODO fix output setting in dynamic way
    std::vector<float> actual_output(output_element_num);
    res = nnfw_set_output(session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output.data(),
                          sizeof(float) * output_element_num);
    ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

    // Do inference
    res = nnfw_run(session);
    ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

    // output value check
    for (int i = 0; i < expected.size(); ++i)
      ASSERT_EQ(expected[i], actual_output[i]);
  };

  // let's call multiple times
  new_shape = {3, 2};
  execute(_session, new_shape, expected);

  new_shape = {1, 6};
  execute(_session, new_shape, expected);

  new_shape = {6, 1};
  execute(_session, new_shape, expected);
}

/**
 * @brief Testing the following model:
 *
 *        #0 = placeholder([None, None])
 *        #1 = placeholder([2, 3])
 *        #2 = concat (#0, #1, axis=0)
 *
 *        Calling sequence:
 *        - nnfw_prepare()
 *        - nnfw_apply_tensor_info(#0, [1, 3])
 *        - nnfw_set_input()
 *        - nnfw_run()
 *
 * @note Run this test with "cpu" backend and "linear" executor
 */
using TestInputUnknownDimInputConcatModelLoaded =
    ValidationTestModelLoaded<NNPackages::UNKNOWN_DIM_INPUT_CONCAT>;

TEST_F(TestInputUnknownDimInputConcatModelLoaded, concat_input1_to_2x3)
{
  const std::vector<float> input1 = {1, 2, 3};          // of shape [1, 3]
  const std::vector<float> input2 = {4, 5, 6, 7, 8, 9}; // of shape [2, 3]

  const int output_element_num = 9;
  const std::vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  NNFW_STATUS res = NNFW_STATUS_ERROR;

  if (!(onlyForCpuBackend(_session))) // && onlyForLinearExecutor(_session)))
  {
    // let's skip this test
    SUCCEED();
    return;
  }

  res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // input reshaping to [1, 3]
  nnfw_tensorinfo ti;
  {
    ti.dtype = NNFW_TYPE_TENSOR_FLOAT32;
    ti.rank = 2;
    ti.dims[0] = 1;
    ti.dims[1] = 3;
  }
  res = nnfw_apply_tensorinfo(_session, 0, ti);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  res = nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input1.data(),
                       sizeof(float) * input1.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  res = nnfw_set_input(_session, 1, NNFW_TYPE_TENSOR_FLOAT32, input2.data(),
                       sizeof(float) * input2.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // TODO fix output setting in dynamic way
  std::vector<float> actual_output(output_element_num);
  res = nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output.data(),
                        sizeof(float) * output_element_num);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // Do inference
  res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // output value check
  for (int i = 0; i < expected.size(); ++i)
    ASSERT_EQ(expected[i], actual_output[i]);
}

/**
 * @brief Negative Test: Testing the following model:
 *
 *        #0 = placeholder([None, None])
 *        #1 = placeholder([2, 3])
 *        #2 = concat (#0, #1, axis=0)
 *
 *        Calling sequence:
 *        - nnfw_prepare()
 *        - nnfw_apply_tensor_info(#0, [2, 2]) ---> input shape is not matched for concat to work
 *        - nnfw_set_input()
 *        - nnfw_run()
 *
 * @note Run this test with "cpu" backend and "linear" executor
 */
using TestInputUnknownDimInputConcatModelLoaded =
    ValidationTestModelLoaded<NNPackages::UNKNOWN_DIM_INPUT_CONCAT>;

TEST_F(TestInputUnknownDimInputConcatModelLoaded, neg_concat_input1_to_wrong_2x2)
{
  const std::vector<float> input1 = {1, 2, 3, 4};       // of shape [2, 2]
  const std::vector<float> input2 = {4, 5, 6, 7, 8, 9}; // of shape [2, 3]

  const int output_element_num = 9;

  NNFW_STATUS res = NNFW_STATUS_ERROR;

  if (!(onlyForCpuBackend(_session))) // && onlyForLinearExecutor(_session)))
  {
    // let's skip this test
    SUCCEED();
    return;
  }

  res = nnfw_prepare(_session);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // input reshaping to [2, 2]
  nnfw_tensorinfo ti;
  {
    ti.dtype = NNFW_TYPE_TENSOR_FLOAT32;
    ti.rank = 2;
    ti.dims[0] = 2;
    ti.dims[1] = 2;
  }
  res = nnfw_apply_tensorinfo(_session, 0, ti);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  res = nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input1.data(),
                       sizeof(float) * input1.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  res = nnfw_set_input(_session, 1, NNFW_TYPE_TENSOR_FLOAT32, input2.data(),
                       sizeof(float) * input2.size());
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // TODO fix output setting in dynamic way
  std::vector<float> actual_output(output_element_num);
  res = nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output.data(),
                        sizeof(float) * output_element_num);
  ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // Do inference
  res = nnfw_run(_session);
  ASSERT_EQ(res, NNFW_STATUS_ERROR);
}

/**
 * @brief Testing the following model:
 *
 *  placeholder (shape = [None,None])    if input is of shape [1, 3]
 *    |     |
 *    |  expand_dims                        shape will be [1, 3, 1]
 *    |    |
 *    |  shape                              shape will be [1, 3, 1]
 *    |   |
 *   reshape                                shape will be [1, 3, 1]
 *      |
 *   squeeze                                shape will be [3]
 *
 *        Calling sequence:
 *        - nnfw_prepare()
 *        - nnfw_apply_tensor_info(#0, [1, 3])
 *        - nnfw_set_input()
 *        - nnfw_run()
 *
 * @note Run this test with "cpu" backend and "linear" executor
 */
using TestDynamicTensorSmallNet00ModelLoaded =
    ValidationTestModelLoaded<NNPackages::DYNAMIC_TENSOR_SMALL_NET_00>;

TEST_F(TestDynamicTensorSmallNet00ModelLoaded, small_net_00)
{

  // const std::vector<float> input1 = {1, 2, 3}; // of shape [1, 3]

  // const int output_element_num = 3;
  // const std::vector<float> expected = {1, 2, 3};

  // NNFW_STATUS res = NNFW_STATUS_ERROR;

  // if (!(onlyForCpuBackend(_session) && onlyForLinearExecutor(_session)))
  // {
  //   // let's skip this test
  //   SUCCEED();
  //   return;
  // }

  // res = nnfw_prepare(_session);
  // ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // // input reshaping to [1, 3]
  // nnfw_tensorinfo ti;
  // {
  //   ti.dtype = NNFW_TYPE_TENSOR_FLOAT32;
  //   ti.rank = 2;
  //   ti.dims[0] = 1;
  //   ti.dims[1] = 3;
  // }
  // res = nnfw_apply_tensorinfo(_session, 0, ti);
  // ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // res = nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input1.data(),
  //                      sizeof(float) * input1.size());
  // ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // // TODO fix output setting in dynamic way
  // std::vector<float> actual_output(output_element_num);
  // res = nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output.data(),
  //                       sizeof(float) * output_element_num);
  // ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // // Do inference
  // res = nnfw_run(_session);
  // ASSERT_EQ(res, NNFW_STATUS_NO_ERROR);

  // // output value check
  // for (int i = 0; i < expected.size(); ++i)
  //   ASSERT_EQ(expected[i], actual_output[i]);
}
