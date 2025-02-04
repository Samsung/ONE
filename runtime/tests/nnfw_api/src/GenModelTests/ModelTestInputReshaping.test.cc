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
#include "common.h"
#include "CircleGen.h"

/**
 * @brief Testing the following model:
 *       #1 = placeholder (shape = [2, 2], dtype=float)
 *       #2 = placeholder (shape = [2], dtype=float)
 *       #3 = add(#1, #2)
 */
auto build_model_add_input_reshaping()
{
  // Model is not important
  CircleGen cgen;
  auto f32 = circle::TensorType::TensorType_FLOAT32;
  int in1 = cgen.addTensor({{2, 2}, f32}); // consider this [None, None]
  int in2 = cgen.addTensor({{2}, f32});
  int out = cgen.addTensor({{}, f32}); // scalar, meaning output shape is unspecified
  cgen.addOperatorAdd({{in1, in2}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in1, in2}, {out});
  auto cbuf = cgen.finish();
  return cbuf;
}

TEST(TestDynamicTensor, input_reshaping)
{
  nnfw_session *session = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  const auto model_buf = build_model_add_input_reshaping();
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, model_buf.buffer(), model_buf.size()));

  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));

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
  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(session, 0, &ti));

  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));

  nnfw_tensorinfo ti_input = {}; // Static inference result will be stored
  NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(session, 0, &ti_input));
  ASSERT_TRUE(tensorInfoEqual(ti, ti_input));

  nnfw_tensorinfo ti_output = {}; // Static inference result will be stored
  NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(session, 0, &ti_output));
  ASSERT_TRUE(tensorInfoEqual(ti, ti_output)); // input/output shapes are same with for this model

  NNFW_ENSURE_SUCCESS(nnfw_set_input(session, 0, NNFW_TYPE_TENSOR_FLOAT32, input1.data(),
                                     sizeof(float) * input1.size()));
  NNFW_ENSURE_SUCCESS(nnfw_set_input(session, 1, NNFW_TYPE_TENSOR_FLOAT32, input2.data(),
                                     sizeof(float) * input2.size()));

  uint64_t output_num_elements = tensorInfoNumElements(ti_output);
  ASSERT_EQ(output_num_elements, expected.size());
  std::vector<float> actual_output(output_num_elements);
  NNFW_ENSURE_SUCCESS(nnfw_set_output(session, 0, NNFW_TYPE_TENSOR_FLOAT32, actual_output.data(),
                                      sizeof(float) * actual_output.size()));

  // Do inference
  NNFW_ENSURE_SUCCESS(nnfw_run(session));

  // compare
  for (int i = 0; i < expected.size(); ++i)
    ASSERT_EQ(expected[i], actual_output[i]);
}
