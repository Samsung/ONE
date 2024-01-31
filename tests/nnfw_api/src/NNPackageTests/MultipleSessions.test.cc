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

#include "fixtures.h"
#include "GenModelTests/one_op_tests/WhileTestModel.h"

TEST_F(ValidationTestTwoSessions, neg_two_sessions_create)
{
  ASSERT_EQ(nnfw_create_session(&_session1), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_create_session(nullptr), NNFW_STATUS_UNEXPECTED_NULL);

  ASSERT_EQ(nnfw_close_session(_session1), NNFW_STATUS_NO_ERROR);
}

class AveragePoolModel
{
public:
  AveragePoolModel(int N, int H, int W, int C)
  {
    CircleGen cgen;
    int in = cgen.addTensor({{N, H, W, C}, circle::TensorType::TensorType_FLOAT32});
    int out = cgen.addTensor({{N, H / 2, W / 2, C}, circle::TensorType::TensorType_FLOAT32});
    cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 2,
                                  circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({in}, {out});
    cbuf = cgen.finish();
  };

  CircleBuffer cbuf;
};

TEST_F(ValidationTestTwoSessionsCreated, two_sessions_run_simple_AaveragePool_model)
{
  constexpr int N = 64, H = 64, W = 64, C = 3;
  AveragePoolModel model(N, H, W, C);

  NNFW_ENSURE_SUCCESS(
    nnfw_load_circle_from_buffer(_session1, model.cbuf.buffer(), model.cbuf.size()));
  NNFW_ENSURE_SUCCESS(
    nnfw_load_circle_from_buffer(_session2, model.cbuf.buffer(), model.cbuf.size()));

  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session1, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session2, "cpu"));

  NNFW_ENSURE_SUCCESS(nnfw_prepare(_session1));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(_session2));

  constexpr int input_count = N * H * W * C;
  constexpr int output_count = N * H / 2 * W / 2 * C;

  std::vector<float> in_buf1(input_count); // any value
  std::vector<float> out_buf1(output_count);

  NNFW_ENSURE_SUCCESS(nnfw_set_input(_session1, 0, NNFW_TYPE_TENSOR_FLOAT32, in_buf1.data(),
                                     in_buf1.size() * sizeof(float)));
  NNFW_ENSURE_SUCCESS(nnfw_set_output(_session1, 0, NNFW_TYPE_TENSOR_FLOAT32, out_buf1.data(),
                                      out_buf1.size() * sizeof(float)));

  std::vector<float> in_buf2(input_count); // any value
  std::vector<float> out_buf2(output_count);

  NNFW_ENSURE_SUCCESS(nnfw_set_input(_session2, 0, NNFW_TYPE_TENSOR_FLOAT32, in_buf2.data(),
                                     in_buf2.size() * sizeof(float)));
  NNFW_ENSURE_SUCCESS(nnfw_set_output(_session2, 0, NNFW_TYPE_TENSOR_FLOAT32, out_buf2.data(),
                                      out_buf2.size() * sizeof(float)));

  NNFW_ENSURE_SUCCESS(nnfw_run_async(_session1));
  NNFW_ENSURE_SUCCESS(nnfw_run_async(_session2));

  NNFW_ENSURE_SUCCESS(nnfw_await(_session1));
  NNFW_ENSURE_SUCCESS(nnfw_await(_session2));

  SUCCEED();
}

TEST_F(ValidationTestTwoSessionsCreated, neg_two_sessions_model_load)
{
  constexpr int N = 64, H = 64, W = 64, C = 3;
  AveragePoolModel model(N, H, W, C);

  NNFW_ENSURE_SUCCESS(
    nnfw_load_circle_from_buffer(_session1, model.cbuf.buffer(), model.cbuf.size()));
  ASSERT_EQ(nnfw_load_circle_from_buffer(nullptr, model.cbuf.buffer(), model.cbuf.size()),
            NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestTwoSessionsCreated, two_sessions_run_simple_While_model)
{
  WhileModelLoop10 model;

  NNFW_ENSURE_SUCCESS(
    nnfw_load_circle_from_buffer(_session1, model.cbuf.buffer(), model.cbuf.size()));
  NNFW_ENSURE_SUCCESS(
    nnfw_load_circle_from_buffer(_session2, model.cbuf.buffer(), model.cbuf.size()));

  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session1, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session2, "cpu"));

  NNFW_ENSURE_SUCCESS(nnfw_prepare(_session1));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(_session2));

  std::vector<float> in_buf1(model.inputCount()); // any value
  std::vector<float> out_buf1(model.outputputCount());

  NNFW_ENSURE_SUCCESS(nnfw_set_input(_session1, 0, NNFW_TYPE_TENSOR_FLOAT32, in_buf1.data(),
                                     in_buf1.size() * model.sizeOfDType()));
  NNFW_ENSURE_SUCCESS(nnfw_set_output(_session1, 0, NNFW_TYPE_TENSOR_FLOAT32, out_buf1.data(),
                                      out_buf1.size() * model.sizeOfDType()));

  std::vector<float> in_buf2(model.inputCount()); // any value
  std::vector<float> out_buf2(model.outputputCount());

  NNFW_ENSURE_SUCCESS(nnfw_set_input(_session2, 0, NNFW_TYPE_TENSOR_FLOAT32, in_buf2.data(),
                                     in_buf2.size() * model.sizeOfDType()));
  NNFW_ENSURE_SUCCESS(nnfw_set_output(_session2, 0, NNFW_TYPE_TENSOR_FLOAT32, out_buf2.data(),
                                      out_buf2.size() * model.sizeOfDType()));

  NNFW_ENSURE_SUCCESS(nnfw_run_async(_session1));
  NNFW_ENSURE_SUCCESS(nnfw_run_async(_session2));

  NNFW_ENSURE_SUCCESS(nnfw_await(_session1));
  NNFW_ENSURE_SUCCESS(nnfw_await(_session2));

  SUCCEED();
}

// TODO Write two-session-test with large models run by threads
