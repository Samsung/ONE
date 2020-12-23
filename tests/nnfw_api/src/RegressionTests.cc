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
#include "NNPackages.h"

#include <nnfw_internal.h>

#include "CircleGen.h"

TEST_F(RegressionTest, github_1535)
{
  auto package_path = NNPackages::get().getModelAbsolutePath(NNPackages::ADD);

  nnfw_session *session1 = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session1));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_file(session1, package_path.c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session1, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(session1));

  nnfw_session *session2 = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session2));
  NNFW_ENSURE_SUCCESS(nnfw_load_model_from_file(session2, package_path.c_str()));
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session2, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(session2));

  NNFW_ENSURE_SUCCESS(nnfw_close_session(session1));
  NNFW_ENSURE_SUCCESS(nnfw_close_session(session2));

  SUCCEED();
}

TEST_F(RegressionTest, neg_github_3826)
{
  // Model is not important
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});
  auto cbuf = cgen.finish();

  nnfw_session *session = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, cbuf.buffer(), cbuf.size()));
  // To test when there is no backends loaded for the session
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "unavailable_backend"));
  ASSERT_EQ(nnfw_prepare(session), NNFW_STATUS_ERROR);
  NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
}

TEST_F(RegressionTest, github_11748)
{
  // At the 1st call, input tensor is static. From the 2nd call, input tensor becomes dynamic.
  // the following model and calling sequence were what nnstreamer people used for their test case.
  CircleGen cgen;
  int lhs = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});

  std::vector<float> rhs_data{2};
  uint32_t rhs_buf = cgen.addBuffer(rhs_data);
  int rhs = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, rhs_buf});

  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs}, {out});
  auto cbuf = cgen.finish();

  nnfw_session *session = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, cbuf.buffer(), cbuf.size()));
  // To test when there is no backends loaded for the session
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));

  uint32_t input_num = -1;
  NNFW_ENSURE_SUCCESS(nnfw_input_size(session, &input_num));

  nnfw_tensorinfo t_input;
  NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(session, 0, &t_input));

  uint32_t output_num = -1;
  NNFW_ENSURE_SUCCESS(nnfw_output_size(session, &output_num));

  nnfw_tensorinfo t_output;
  NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(session, 0, &t_output));

  // when new_dim == 1, input tensor is static. From 2, input tensor becomes dynamic.
  for (int32_t new_dim = 1; new_dim <= 4; new_dim++)
  {
    nnfw_tensorinfo t_new_input;
    t_new_input.dtype = t_input.dtype;
    t_new_input.rank = 1;
    t_new_input.dims[0] = new_dim;
    NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(session, 0, &t_new_input));

    NNFW_ENSURE_SUCCESS(nnfw_input_size(session, &input_num));
    NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(session, 0, &t_input));

    ASSERT_EQ(input_num, 1);
    ASSERT_EQ(t_input.rank, t_new_input.rank);
    ASSERT_EQ(t_input.dims[0], new_dim);

    uint8_t input_buf[new_dim * sizeof(float)];
    NNFW_ENSURE_SUCCESS(
      nnfw_set_input(session, 0, t_input.dtype, &input_buf, new_dim * sizeof(float)));

    uint8_t output_buf[new_dim * sizeof(float)];
    NNFW_ENSURE_SUCCESS(
      nnfw_set_output(session, 0, t_output.dtype, &output_buf, new_dim * sizeof(float)));

    NNFW_ENSURE_SUCCESS(nnfw_run(session));

    NNFW_ENSURE_SUCCESS(nnfw_output_size(session, &output_num));
    NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(session, 0, &t_output));

    ASSERT_EQ(output_num, 1);
    ASSERT_EQ(t_output.rank, t_new_input.rank);
    ASSERT_EQ(t_output.dims[0], new_dim);

    // seems weird calling but anyway nnstreamer people case calls this again.
    // Anyways, runtime should work
    NNFW_ENSURE_SUCCESS(
      nnfw_set_input(session, 0, t_input.dtype, &input_buf, new_dim * sizeof(float)));
    NNFW_ENSURE_SUCCESS(
      nnfw_set_output(session, 0, t_output.dtype, &output_buf, new_dim * sizeof(float)));
    NNFW_ENSURE_SUCCESS(nnfw_run(session));
  }

  NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
}

TEST_F(RegressionTest, github_4585)
{
  // A single tensor which is an input and an output at the same time
  CircleGen cgen;
  int t = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.setInputsAndOutputs({t}, {t});
  auto cbuf = cgen.finish();

  nnfw_session *session = nullptr;
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&session));
  NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(session, cbuf.buffer(), cbuf.size()));
  // To test when there is no backends loaded for the session
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(session, "cpu"));
  NNFW_ENSURE_SUCCESS(nnfw_prepare(session));

  // Change input tensorinfo (Make dynamic shape inference happen)
  nnfw_tensorinfo ti_new = {NNFW_TYPE_TENSOR_FLOAT32, 2, {1, 2}};
  NNFW_ENSURE_SUCCESS(nnfw_set_input_tensorinfo(session, 0, &ti_new));

  std::vector<float> in_buf{1, 1};
  std::vector<float> out_buf{-1, -1};

  NNFW_ENSURE_SUCCESS(
    nnfw_set_input(session, 0, ti_new.dtype, in_buf.data(), in_buf.size() * sizeof(float)));
  NNFW_ENSURE_SUCCESS(
    nnfw_set_output(session, 0, ti_new.dtype, out_buf.data(), out_buf.size() * sizeof(float)));

  NNFW_ENSURE_SUCCESS(nnfw_run(session));

  ASSERT_EQ(in_buf, out_buf);

  NNFW_ENSURE_SUCCESS(nnfw_close_session(session));
}
