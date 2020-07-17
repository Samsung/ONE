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

using ValidationTestAddSessionPrepared = ValidationTestSessionPrepared<NNPackages::ADD>;

TEST_F(ValidationTestAddSessionPrepared, run)
{
  SetInOutBuffers();
  _input[0] = 3.0;
  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_NO_ERROR);
  ASSERT_FLOAT_EQ(_output[0], 5.0);
}

TEST_F(ValidationTestAddSessionPrepared, run_twice)
{
  SetInOutBuffers();
  _input[0] = 4.0;
  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_NO_ERROR);
  ASSERT_FLOAT_EQ(_output[0], 6.0);

  _input[0] = 5.0f;
  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_NO_ERROR);
  ASSERT_FLOAT_EQ(_output[0], 7.0);
}

TEST_F(ValidationTestAddSessionPrepared, run_async)
{
  SetInOutBuffers();
  _input[0] = 3.0;
  ASSERT_EQ(nnfw_run_async(_session), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_await(_session), NNFW_STATUS_NO_ERROR);
  ASSERT_FLOAT_EQ(_output[0], 5.0);
}

TEST_F(ValidationTestAddSessionPrepared, set_input_001)
{
  char input[32];
  ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input, sizeof(input)),
            NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, get_input_size)
{
  uint32_t size = 0;
  ASSERT_EQ(nnfw_input_size(_session, &size), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(size, 1);
}

TEST_F(ValidationTestAddSessionPrepared, get_output_size)
{
  uint32_t size = 0;
  ASSERT_EQ(nnfw_output_size(_session, &size), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(size, 1);
}

TEST_F(ValidationTestAddSessionPrepared, output_tensorinfo)
{
  nnfw_tensorinfo tensor_info;
  ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, &tensor_info), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(tensor_info.rank, 1);
  ASSERT_EQ(tensor_info.dims[0], 1);
}

TEST_F(ValidationTestAddSessionPrepared, neg_await_without_async_run)
{
  SetInOutBuffers();
  ASSERT_EQ(nnfw_await(_session), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_await_after_sync_run)
{
  SetInOutBuffers();
  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_await(_session), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_await_twice)
{
  SetInOutBuffers();
  ASSERT_EQ(nnfw_run_async(_session), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_await(_session), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_await(_session), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_run_during_async_run)
{
  SetInOutBuffers();
  ASSERT_EQ(nnfw_run_async(_session), NNFW_STATUS_NO_ERROR);
  EXPECT_EQ(nnfw_run(_session), NNFW_STATUS_ERROR);
  ASSERT_EQ(nnfw_await(_session), NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_set_input_001)
{
  ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 1), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_set_input_002)
{
  char input[1]; // buffer size is too small
  ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input, sizeof(input)),
            NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, set_output_001)
{
  char buffer[32];
  ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, buffer, sizeof(buffer)),
            NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_set_output_001)
{
  ASSERT_EQ(nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 1), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_set_output_002)
{
  char input[1]; // buffer size is too small
  ASSERT_EQ(nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, input, sizeof(input)),
            NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_get_input_size)
{
  ASSERT_EQ(nnfw_input_size(_session, nullptr), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_get_output_size)
{
  ASSERT_EQ(nnfw_output_size(_session, nullptr), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_load_model)
{
  // Load model twice
  ASSERT_EQ(nnfw_load_model_from_file(
                _session, NNPackages::get().getModelAbsolutePath(NNPackages::ADD).c_str()),
            NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestAddSessionPrepared, neg_prepare)
{
  // Call Prepare twice
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_ERROR);
}

// TODO Validation check when "nnfw_run" is called without input & output tensor setting
