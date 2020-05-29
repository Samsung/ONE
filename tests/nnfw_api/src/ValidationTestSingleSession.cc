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

#include "NNPackages.h"
#include "fixtures.h"

TEST_F(ValidationTestSingleSession, create_001)
{
  ASSERT_EQ(nnfw_create_session(&_session), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(nnfw_close_session(_session), NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTestSingleSession, neg_create_001)
{
  ASSERT_EQ(nnfw_create_session(nullptr), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSingleSession, neg_run_001)
{
  ASSERT_EQ(nnfw_run(nullptr), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSingleSession, neg_set_input_001)
{
  // Invalid session
  ASSERT_EQ(nnfw_set_input(nullptr, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 0), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSingleSession, neg_set_input_002)
{
  char input[32];
  ASSERT_EQ(nnfw_set_input(nullptr, 0, NNFW_TYPE_TENSOR_FLOAT32, input, sizeof(input)),
            NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSingleSession, neg_set_output_001)
{
  // Invalid session
  ASSERT_EQ(nnfw_set_output(nullptr, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 0), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSingleSession, neg_set_output_002)
{
  char buffer[32];
  ASSERT_EQ(nnfw_set_output(nullptr, 0, NNFW_TYPE_TENSOR_FLOAT32, buffer, sizeof(buffer)),
            NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSingleSession, neg_get_input_size)
{
  uint32_t size = 10000;
  ASSERT_EQ(nnfw_input_size(nullptr, &size), NNFW_STATUS_ERROR);
  ASSERT_EQ(size, 10000);
}

TEST_F(ValidationTestSingleSession, neg_get_output_size)
{
  uint32_t size = 10000;
  ASSERT_EQ(nnfw_output_size(nullptr, &size), NNFW_STATUS_ERROR);
  ASSERT_EQ(size, 10000);
}

TEST_F(ValidationTestSingleSession, neg_load_model)
{
  // Invalid state
  ASSERT_EQ(nnfw_load_model_from_file(
                nullptr, NNPackages::get().getModelAbsolutePath(NNPackages::ADD).c_str()),
            NNFW_STATUS_ERROR);
}
