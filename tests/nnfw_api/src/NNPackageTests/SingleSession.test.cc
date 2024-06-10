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
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));
  NNFW_ENSURE_SUCCESS(nnfw_close_session(_session));

  SUCCEED();
}

TEST_F(ValidationTestSingleSession, query_info_u32)
{
  uint32_t val = 0;
  NNFW_ENSURE_SUCCESS(nnfw_query_info_u32(nullptr, NNFW_INFO_ID_VERSION, &val));

  SUCCEED();
}

TEST_F(ValidationTestSingleSession, neg_create_001)
{
  ASSERT_EQ(nnfw_create_session(nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_run_001)
{
  ASSERT_EQ(nnfw_run(nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_set_input_001)
{
  // Invalid session
  ASSERT_EQ(nnfw_set_input(nullptr, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 0),
            NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_set_input_002)
{
  char input[32];
  ASSERT_EQ(nnfw_set_input(nullptr, 0, NNFW_TYPE_TENSOR_FLOAT32, input, sizeof(input)),
            NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_set_output_001)
{
  // Invalid session
  ASSERT_EQ(nnfw_set_output(nullptr, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 0),
            NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_set_output_002)
{
  char buffer[32];
  ASSERT_EQ(nnfw_set_output(nullptr, 0, NNFW_TYPE_TENSOR_FLOAT32, buffer, sizeof(buffer)),
            NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_get_input_size)
{
  uint32_t size = 10000;
  ASSERT_EQ(nnfw_input_size(nullptr, &size), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(size, 10000);
}

TEST_F(ValidationTestSingleSession, neg_get_output_size)
{
  uint32_t size = 10000;
  ASSERT_EQ(nnfw_output_size(nullptr, &size), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(size, 10000);
}

TEST_F(ValidationTestSingleSession, neg_load_model)
{
  // Invalid state
  ASSERT_EQ(nnfw_load_model_from_file(
              nullptr, NNPackages::get().getModelAbsolutePath(NNPackages::ADD).c_str()),
            NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_prepare_001)
{
  ASSERT_EQ(nnfw_prepare(nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_query_info_u32)
{
  ASSERT_EQ(nnfw_query_info_u32(nullptr, NNFW_INFO_ID_VERSION, nullptr), NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSingleSession, neg_output_tensorinfo)
{
  nnfw_tensorinfo tensor_info;
  ASSERT_EQ(nnfw_output_tensorinfo(nullptr, 0, &tensor_info), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(nnfw_output_tensorinfo(nullptr, 0, nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSingleSession, neg_experimental_input_tensorindex_session_null)
{
  uint32_t ind = 999;
  ASSERT_EQ(nnfw_input_tensorindex(nullptr, "X_input", &ind), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(ind, 999);
}

TEST_F(ValidationTestSingleSession, neg_experimental_output_tensorindex_session_null)
{
  uint32_t ind = 999;
  ASSERT_EQ(nnfw_output_tensorindex(nullptr, "ADD_TOP", &ind), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(ind, 999);
}

TEST_F(ValidationTestSingleSession, neg_internal_set_config)
{
  ASSERT_EQ(nnfw_set_config(nullptr, "TRACING_MODE", "0"), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(nnfw_set_config(nullptr, "GRAPH_DOT_DUMP", "0"), NNFW_STATUS_UNEXPECTED_NULL);
}
