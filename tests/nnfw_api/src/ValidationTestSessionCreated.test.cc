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

TEST_F(ValidationTestSessionCreated, load_session_001)
{
  // Existing model must
  ASSERT_EQ(nnfw_load_model_from_file(
              _session, NNPackages::get().getModelAbsolutePath(NNPackages::ADD).c_str()),
            NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTestSessionCreated, close_and_create_again)
{
  NNFW_ENSURE_SUCCESS(nnfw_close_session(_session));
  NNFW_ENSURE_SUCCESS(nnfw_create_session(&_session));

  SUCCEED();
}

TEST_F(ValidationTestSessionCreated, neg_load_session_1)
{
  ASSERT_EQ(nnfw_load_model_from_file(
              _session, NNPackages::get().getModelAbsolutePath("nonexisting_directory").c_str()),
            NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSessionCreated, neg_load_session_2)
{
  ASSERT_EQ(nnfw_load_model_from_file(_session, nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestSessionCreated, neg_load_session_3)
{
  // Too long path
  const std::string long_path(1024, 'x');
  ASSERT_EQ(nnfw_load_model_from_file(
              _session, NNPackages::get().getModelAbsolutePath(long_path.c_str()).c_str()),
            NNFW_STATUS_ERROR);
}

TEST_F(ValidationTestSessionCreated, neg_load_invalid_package_1)
{
  ASSERT_EQ(
    nnfw_load_model_from_file(
      _session, NNPackages::get().getModelAbsolutePath(NNPackages::ADD_NO_MANIFEST).c_str()),
    NNFW_STATUS_ERROR);
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestSessionCreated, neg_load_invalid_package_2)
{
  ASSERT_EQ(
    nnfw_load_model_from_file(
      _session, NNPackages::get().getModelAbsolutePath(NNPackages::ADD_INVALID_MANIFEST).c_str()),
    NNFW_STATUS_ERROR);
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestSessionCreated, neg_prepare_001)
{
  // nnfw_load_model_from_file was not called
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestSessionCreated, neg_run_001)
{
  // nnfw_load_model_from_file and nnfw_prepare was not called
  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestSessionCreated, neg_set_input_001)
{
  ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 0),
            NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestSessionCreated, neg_set_output_001)
{
  ASSERT_EQ(nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 0),
            NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestSessionCreated, neg_get_input_size)
{
  uint32_t size = 10000;
  ASSERT_EQ(nnfw_input_size(_session, &size), NNFW_STATUS_INVALID_STATE);
  ASSERT_EQ(size, 10000); // Remain unchanged
}

TEST_F(ValidationTestSessionCreated, neg_get_output_size)
{
  uint32_t size = 10000;
  ASSERT_EQ(nnfw_output_size(_session, &size), NNFW_STATUS_INVALID_STATE);
  ASSERT_EQ(size, 10000); // Remain unchanged
}

TEST_F(ValidationTestSessionCreated, neg_output_tensorinfo)
{
  nnfw_tensorinfo tensor_info;
  // model is not loaded
  ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, &tensor_info), NNFW_STATUS_INVALID_STATE);
  // model is not loaded and tensor_info is null
  ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, nullptr), NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestSessionCreated, neg_internal_set_config)
{
  // All arguments are valid, but the session state is wrong
  ASSERT_EQ(nnfw_set_config(_session, "TRACE_FILEPATH", ""), NNFW_STATUS_INVALID_STATE);
  ASSERT_EQ(nnfw_set_config(_session, "GRAPH_DOT_DUMP", "0"), NNFW_STATUS_INVALID_STATE);
}
