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

using ValidationTestAddModelLoaded = ValidationTestModelLoaded<NNPackages::ADD>;

TEST_F(ValidationTestAddModelLoaded, prepare_001)
{
  ASSERT_EQ(nnfw_prepare(_session), NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTestAddModelLoaded, set_available_backends_001)
{
  ASSERT_EQ(nnfw_set_available_backends(_session, "cpu"), NNFW_STATUS_NO_ERROR);
}

TEST_F(ValidationTestAddModelLoaded, get_input_size)
{
  uint32_t size = 0;
  ASSERT_EQ(nnfw_input_size(_session, &size), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(size, 1);
}

TEST_F(ValidationTestAddModelLoaded, get_output_size)
{
  uint32_t size = 0;
  ASSERT_EQ(nnfw_output_size(_session, &size), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(size, 1);
}

TEST_F(ValidationTestAddModelLoaded, output_tensorinfo)
{
  nnfw_tensorinfo tensor_info;
  ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, &tensor_info), NNFW_STATUS_NO_ERROR);
  ASSERT_EQ(tensor_info.rank, 1);
  ASSERT_EQ(tensor_info.dims[0], 1);
}

TEST_F(ValidationTestAddModelLoaded, neg_run)
{
  // nnfw_prepare is not called
  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestAddModelLoaded, neg_set_input)
{
  // nnfw_prepare is not called
  ASSERT_EQ(nnfw_set_input(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 0),
            NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestAddModelLoaded, neg_set_output)
{
  // nnfw_prepare is not called
  ASSERT_EQ(nnfw_set_output(_session, 0, NNFW_TYPE_TENSOR_FLOAT32, nullptr, 0),
            NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestAddModelLoaded, neg_get_input_size)
{
  ASSERT_EQ(nnfw_input_size(_session, nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestAddModelLoaded, neg_get_output_size)
{
  ASSERT_EQ(nnfw_output_size(_session, nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestAddModelLoaded, neg_load_model)
{
  // load model twice
  ASSERT_EQ(nnfw_load_model_from_file(
                _session, NNPackages::get().getModelAbsolutePath(NNPackages::ADD).c_str()),
            NNFW_STATUS_INVALID_STATE);
}

TEST_F(ValidationTestAddModelLoaded, neg_output_tensorinfo)
{
  // tensor_info is null
  ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}
