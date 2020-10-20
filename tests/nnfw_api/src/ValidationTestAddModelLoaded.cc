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
  NNFW_ENSURE_SUCCESS(nnfw_prepare(_session));

  SUCCEED();
}

TEST_F(ValidationTestAddModelLoaded, set_available_backends_001)
{
  NNFW_ENSURE_SUCCESS(nnfw_set_available_backends(_session, "cpu"));

  SUCCEED();
}

TEST_F(ValidationTestAddModelLoaded, get_input_size)
{
  uint32_t size = 0;
  NNFW_ENSURE_SUCCESS(nnfw_input_size(_session, &size));
  ASSERT_EQ(size, 1);
}

TEST_F(ValidationTestAddModelLoaded, get_output_size)
{
  uint32_t size = 0;
  NNFW_ENSURE_SUCCESS(nnfw_output_size(_session, &size));
  ASSERT_EQ(size, 1);
}

TEST_F(ValidationTestAddModelLoaded, output_tensorinfo)
{
  nnfw_tensorinfo tensor_info;
  NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(_session, 0, &tensor_info));
  ASSERT_EQ(tensor_info.rank, 1);
  ASSERT_EQ(tensor_info.dims[0], 1);
}

TEST_F(ValidationTestAddModelLoaded, input_output_tensorindex)
{
  uint32_t in_ind = 100;
  NNFW_ENSURE_SUCCESS(nnfw_input_tensorindex(_session, "X_input", &in_ind));
  ASSERT_EQ(in_ind, 0);

  uint32_t out_ind = 100;
  NNFW_ENSURE_SUCCESS(nnfw_output_tensorindex(_session, "ADD_TOP", &out_ind));
  ASSERT_EQ(out_ind, 0);
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

TEST_F(ValidationTestAddModelLoaded, neg_input_output_tensorindex)
{
  uint32_t in_ind = 100;
  ASSERT_EQ(nnfw_input_tensorindex(_session, "ADD_TOP", &in_ind), NNFW_STATUS_ERROR);
  ASSERT_EQ(in_ind, 100);
  ASSERT_EQ(nnfw_input_tensorindex(_session, "y_var", &in_ind), NNFW_STATUS_ERROR);
  ASSERT_EQ(in_ind, 100);

  uint32_t out_ind = 100;
  ASSERT_EQ(nnfw_output_tensorindex(_session, "X_input", &out_ind), NNFW_STATUS_ERROR);
  ASSERT_EQ(out_ind, 100);
}

TEST_F(ValidationTestAddModelLoaded, experimental_input_tensorindex)
{
  uint32_t ind = 999;
  NNFW_ENSURE_SUCCESS(nnfw_input_tensorindex(_session, "X_input", &ind));
  ASSERT_EQ(ind, 0);
}

TEST_F(ValidationTestAddModelLoaded, neg_experimental_input_tensorindex_name_null)
{
  uint32_t ind = 999;
  ASSERT_EQ(nnfw_input_tensorindex(_session, nullptr, &ind), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(ind, 999);
}

TEST_F(ValidationTestAddModelLoaded, neg_experimental_input_tensorindex_index_null)
{
  ASSERT_EQ(nnfw_input_tensorindex(_session, "X_input", nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestAddModelLoaded, neg_experimental_input_name_too_long)
{
  std::string long_name(1024, 'x'); // Too long
  uint32_t ind = 999;
  ASSERT_EQ(nnfw_output_tensorindex(_session, long_name.c_str(), &ind), NNFW_STATUS_ERROR);
  ASSERT_EQ(ind, 999);
}

TEST_F(ValidationTestAddModelLoaded, neg_experimental_input_no_such_name)
{
  uint32_t ind = 999;
  ASSERT_EQ(nnfw_output_tensorindex(_session, "NO_SUCH_TENSOR_NAME", &ind), NNFW_STATUS_ERROR);
  ASSERT_EQ(ind, 999);
}

TEST_F(ValidationTestAddModelLoaded, experimental_output_tensorindex)
{
  uint32_t ind = 999;
  NNFW_ENSURE_SUCCESS(nnfw_output_tensorindex(_session, "ADD_TOP", &ind));
  ASSERT_EQ(ind, 0);
}

TEST_F(ValidationTestAddModelLoaded, neg_experimental_output_tensorindex_name_null)
{
  uint32_t ind = 999;
  ASSERT_EQ(nnfw_output_tensorindex(_session, nullptr, &ind), NNFW_STATUS_UNEXPECTED_NULL);
  ASSERT_EQ(ind, 999);
}

TEST_F(ValidationTestAddModelLoaded, neg_experimental_output_tensorindex_index_null)
{
  ASSERT_EQ(nnfw_output_tensorindex(_session, "ADD_TOP", nullptr), NNFW_STATUS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestAddModelLoaded, neg_experimental_output_name_too_long)
{
  std::string long_name(1024, 'x'); // Too long
  uint32_t ind = 999;
  ASSERT_EQ(nnfw_output_tensorindex(_session, long_name.c_str(), &ind), NNFW_STATUS_ERROR);
  ASSERT_EQ(ind, 999);
}

TEST_F(ValidationTestAddModelLoaded, neg_experimental_output_no_such_name)
{
  uint32_t ind = 999;
  ASSERT_EQ(nnfw_output_tensorindex(_session, "NO_SUCH_TENSOR_NAME", &ind), NNFW_STATUS_ERROR);
  ASSERT_EQ(ind, 999);
}
