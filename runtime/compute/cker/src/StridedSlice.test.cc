/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cker/operation/StridedSlice.h>
#include <cker/Types.h>

#include <gtest/gtest.h>
#include <vector>

#include "DeathTestMacro.h"

TEST(CKer_Operation, StridedSlice5D)
{
  nnfw::cker::StridedSliceParams op_params{};
  op_params.start_indices_count = 5;
  op_params.stop_indices_count = 5;
  op_params.strides_count = 5;

  op_params.stop_indices[0] = 1;
  op_params.stop_indices[1] = 1;
  op_params.stop_indices[2] = 2;
  op_params.stop_indices[3] = 2;
  op_params.stop_indices[4] = 2;

  op_params.strides[0] = 1;
  op_params.strides[1] = 1;
  op_params.strides[2] = 1;
  op_params.strides[3] = 1;
  op_params.strides[4] = 1;

  nnfw::cker::Shape input_shape{2, 1, 2, 2, 2};
  std::vector<float> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  nnfw::cker::Shape output_shape{1, 1, 2, 2, 2};
  std::vector<float> output(output_shape.FlatSize());

  nnfw::cker::StridedSlice(op_params, input_shape, input.data(), output_shape, output.data());

  std::vector<float> expected_output = {0, 1, 2, 3, 4, 5, 6, 7};
  for (size_t i = 0; i < expected_output.size(); ++i)
    EXPECT_NEAR(output[i], expected_output[i], 1e-5f);
}

TEST(CKer_Operation, neg_StridedSliceNotSupportedDims)
{
  nnfw::cker::StridedSliceParams op_params{};
  op_params.start_indices_count = 5;
  op_params.stop_indices_count = 5;
  op_params.strides_count = 5;

  op_params.stop_indices[0] = 1;
  op_params.stop_indices[1] = 1;
  op_params.stop_indices[2] = 2;
  op_params.stop_indices[3] = 2;
  op_params.stop_indices[4] = 2;

  op_params.strides[0] = 1;
  op_params.strides[1] = 1;
  op_params.strides[2] = 1;
  op_params.strides[3] = 1;
  op_params.strides[4] = 1;

  nnfw::cker::Shape input_shape{2, 1, 2, 2, 2, 1};
  std::vector<float> input = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  nnfw::cker::Shape output_shape{1, 1, 2, 2, 2, 1};
  std::vector<float> output(output_shape.FlatSize());

  EXPECT_EXIT_BY_ABRT_DEBUG_ONLY(
    {
      nnfw::cker::StridedSlice(op_params, input_shape, input.data(), output_shape, output.data());
    },
    ".*");
}
