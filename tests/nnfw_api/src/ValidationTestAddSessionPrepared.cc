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

TEST_F(ValidationTestAddSessionPrepared, run_001)
{
  nnfw_tensorinfo ti_input;
  std::vector<float> input_buffer;
  ASSERT_EQ(nnfw_input_tensorinfo(_session, 0, &ti_input), NNFW_STATUS_NO_ERROR);
  uint64_t input_elements = num_elems(&ti_input);
  input_buffer.resize(input_elements);
  ASSERT_EQ(nnfw_set_input(_session, 0, ti_input.dtype, input_buffer.data(),
                           sizeof(float) * input_elements),
            NNFW_STATUS_NO_ERROR);

  nnfw_tensorinfo ti_output;
  std::vector<float> output_buffer;
  ASSERT_EQ(nnfw_output_tensorinfo(_session, 0, &ti_output), NNFW_STATUS_NO_ERROR);
  uint64_t output_elements = num_elems(&ti_output);
  output_buffer.resize(output_elements);
  ASSERT_EQ(nnfw_set_output(_session, 0, ti_output.dtype, output_buffer.data(),
                            sizeof(float) * output_elements),
            NNFW_STATUS_NO_ERROR);

  ASSERT_EQ(nnfw_run(_session), NNFW_STATUS_NO_ERROR);
}

// TODO Validation check when "nnfw_run" is called without input & output tensor setting
