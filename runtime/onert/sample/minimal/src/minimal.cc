/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nnfw.h"
#include <vector>
#include <iostream>

uint64_t num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti->rank; ++i)
  {
    n *= ti->dims[i];
  }
  return n;
}

int main(const int argc, char **argv)
{
  nnfw_session *session = nullptr;
  nnfw_create_session(&session);

  // Loading nnpackage
  nnfw_load_model_from_file(session, argv[1]);

  // Use acl_neon backend for CONV_2D and acl_cl for otherwise.
  // Note that defalut backend is acl_cl
  nnfw_set_op_backend(session, "CONV_2D", "acl_neon");

  // Compile model
  nnfw_prepare(session);

  // Prepare input. Here we just allocate dummy input arrays.
  std::vector<float> input;
  nnfw_tensorinfo ti;
  nnfw_input_tensorinfo(session, 0, &ti); // get first input's info
  uint32_t input_elements = num_elems(&ti);
  input.resize(input_elements);
  // TODO: Please add initialization for your input.
  nnfw_set_input(session, 0, ti.dtype, input.data(), sizeof(float) * input_elements);

  // Prepare output
  std::vector<float> output;
  nnfw_output_tensorinfo(session, 0, &ti); // get first output's info
  uint32_t output_elements = num_elems(&ti);
  output.resize(output_elements);
  nnfw_set_output(session, 0, ti.dtype, output.data(), sizeof(float) * output_elements);

  // Do inference
  nnfw_run(session);

  // TODO: Please print or compare the output value in your way.

  nnfw_close_session(session);

  std::cout << "nnpackage " << argv[1] << " runs successfully." << std::endl;
  return 0;
}
