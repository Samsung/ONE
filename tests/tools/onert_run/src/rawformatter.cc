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

#include "rawformatter.h"
#include "nnfw.h"
#include "nnfw_util.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

namespace onert_run
{
void RawFormatter::loadInputs(const std::string &filename, std::vector<Allocation> &inputs)
{
  uint32_t num_inputs;
  NNPR_ENSURE_STATUS(nnfw_input_size(session_, &num_inputs));

  // Support multiple inputs
  // Option 1: Get comman-separated input file list like --load:raw a,b,c
  // Option 2: Get prefix --load:raw in
  //           Internally access in.0, in.1, in.2, ... in.{N-1} where N is determined by nnfw info
  //           query api.
  //
  // Currently Option 2 is implemented.
  try
  {
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session_, i, &ti));

      // allocate memory for data
      auto bufsz = bufsize_for(&ti);
      inputs[i].alloc(bufsz);

      std::ifstream file(filename + "." + std::to_string(i), std::ios::ate | std::ios::binary);
      auto filesz = file.tellg();
      if (bufsz != filesz)
      {
        throw std::runtime_error("Input " + std::to_string(i) +
                                 " size does not match: " + std::to_string(bufsz) +
                                 " expected, but " + std::to_string(filesz) + " provided.");
      }
      file.seekg(0, std::ios::beg);
      file.read(reinterpret_cast<char *>(inputs[i].data()), filesz);
      file.close();

      NNPR_ENSURE_STATUS(nnfw_set_input(session_, i, ti.dtype, inputs[i].data(), bufsz));
      NNPR_ENSURE_STATUS(nnfw_set_input_layout(session_, i, NNFW_LAYOUT_CHANNELS_LAST));
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    std::exit(-1);
  }
};

void RawFormatter::dumpOutputs(const std::string &filename, std::vector<Allocation> &outputs)
{
  uint32_t num_outputs;
  NNPR_ENSURE_STATUS(nnfw_output_size(session_, &num_outputs));
  try
  {
    for (uint32_t i = 0; i < num_outputs; i++)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session_, i, &ti));
      auto bufsz = bufsize_for(&ti);

      std::ofstream file(filename + "." + std::to_string(i), std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<const char *>(outputs[i].data()), bufsz);
      file.close();
      std::cerr << filename + "." + std::to_string(i) + " is generated.\n";
    }
  }
  catch (const std::runtime_error &e)
  {
    std::cerr << "Error during dumpOutputs on onert_run : " << e.what() << std::endl;
    std::exit(-1);
  }
}

void RawFormatter::dumpInputs(const std::string &filename, std::vector<Allocation> &inputs)
{
  uint32_t num_inputs;
  NNPR_ENSURE_STATUS(nnfw_input_size(session_, &num_inputs));
  try
  {
    for (uint32_t i = 0; i < num_inputs; i++)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session_, i, &ti));
      auto bufsz = bufsize_for(&ti);

      std::ofstream file(filename + "." + std::to_string(i), std::ios::out | std::ios::binary);
      file.write(reinterpret_cast<const char *>(inputs[i].data()), bufsz);
      file.close();
      std::cerr << filename + "." + std::to_string(i) + " is generated.\n";
    }
  }
  catch (const std::runtime_error &e)
  {
    std::cerr << "Error during dumpRandomInputs on onert_run : " << e.what() << std::endl;
    std::exit(-1);
  }
}
} // end of namespace onert_run
