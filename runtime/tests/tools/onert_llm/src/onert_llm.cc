/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "allocation.h"
#include "args.h"
#include "nnfw.h"
#include "nnfw_util.h"
#include "nnfw_internal.h"
#include "nnfw_experimental.h"
#include "rawformatter.h"

#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(const int argc, char **argv)
{
  using namespace onert_llm;

  try
  {
    Args args(argc, argv);

    nnfw_session *session = nullptr;
    NNPR_ENSURE_STATUS(nnfw_create_session(&session));
    NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, args.getPackageFilename().c_str()));

    // Configurations from environment variable
    const char *available_backends = std::getenv("BACKENDS");
    if (available_backends)
      NNPR_ENSURE_STATUS(nnfw_set_available_backends(session, available_backends));
    const char *log_enable = std::getenv("ENABLE_LOG");
    if (log_enable)
      NNPR_ENSURE_STATUS(nnfw_set_config(session, "ENABLE_LOG", log_enable));
    const char *num_threads = std::getenv("NUM_THREADS");
    if (num_threads)
      NNPR_ENSURE_STATUS(nnfw_set_config(session, "NUM_THREADS", num_threads));

    NNPR_ENSURE_STATUS(nnfw_prepare(session));

    uint32_t num_inputs;
    NNPR_ENSURE_STATUS(nnfw_input_size(session, &num_inputs));
    std::vector<Allocation> inputs(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));

      if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
      {
        std::cerr << "E: not supported input type" << std::endl;
        exit(-1);
      }

      auto input_size_in_bytes = bufsize_for(&ti);
      inputs[i].alloc(input_size_in_bytes, ti.dtype);

      NNPR_ENSURE_STATUS(
        nnfw_set_input(session, i, ti.dtype, inputs[i].data(), input_size_in_bytes));
    }

    uint32_t num_outputs;
    NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_outputs));
    std::vector<Allocation> outputs(num_outputs);
    for (uint32_t i = 0; i < num_outputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));

      if (ti.dtype < NNFW_TYPE_TENSOR_FLOAT32 || ti.dtype > NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED)
      {
        std::cerr << "E: not supported output type" << std::endl;
        exit(-1);
      }
      uint64_t output_size_in_bytes = bufsize_for(&ti);
      outputs[i].alloc(output_size_in_bytes, ti.dtype);
      NNPR_ENSURE_STATUS(
        nnfw_set_output(session, i, ti.dtype, outputs[i].data(), output_size_in_bytes));
    }

    // Set input data
    if (!args.getLoadRawFilename().empty())
      RawFormatter().loadInputs(args.getLoadRawFilename(), inputs);

    if (const char *trace_enable = std::getenv("TRACING_MODE");
        trace_enable != nullptr && std::string(trace_enable) == "1")
      NNPR_ENSURE_STATUS(nnfw_set_execute_config(session, NNFW_RUN_CONFIG_TRACE, nullptr));

    NNPR_ENSURE_STATUS(nnfw_run(session));

    if (!args.getDumpRawFilename().empty())
      RawFormatter().dumpOutputs(args.getDumpRawFilename(), outputs);

    NNPR_ENSURE_STATUS(nnfw_close_session(session));
  }
  catch (std::runtime_error &e)
  {
    std::cerr << "E: Fail to run by runtime error: " << e.what() << std::endl;
    exit(-1);
  }
}
