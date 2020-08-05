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
#include "nnfw_experimental.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#define NNPR_ENSURE_STATUS(a)        \
  do                                 \
  {                                  \
    if ((a) != NNFW_STATUS_NO_ERROR) \
    {                                \
      exit(-1);                      \
    }                                \
  } while (0)

extern "C" void FillFromEval(nnfw_custom_kernel_params *params, char *userdata,
                             size_t userdata_size);

const nnfw_custom_eval custom_func_ptr_list[] = {FillFromEval};
const char *custom_func_name_list[] = {"FillFrom"};
int custom_func_list_size = 1;

void register_custom_operations(nnfw_session *session)
{
  for (int i = 0; i < custom_func_list_size; ++i)
  {
    auto name = custom_func_name_list[i];
    custom_kernel_registration_info info;
    info.eval_function = custom_func_ptr_list[i];
    NNPR_ENSURE_STATUS(nnfw_register_custom_op_info(session, name, &info));
  }
}

uint64_t NowMicros()
{
  auto time_point = std::chrono::high_resolution_clock::now();
  auto since_epoch = time_point.time_since_epoch();
  // default precision of high resolution clock is 10e-9 (nanoseconds)
  return std::chrono::duration_cast<std::chrono::microseconds>(since_epoch).count();
}

uint64_t num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti->rank; ++i)
  {
    assert(ti->dims[i] >= 0);
    n *= ti->dims[i];
  }
  return n;
};

// TODO replace with data import
// Valid only for model FillFrom.tflite
// FillFrom(idx=3, val=1.1)
static const float in_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const float ref_data[10] = {1, 2, 3, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1};

std::vector<float> genData(uint64_t size)
{
  assert(size == sizeof(in_data) / sizeof(in_data[0]));
  std::cout << "Warning: runner uses hardcoded data form in_data" << std::endl;
  std::vector<float> vec(size);
  for (uint64_t i = 0; i < size; i++)
    vec[i] = in_data[i];
  return vec;
}

template <typename InIter1, typename InIter2>
static auto findMaxDifference(InIter1 first1, InIter1 last1, InIter2 first2)
    -> decltype(*first1 - *first2)
{
  auto max_difference = std::abs(*first1 - *first2);
  for (; first1 != last1; ++first1, ++first2)
  {
    auto diff = std::abs(*first1 - *first2);
    if (diff > max_difference)
    {
      max_difference = diff;
    }
  }
  return max_difference;
}

std::string dirFilename(const std::string &str)
{
  std::size_t found = str.find_last_of("/\\");
  // Finished with '/' or '\' to merge
  return str.substr(0, found + 1);
}

int main(const int argc, char **argv)
{
  std::string dir = dirFilename(argv[0]);
  std::string model_path = dir + "nnpkgs/FillFrom";

  if (argc == 1)
  {
    std::cout << "[WARNING] Use default package path\n";
  }
  else if (argc == 2)
  {
    model_path = argv[1];
  }
  else
  {
    std::cerr << "[ERROR] Invalid argument\n";
    return 1;
  }
  nnfw_session *session = nullptr;
  NNPR_ENSURE_STATUS(nnfw_create_session(&session));

  register_custom_operations(session);

  NNPR_ENSURE_STATUS(nnfw_load_model_from_file(session, model_path.c_str()));

  uint32_t num_inputs;
  NNPR_ENSURE_STATUS(nnfw_input_size(session, &num_inputs));

  // verify input and output

  if (num_inputs == 0)
  {
    std::cerr << "[ ERROR ] "
              << "No inputs in model => execution is not possible" << std::endl;
    exit(1);
  }

  auto verifyInputTypes = [session]() {
    uint32_t sz;
    NNPR_ENSURE_STATUS(nnfw_input_size(session, &sz));
    for (uint32_t i = 0; i < sz; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));
      if (ti.dtype != NNFW_TYPE_TENSOR_FLOAT32)
      {
        std::cerr << "Only float 32bit is supported." << std::endl;
        exit(-1);
      }
    }
  };

  auto verifyOutputTypes = [session]() {
    uint32_t sz;
    NNPR_ENSURE_STATUS(nnfw_output_size(session, &sz));

    for (uint32_t i = 0; i < sz; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));
      if (ti.dtype != NNFW_TYPE_TENSOR_FLOAT32)
      {
        std::cerr << "Only float 32bit is supported." << std::endl;
        exit(-1);
      }
    }
  };

  verifyInputTypes();
  verifyOutputTypes();

  // prepare execution

  uint64_t prepare_ms = NowMicros();
  NNPR_ENSURE_STATUS(nnfw_prepare(session));
  prepare_ms = NowMicros() - prepare_ms;

  // prepare input
  std::vector<std::vector<float>> inputs(num_inputs);

  auto generateInputs = [session, num_inputs, &inputs]() {
    // generate random data
    const int seed = 1;
    for (uint32_t i = 0; i < num_inputs; ++i)
    {
      nnfw_tensorinfo ti;
      NNPR_ENSURE_STATUS(nnfw_input_tensorinfo(session, i, &ti));
      auto input_num_elements = num_elems(&ti);
      inputs[i] = genData(input_num_elements);
      NNPR_ENSURE_STATUS(nnfw_set_input(session, i, NNFW_TYPE_TENSOR_FLOAT32, inputs[i].data(),
                                        sizeof(float) * input_num_elements));
      NNPR_ENSURE_STATUS(nnfw_set_input_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
    }
  };

  generateInputs();

  // prepare output
  uint32_t num_outputs = 0;
  NNPR_ENSURE_STATUS(nnfw_output_size(session, &num_outputs));
  std::vector<std::vector<float>> outputs(num_outputs);

  for (uint32_t i = 0; i < num_outputs; i++)
  {
    nnfw_tensorinfo ti;
    NNPR_ENSURE_STATUS(nnfw_output_tensorinfo(session, i, &ti));
    auto output_num_elements = num_elems(&ti);
    outputs[i].resize(output_num_elements);
    NNPR_ENSURE_STATUS(nnfw_set_output(session, i, NNFW_TYPE_TENSOR_FLOAT32, outputs[i].data(),
                                       sizeof(float) * output_num_elements));
    NNPR_ENSURE_STATUS(nnfw_set_output_layout(session, i, NNFW_LAYOUT_CHANNELS_LAST));
  }

  uint64_t run_ms = NowMicros();
  NNPR_ENSURE_STATUS(nnfw_run(session));
  run_ms = NowMicros() - run_ms;

  const float tolerance = 0.01f;
  auto max_difference =
      findMaxDifference(outputs[0].begin(), outputs[0].end(), std::begin(ref_data));

  int exit_code = 0;
  if (max_difference > tolerance)
  {
    std::cout << "Max difference is more than tolerance" << std::endl;
    std::cout << "Max difference is " << max_difference << std::endl;
    exit_code = 1;
  }

  std::cout << "nnfw_prepare takes " << prepare_ms / 1e3 << " sec" << std::endl;
  std::cout << "nnfw_run     takes " << run_ms / 1e3 << " sec" << std::endl;

  NNPR_ENSURE_STATUS(nnfw_close_session(session));

  return exit_code;
}
