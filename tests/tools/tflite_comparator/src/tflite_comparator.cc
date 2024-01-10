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

#include "args.h"

#include <nnfw_experimental.h>
#include <nnfw_internal.h>

#include <misc/EnvVar.h>
#include <misc/fp32.h>
#include <misc/RandomGenerator.h>

#include <tflite/Assert.h>
#include <tflite/InterpreterSession.h>

#include <CLI11.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cstring>

const int RUN_FAILED = 1;

using namespace nnfw::tflite;

const int FILE_ERROR = 2;

#define NNFW_ASSERT_FAIL(expr, msg)   \
  if ((expr) != NNFW_STATUS_NO_ERROR) \
  {                                   \
    std::cerr << msg << std::endl;    \
    exit(-1);                         \
  }

// Read vector of floats from selected file
void readData(const std::string &path, std::vector<uint8_t> &dest)
{
  std::ifstream in(path);
  if (!in.good())
  {
    std::cerr << "can not open data file " << path << "\n";
    exit(FILE_ERROR);
  }
  in.seekg(0, std::ifstream::end);
  size_t len = in.tellg();
  in.seekg(0, std::ifstream::beg);

  assert(dest.size() == len);
  in.read(reinterpret_cast<char *>(dest.data()), len);
}

template <typename T>
void randomData(nnfw::misc::RandomGenerator &randgen, std::vector<uint8_t> &dest)
{
  size_t elements = dest.size() / sizeof(T);
  assert(dest.size() % sizeof(T) == 0);

  std::vector<T> vec(elements);
  for (uint64_t i = 0; i < elements; i++)
  {
    vec[i] = randgen.generate<T>();
  }
  memcpy(dest.data(), vec.data(), elements * sizeof(T));
}

void randomBoolData(nnfw::misc::RandomGenerator &randgen, std::vector<uint8_t> &dest)
{
  size_t elements = dest.size();
  std::vector<uint8_t> vec(elements);
  for (uint64_t i = 0; i < elements; i++)
  {
    bool value = randgen.generate<bool>();
    dest[i] = value ? 1 : 0;
  }
}

inline uint64_t num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti->rank; ++i)
  {
    n *= ti->dims[i];
  }
  return n;
}

inline size_t sizeOfNnfwType(NNFW_TYPE type)
{
  switch (type)
  {
    case NNFW_TYPE_TENSOR_BOOL:
    case NNFW_TYPE_TENSOR_UINT8:
    case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
    case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
      return 1;
    case NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
      return 2;
    case NNFW_TYPE_TENSOR_FLOAT32:
    case NNFW_TYPE_TENSOR_INT32:
      return 4;
    case NNFW_TYPE_TENSOR_INT64:
      return 8;
    default:
      throw std::runtime_error{"Invalid tensor type"};
  }
}

template <typename T>
bool isClose(const T *ref_buf, const std::vector<uint8_t> &act_buf, uint32_t index)
{
  // TODO better way for handling quant error?
  auto tolerance = static_cast<uint64_t>(nnfw::misc::EnvVar("TOLERANCE").asInt(0));
  bool match = true;

  for (uint32_t e = 0; e < act_buf.size() / sizeof(T); e++)
  {
    T ref = ref_buf[e];
    T act = reinterpret_cast<const T *>(act_buf.data())[e];
    uint64_t diff = static_cast<uint64_t>(((ref > act) ? (ref - act) : (act - ref)));

    if (ref != act && diff > tolerance)
    {
      std::cerr << "Output #" << index << ", Element Index : " << e << ", ref: " << ref
                << ", act: " << act << " (diff: " << diff << ")" << std::endl;
      match = false;
    }
  }

  return match;
}

template <>
bool isClose<float>(const float *ref_buf, const std::vector<uint8_t> &act_buf, uint32_t index)
{
  uint32_t tolerance = nnfw::misc::EnvVar("TOLERANCE").asInt(1);
  bool match = true;

  for (uint32_t e = 0; e < act_buf.size() / sizeof(float); e++)
  {
    float ref = ref_buf[e];
    float act = reinterpret_cast<const float *>(act_buf.data())[e];
    float diff = std::fabs(ref - act);

    bool match_elem = nnfw::misc::fp32::absolute_epsilon_equal(ref, act)
                        ? true
                        : nnfw::misc::fp32::epsilon_equal(ref, act, tolerance);

    if (!match_elem)
    {
      std::cerr << "Output #" << index << ", Element Index : " << e << ", ref: " << ref
                << ", act: " << act << " (diff: " << diff << ")" << std::endl;
      match = false;
    }
  }

  return match;
}

bool exact(const uint8_t *ref_buf, const std::vector<uint8_t> &act_buf, uint32_t index)
{
  bool match = true;
  for (uint32_t e = 0; e < act_buf.size() / sizeof(uint8_t); e++)
  {
    uint8_t ref_raw = ref_buf[e];
    bool ref = (ref_raw != 0 ? true : false);
    uint8_t act_raw = reinterpret_cast<const uint8_t *>(act_buf.data())[e];
    bool act = (act_raw != 0 ? true : false);
    if (ref != act)
    {
      std::cerr << "Output #" << index << ", Element Index : " << e << ", ref: " << ref
                << ", act: " << act << std::endl;
      match = false;
    }
  }

  return match;
}

int main(const int argc, char **argv)
{
  TFLiteRun::Args args(argc, argv);

  auto tflite_file = args.getTFLiteFilename();
  auto data_files = args.getDataFilenames();

  if (tflite_file.empty())
  {
    args.print();
    return RUN_FAILED;
  }

  std::cout << "[Execution] Stage start!" << std::endl;
  // Loading
  nnfw_session *onert_session = nullptr;
  NNFW_ASSERT_FAIL(nnfw_create_session(&onert_session), "[ ERROR ] Failure during model load");
  if (onert_session == nullptr)
  {
    std::cerr << "[ ERROR ] Failure to open session" << std::endl;
    exit(-1);
  }

  NNFW_ASSERT_FAIL(nnfw_load_model_from_modelfile(onert_session, tflite_file.c_str()),
                   "[ ERROR ] Failure during model load");

  uint32_t num_inputs;
  uint32_t num_outputs;
  NNFW_ASSERT_FAIL(nnfw_input_size(onert_session, &num_inputs),
                   "[ ERROR ] Failure during get model inputs");
  NNFW_ASSERT_FAIL(nnfw_output_size(onert_session, &num_outputs),
                   "[ ERROR ] Failure during get model outputs");

  std::cout << "[Execution] Model is deserialized!" << std::endl;

  // Compile
  nnfw_prepare(onert_session);

  std::cout << "[Execution] Model compiled!" << std::endl;

  // Prepare input/output data
  std::vector<std::vector<uint8_t>> inputs(num_inputs);
  std::vector<std::vector<uint8_t>> outputs(num_outputs);

  bool generate_data = data_files.empty();
  bool read_data = data_files.size() == num_inputs;
  if (!generate_data && !read_data)
  {
    std::cerr << "[ ERROR ] "
              << "Wrong number of input files." << std::endl;
    exit(1);
  }

  const int seed = 1; /* TODO Add an option for seed value */
  nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};

  for (uint32_t i = 0; i < num_inputs; i++)
  {
    nnfw_tensorinfo ti_input;
    NNFW_ASSERT_FAIL(nnfw_input_tensorinfo(onert_session, i, &ti_input),
                     "[ ERROR ] Failure during get input data info");
    size_t input_size = num_elems(&ti_input) * sizeOfNnfwType(ti_input.dtype);

    inputs[i].resize(input_size);

    if (generate_data)
    {
      switch (ti_input.dtype)
      {
        case NNFW_TYPE_TENSOR_BOOL:
          randomBoolData(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_UINT8:
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
          randomData<uint8_t>(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
          randomData<int8_t>(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED:
          randomData<int16_t>(randgen, inputs[i]);
        case NNFW_TYPE_TENSOR_FLOAT32:
          randomData<float>(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_INT32:
          randomData<int32_t>(randgen, inputs[i]);
          break;
        case NNFW_TYPE_TENSOR_INT64:
          randomData<uint64_t>(randgen, inputs[i]);
          break;
        default:
          std::cerr << "[ ERROR ] "
                    << "Unspported input data type" << std::endl;
          exit(-1);
          break;
      }
    }
    else /* read_data */
      readData(data_files[i], inputs[i]);

    NNFW_ASSERT_FAIL(nnfw_set_input(onert_session, i, ti_input.dtype, inputs[i].data(), input_size),
                     "[ ERROR ] Failure to set input tensor buffer");
  }

  std::cout << "[Execution] Input data is defined!" << std::endl;

  for (uint32_t i = 0; i < num_outputs; i++)
  {
    nnfw_tensorinfo ti_output;
    NNFW_ASSERT_FAIL(nnfw_output_tensorinfo(onert_session, i, &ti_output),
                     "[ ERROR ] Failure during get output tensor info");

    uint64_t output_elements = num_elems(&ti_output);
    size_t output_size = output_elements * sizeOfNnfwType(ti_output.dtype);
    outputs[i].resize(output_size);

    NNFW_ASSERT_FAIL(
      nnfw_set_output(onert_session, i, ti_output.dtype, outputs[i].data(), output_size),
      "[ ERROR ] Failure to set output tensor buffer");
  }

  // Execute
  NNFW_ASSERT_FAIL(nnfw_run(onert_session), "[Execution] Can't execute");

  std::cout << "[Execution] Done!" << std::endl;

  // Compare with tflite
  std::cout << "[Comparison] Stage start!" << std::endl;
  // Read tflite model
  auto model = TfLiteModelCreateFromFile(tflite_file.c_str());
  auto options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, nnfw::misc::EnvVar("THREAD").asInt(1));
  auto interpreter = TfLiteInterpreterCreate(model, options);

  auto sess = std::make_shared<nnfw::tflite::InterpreterSession>(interpreter);
  sess->prepare();
  // Set input and run
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto input_tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
    memcpy(TfLiteTensorData(input_tensor), inputs[i].data(), inputs[i].size());
  }
  if (!sess->run())
  {
    std::cout << "[Comparison] TFLite run failed!" << std::endl;
    assert(0 && "Run failed!");
  }
  std::cout << "[Comparison] TFLite run done!" << std::endl;

  bool find_unmatched_output = false;

  for (uint32_t out_idx = 0; out_idx < num_outputs; out_idx++)
  {
    nnfw_tensorinfo ti;
    nnfw_output_tensorinfo(onert_session, out_idx, &ti);

    bool matched = true;
    // Check output tensor values
    auto output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, out_idx);
    auto ref_output = TfLiteTensorData(output_tensor);
    const auto &output = outputs[out_idx];

    switch (ti.dtype)
    {
      case NNFW_TYPE_TENSOR_BOOL:
        matched = exact(reinterpret_cast<uint8_t *>(ref_output), output, out_idx);
        break;
      case NNFW_TYPE_TENSOR_UINT8:
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
        matched = isClose<uint8_t>(reinterpret_cast<uint8_t *>(ref_output), output, out_idx);
        break;
      case NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED:
        matched = isClose<int8_t>(reinterpret_cast<int8_t *>(ref_output), output, out_idx);
        break;
      case NNFW_TYPE_TENSOR_INT32:
        matched = isClose<int32_t>(reinterpret_cast<int32_t *>(ref_output), output, out_idx);
        break;
      case NNFW_TYPE_TENSOR_FLOAT32:
        matched = isClose<float>(reinterpret_cast<float *>(ref_output), output, out_idx);
        break;
      case NNFW_TYPE_TENSOR_INT64:
        matched = isClose<int64_t>(reinterpret_cast<int64_t *>(ref_output), output, out_idx);
        break;
      default:
        throw std::runtime_error{"Invalid tensor type"};
    }

    if (!matched)
      find_unmatched_output = true;
  }

  // Print results
  int ret = 0;
  if (find_unmatched_output)
  {
    std::cout << "[Comparison] outputs is not equal!" << std::endl;
    ret = 1;
  }
  else
  {
    std::cout << "[Comparison] Outputs is equal!" << std::endl;
  }
  std::cout << "[Comparison] Done!" << std::endl;

  nnfw_close_session(onert_session);

  return ret;
}
