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

#include "tflite/ext/kernels/register.h"

#include "args.h"
#include "tflite/InterpreterSession.h"
#include "tflite/Assert.h"
#include "tflite/Diff.h"
#include "misc/tensor/IndexIterator.h"

#include <nnfw_experimental.h>
#include <nnfw_internal.h>

#include <iostream>
#include <fstream>
#include <memory>

const int RUN_FAILED = 1;

using namespace tflite;
using namespace nnfw::tflite;

const int FILE_ERROR = 2;
const float DIFFERENCE_THRESHOLD = 10e-5;

#define NNFW_ASSERT_FAIL(expr, msg)   \
  if ((expr) != NNFW_STATUS_NO_ERROR) \
  {                                   \
    std::cerr << msg << std::endl;    \
    exit(-1);                         \
  }

// Read vector of floats from selected file
std::vector<float> readData(const string &path)
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
  assert(len % sizeof(float) == 0);
  size_t size = len / sizeof(float);
  std::vector<float> vec(size);
  for (size_t i = 0; i < size; ++i)
  {
    in.read(reinterpret_cast<char *>(&vec[i]), sizeof(float));
  }
  return vec;
}

std::vector<float> randomData(nnfw::misc::RandomGenerator &randgen, const uint64_t size)
{
  std::vector<float> vec(size);
  for (uint64_t i = 0; i < size; i++)
  {
    vec[i] = randgen.generate<float>();
  }
  return vec;
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
    case NNFW_TYPE_TENSOR_FLOAT32:
    case NNFW_TYPE_TENSOR_INT32:
      return 4;
    case NNFW_TYPE_TENSOR_INT64:
      return 8;
    default:
      throw std::runtime_error{"Invalid tensor type"};
  }
}

int main(const int argc, char **argv)
{
  TFLiteRun::Args args(argc, argv);

  auto tflite_file = args.getTFLiteFilename();
  auto data_files = args.getDataFilenames();

  if (tflite_file.empty())
  {
    args.print(argv);
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
  // TODO: Support another input/output types
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    nnfw_tensorinfo ti_input;
    NNFW_ASSERT_FAIL(nnfw_input_tensorinfo(onert_session, i, &ti_input),
                     "[ ERROR ] Failure during get input data info");
    assert(ti_input.dtype == NNFW_TYPE_TENSOR_FLOAT32 && "Only FLOAT32 inputs are supported");
  }
  for (uint32_t i = 0; i < num_outputs; i++)
  {
    nnfw_tensorinfo ti_output;
    NNFW_ASSERT_FAIL(nnfw_output_tensorinfo(onert_session, i, &ti_output),
                     "[ ERROR ] Failure during get output data info");
    assert(ti_output.dtype == NNFW_TYPE_TENSOR_FLOAT32 && "Only FLOAT32 outputs are supported");
  }

  std::cout << "[Execution] Model is deserialized!" << std::endl;

  // Compile
  nnfw_prepare(onert_session);

  std::cout << "[Execution] Model compiled!" << std::endl;

  // Prepare input/output data
  std::vector<std::vector<float>> inputs(num_inputs);
  std::vector<std::vector<float>> outputs(num_outputs);

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
  try
  {
    for (uint32_t i = 0; i < num_inputs; i++)
    {
      nnfw_tensorinfo ti_input;
      NNFW_ASSERT_FAIL(nnfw_input_tensorinfo(onert_session, i, &ti_input),
                       "[ ERROR ] Failure during get input data info");
      uint64_t input_elements = num_elems(&ti_input);

      if (generate_data)
      {
        inputs[i] = randomData(randgen, input_elements);
      }
      else /* read_data */
        inputs[i] = readData(data_files[i]);

      size_t input_size = input_elements * sizeOfNnfwType(ti_input.dtype);
      NNFW_ASSERT_FAIL(
          nnfw_set_input(onert_session, i, ti_input.dtype, inputs[i].data(), input_size),
          "[ ERROR ] ailure to set input tensor buffer");
    }
  }
  catch (std::exception &e)
  {
    std::cerr << "[ ERROR ] "
              << "Failure during input data generation" << std::endl;
    std::cerr << e.what() << std::endl;
    exit(-1);
  }

  std::cout << "[Execution] Input data is defined!" << std::endl;

  for (uint32_t i = 0; i < num_outputs; i++)
  {
    nnfw_tensorinfo ti_output;
    NNFW_ASSERT_FAIL(nnfw_output_tensorinfo(onert_session, i, &ti_output),
                     "[ ERROR ] Failure during get output tensor info");

    uint64_t output_elements = num_elems(&ti_output);
    outputs[i].resize(output_elements);

    size_t output_size = output_elements * sizeOfNnfwType(ti_output.dtype);
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
  StderrReporter error_reporter;
  auto model = FlatBufferModel::BuildFromFile(tflite_file.c_str(), &error_reporter);

  BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<Interpreter> interpreter;
  try
  {
    TFLITE_ENSURE(builder(&interpreter));
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    exit(FILE_ERROR);
  }
  interpreter->SetNumThreads(2);

  auto sess = std::make_shared<nnfw::tflite::InterpreterSession>(interpreter.get());
  sess->prepare();
  // Set input and run
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto input_tensor = interpreter->tensor(interpreter->inputs().at(i));
    memcpy(input_tensor->data.f, inputs[i].data(), inputs[i].size() * sizeof(float));
  }
  if (!sess->run())
  {
    std::cout << "[Comparison] TFLite run failed!" << std::endl;
    assert(0 && "Run failed!");
  }
  std::cout << "[Comparison] TFLite run done!" << std::endl;

  // Calculate max difference over all outputs
  float max_difference = 0.0f;
  for (uint32_t out_idx = 0; out_idx < num_outputs; out_idx++)
  {
    const auto &tflite_output_tensor = interpreter->tensor(interpreter->outputs().at(out_idx));
    const auto &nnfw_output_tensor = outputs[out_idx];

    if (nnfw_output_tensor.size() != tflite_output_tensor->bytes / sizeof(float))
      std::cout << "[Comparison] Different size of outputs!" << std::endl;
    // Check max difference
    float *tflite_out_ptr = tflite_output_tensor->data.f;
    for (const auto &nnfw_out : nnfw_output_tensor)
    {
      if (std::abs(nnfw_out - *tflite_out_ptr) > max_difference)
        max_difference = std::abs(nnfw_out - *tflite_out_ptr);

      tflite_out_ptr++;
    }
  }

  // Print results
  std::cout << "[Comparison] Max difference: " << max_difference << std::endl;
  int ret = 0;
  if (max_difference > DIFFERENCE_THRESHOLD)
  {
    std::cout << "[Comparison] Outputs is not equal!" << std::endl;
    ret = 1;
  }
  else
  {
    std::cout << "[Comparison] Outputs is equal!" << std::endl;
  }
  std::cout << "[Comparison] Done!" << std::endl;

  return ret;
}
