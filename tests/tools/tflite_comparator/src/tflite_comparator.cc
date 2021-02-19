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
#include "InputInitializer.h"
#include "IOManager.h"
#include "MatchApp.h"

#include <nnfw_experimental.h>
#include <nnfw_internal.h>

#include <misc/EnvVar.h>
#include <misc/fp32.h>
#include <misc/RandomGenerator.h>
#include <misc/tensor/Comparator.h>

#include <tflite/Assert.h>
#include <tflite/InterpreterSession.h>
#include <tflite/ext/kernels/register.h>

#include <iostream>
#include <fstream>
#include <memory>

const int RUN_FAILED = 1;

using namespace tflite;
using namespace nnfw::tflite;

const int FILE_ERROR = 2;

#define NNFW_ASSERT_FAIL(expr, msg)   \
  if ((expr) != NNFW_STATUS_NO_ERROR) \
  {                                   \
    std::cerr << msg << std::endl;    \
    exit(-1);                         \
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

  //////////////////////////////////
  // Model Loading
  //////////////////////////////////
  nnfw_session *onert_session = nullptr;
  NNFW_ASSERT_FAIL(nnfw_create_session(&onert_session), "[ ERROR ] Failure during model load");
  if (onert_session == nullptr)
  {
    std::cerr << "[ ERROR ] Failure to open session" << std::endl;
    exit(-1);
  }

  NNFW_ASSERT_FAIL(nnfw_load_model_from_modelfile(onert_session, tflite_file.c_str()),
                   "[ ERROR ] Failure during model load");

  std::cout << "[Execution] Model is deserialized!" << std::endl;

  //////////////////////////////////
  // Compile
  //////////////////////////////////
  nnfw_prepare(onert_session);

  std::cout << "[Execution] Model compiled!" << std::endl;

  //////////////////////////////////
  // Set input / output
  //////////////////////////////////
  nnfw::onert_cmp::IOManager manager{onert_session};
  manager.prepareIOBuffers();

  uint32_t num_inputs = manager.inputs();
  uint32_t num_outputs = manager.outputs();

  bool generate_data = data_files.empty();
  if (generate_data)
  {
    const int seed = 1; /* TODO Add an option for seed value */
    nnfw::misc::RandomGenerator randgen{seed, 0.0f, 2.0f};
    nnfw::onert_cmp::RandomInputInitializer initializer{randgen};

    initializer.run(manager);
  }
  else
  {
    nnfw::onert_cmp::FileInputInitializer initialier{data_files};
    initialier.run(manager);
  }

  std::cout << "[Execution] Input data is defined!" << std::endl;

  //////////////////////////////////
  // Execute
  //////////////////////////////////
  NNFW_ASSERT_FAIL(nnfw_run(onert_session), "[Execution] Can't execute");

  std::cout << "[Execution] Done!" << std::endl;

  // Compare with tflite
  std::cout << "[Comparison] Stage start!" << std::endl;

  //////////////////////////////////
  // Read tflite model
  //////////////////////////////////
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
  interpreter->SetNumThreads(nnfw::misc::EnvVar("THREAD").asInt(1));

  auto sess = std::make_shared<nnfw::tflite::InterpreterSession>(interpreter.get());
  sess->prepare();

  //////////////////////////////////
  // Set input and run
  //////////////////////////////////
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto input_tensor = interpreter->tensor(interpreter->inputs().at(i));
    memcpy(input_tensor->data.uint8, manager.inputBase(i).data(), input_tensor->bytes);
  }
  if (!sess->run())
  {
    std::cout << "[Comparison] TFLite run failed!" << std::endl;
    assert(0 && "Run failed!");
  }
  std::cout << "[Comparison] TFLite run done!" << std::endl;

  //////////////////////////////////
  // Calculate max difference over all outputs
  //////////////////////////////////
  const auto tolerance = nnfw::misc::EnvVar("TOLERANCE").asInt(1);
  auto equals = [tolerance](float lhs, float rhs) {
    // NOTE Hybrid approach
    // TODO Allow users to set tolerance for absolute_epsilon_equal
    if (nnfw::misc::fp32::absolute_epsilon_equal(lhs, rhs))
    {
      return true;
    }

    return nnfw::misc::fp32::epsilon_equal(lhs, rhs, tolerance);
  };

  nnfw::misc::tensor::Comparator comparator(equals);
  nnfw::onert_cmp::MatchApp app(comparator);

  app.verbose() = nnfw::misc::EnvVar("VERBOSE").asInt(0);

  bool res = app.run(*interpreter, manager);

  nnfw_close_session(onert_session);

  //////////////////////////////////
  // Print results
  //////////////////////////////////
  if (!res)
  {
    std::cout << "[Comparison] outputs is not equal!" << std::endl;
    return 1;
  }

  std::cout << "[Comparison] Done!" << std::endl;

  return 0;
}
