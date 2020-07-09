/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "tensorflow/lite/model.h"

#include "tflite/interp/FlatBufferBuilder.h"
#include "tflite/RandomTestRunner.h"

#include <iostream>
#include <stdexcept>

#include "args.h"

using namespace tflite;
using namespace nnfw::tflite;
using namespace nnapi_test;

int main(const int argc, char **argv)
{
  Args args(argc, argv);

  const auto filename = args.getTfliteFilename();

  StderrReporter error_reporter;

  auto model = FlatBufferModel::BuildFromFile(filename.c_str(), &error_reporter);

  if (model == nullptr)
  {
    // error_reporter must have shown the error message already
    return 1;
  }

  const nnfw::tflite::FlatBufferBuilder builder(*model);

  try
  {
    const auto seed = static_cast<uint32_t>(args.getSeed());
    auto runner = nnfw::tflite::RandomTestRunner::make(seed);
    const auto num_runs = static_cast<size_t>(args.getNumRuns());
    runner.compile(builder);
    return runner.run(num_runs);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
