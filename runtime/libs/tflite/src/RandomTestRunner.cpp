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

#include "tflite/CopyInputInitializer.h"
#include "tflite/OutputResetter.h"
#include "tflite/RandomInputInitializer.h"
#include "tflite/RandomTestRunner.h"
#include "tflite/Diff.h"
#include "tflite/TensorLogger.h"

#include <misc/tensor/IndexIterator.h>
#include <misc/tensor/Object.h>
#include <misc/EnvVar.h>
#include <misc/fp32.h>

#include <cassert>
#include <map>
#include <functional>
#include <iostream>

namespace nnfw
{
namespace tflite
{

using namespace std::placeholders;

void RandomTestRunner::compile(const nnfw::tflite::Builder &builder)
{
  _tfl_interp = builder.build();
  _nnapi = builder.build();

  _tfl_interp->UseNNAPI(false);
  _nnapi->UseNNAPI(true);

  // Allocate Tensors
  _tfl_interp->AllocateTensors();
  _nnapi->AllocateTensors();
}

int RandomTestRunner::run(size_t running_count)
{
  assert(_tfl_interp->inputs() == _nnapi->inputs());
  assert(_tfl_interp->outputs() == _nnapi->outputs());

  nnfw::tflite::OutputResetter resetter;
  resetter.run(*(_tfl_interp.get()));

  RandomInputInitializer initializer{_randgen};
  initializer.run(*(_tfl_interp.get()));

  std::cout << "[NNAPI TEST] Run T/F Lite Interpreter without NNAPI" << std::endl;
  _tfl_interp->Invoke();

  for (size_t i = 1; i <= running_count; ++i)
  {
    resetter.run(*(_nnapi.get()));

    CopyInputInitializer copy_initializer{*(_tfl_interp.get())};
    copy_initializer.run(*(_nnapi.get()));

    std::cout << "[NNAPI TEST #" << i << "] Run T/F Lite Interpreter with NNAPI" << std::endl;

    if (_nnapi->Invoke() != kTfLiteOk)
    {
      throw std::runtime_error{"Failed to Run T/F Lite Interpreter with NNAPI"};
    }

    // Compare OFM
    std::cout << "[NNAPI TEST #" << i << "] Compare the result" << std::endl;

    const auto tolerance = _param.tolerance;

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
    TfLiteInterpMatchApp app(comparator);

    app.verbose() = _param.verbose;

    bool res = app.run(*_tfl_interp, *_nnapi);

    if (!res)
    {
      return 255;
    }

    std::cout << "[NNAPI TEST #" << i << "] PASSED" << std::endl << std::endl;

    if (_param.tensor_logging)
      nnfw::tflite::TensorLogger::get().save(_param.log_path, *_tfl_interp);
  }

  return 0;
}

RandomTestRunner RandomTestRunner::make(uint32_t seed)
{
  RandomTestParam param;

  param.verbose = nnfw::misc::EnvVar("VERBOSE").asInt(0);
  param.tolerance = nnfw::misc::EnvVar("TOLERANCE").asInt(1);
  param.tensor_logging = nnfw::misc::EnvVar("TENSOR_LOGGING").asBool(false);
  param.log_path = nnfw::misc::EnvVar("TENSOR_LOGGING").asString("tensor_log.txt");

  return RandomTestRunner{seed, param};
}

} // namespace tflite
} // namespace nnfw
