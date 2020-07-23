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

#include "nnkit/support/tflite/AbstractBackend.h"

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <stdexcept>

namespace
{

class GenericBackend final : public nnkit::support::tflite::AbstractBackend
{
public:
  GenericBackend(const std::string &path)
  {
    ::tflite::StderrReporter error_reporter;

    _model = ::tflite::FlatBufferModel::BuildFromFile(path.c_str(), &error_reporter);

    ::tflite::ops::builtin::BuiltinOpResolver resolver;
    ::tflite::InterpreterBuilder builder(*_model, resolver);

    if (kTfLiteOk != builder(&_interp))
    {
      throw std::runtime_error{"Failed to build a tflite interpreter"};
    }

    _interp->SetNumThreads(1);
  }

public:
  ::tflite::Interpreter &interpreter(void) override { return *_interp; }

private:
  std::unique_ptr<::tflite::FlatBufferModel> _model;
  std::unique_ptr<::tflite::Interpreter> _interp;
};
} // namespace

#include <nnkit/CmdlineArguments.h>
#include <stdex/Memory.h>

extern "C" std::unique_ptr<nnkit::Backend> make_backend(const nnkit::CmdlineArguments &args)
{
  return stdex::make_unique<GenericBackend>(args.at(0));
}
