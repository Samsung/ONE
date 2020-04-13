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

#include "mlapse/tfl/load.h"

#include <tflite/ext/kernels/register.h>

namespace
{

tflite::StderrReporter error_reporter;

} // namespace

namespace mlapse
{
namespace tfl
{

std::unique_ptr<tflite::FlatBufferModel> load_model(const std::string &path)
{
  return tflite::FlatBufferModel::BuildFromFile(path.c_str(), &error_reporter);
}

std::unique_ptr<tflite::Interpreter> make_interpreter(const tflite::FlatBufferModel *model)
{
  // Let's use extended resolver!
  nnfw::tflite::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<tflite::Interpreter> interpreter;

  if (builder(&interpreter) != kTfLiteOk)
  {
    return nullptr;
  }

  return std::move(interpreter);
}

} // namespace tfl
} // namespace mlapse
