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

#include "InferenceInterface.h"

using namespace tflite;
using namespace tflite::ops::builtin;

InferenceInterface::InferenceInterface(const std::string &model_file, const bool use_nnapi)
  : _interpreter(nullptr), _model(nullptr), _sess(nullptr)
{
  // Load model
  StderrReporter error_reporter;
  _model = FlatBufferModel::BuildFromFile(model_file.c_str(), &error_reporter);
  BuiltinOpResolver resolver;
  InterpreterBuilder builder(*_model, resolver);
  builder(&_interpreter);

  if (use_nnapi)
  {
    _sess = std::make_shared<nnfw::tflite::NNAPISession>(_interpreter.get());
  }
  else
  {
    _sess = std::make_shared<nnfw::tflite::InterpreterSession>(_interpreter.get());
  }

  _sess->prepare();
}

InferenceInterface::~InferenceInterface() { _sess->teardown(); }

void InferenceInterface::feed(const std::string &input_name, const std::vector<float> &data,
                              const int batch, const int height, const int width, const int channel)
{
  // Set input tensor
  for (const auto &id : _interpreter->inputs())
  {
    if (_interpreter->tensor(id)->name == input_name)
    {
      assert(_interpreter->tensor(id)->type == kTfLiteFloat32);
      float *p = _interpreter->tensor(id)->data.f;

      // TODO consider batch
      for (int y = 0; y < height; ++y)
      {
        for (int x = 0; x < width; ++x)
        {
          for (int c = 0; c < channel; ++c)
          {
            *p++ = data[y * width * channel + x * channel + c];
          }
        }
      }
    }
  }
}

void InferenceInterface::run(const std::string &output_name)
{
  // Run model
  _sess->run();
}

void InferenceInterface::fetch(const std::string &output_name, std::vector<float> &outputs)
{
  // Get output tensor
  for (const auto &id : _interpreter->outputs())
  {
    if (_interpreter->tensor(id)->name == output_name)
    {
      assert(_interpreter->tensor(id)->type == kTfLiteFloat32);
      assert(getTensorSize(output_name) == outputs.capacity());
      float *p = _interpreter->tensor(id)->data.f;

      outputs.clear();
      for (int i = 0; i < outputs.capacity(); ++i)
      {
        outputs.push_back(p[i]);
      }
    }
  }
}

int InferenceInterface::getTensorSize(const std::string &name)
{
  for (const auto &id : _interpreter->outputs())
  {
    if (_interpreter->tensor(id)->name == name)
    {
      TfLiteTensor *t = _interpreter->tensor(id);
      int v = 1;
      for (int i = 0; i < t->dims->size; ++i)
      {
        v *= t->dims->data[i];
      }
      return v;
    }
  }
  return -1;
}
