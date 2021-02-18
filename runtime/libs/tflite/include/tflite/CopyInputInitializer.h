/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_TFLITE_COPY_INPUT_INITIALIZER_H__
#define __NNFW_TFLITE_COPY_INPUT_INITIALIZER_H__

#include <tensorflow/lite/interpreter.h>

namespace nnfw
{
namespace tflite
{

class CopyInputInitializer
{
public:
  CopyInputInitializer(::tflite::Interpreter &from) : _from{from}
  {
    // DO NOTHING
  }

  void run(::tflite::Interpreter &interp);

private:
  template <typename T> void setValue(::tflite::Interpreter &interp, int tensor_idx);

private:
  ::tflite::Interpreter &_from;
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_COPY_INPUT_INITIALIZER_H__
