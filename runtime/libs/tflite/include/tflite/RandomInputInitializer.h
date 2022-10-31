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

#ifndef __NNFW_TFLITE_RANDOM_INPUT_INITIALIZER_H__
#define __NNFW_TFLITE_RANDOM_INPUT_INITIALIZER_H__

#include <misc/RandomGenerator.h>

#include <tensorflow/lite/c/c_api.h>

namespace nnfw
{
namespace tflite
{

class RandomInputInitializer
{
public:
  RandomInputInitializer(misc::RandomGenerator &randgen) : _randgen{randgen}
  {
    // DO NOTHING
  }

  void run(TfLiteInterpreter &interp);

private:
  nnfw::misc::RandomGenerator &_randgen;
};

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_RANDOM_INPUT_INITIALIZER_H__
