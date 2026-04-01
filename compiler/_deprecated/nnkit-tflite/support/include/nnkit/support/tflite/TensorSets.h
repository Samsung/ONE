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

#ifndef __NNKIT_SUPPORT_TFLITE_TENSOR_SETS_H__
#define __NNKIT_SUPPORT_TFLITE_TENSOR_SETS_H__

#include "nnkit/support/tflite/TensorSet.h"

#include <tensorflow/lite/interpreter.h>

namespace nnkit
{
namespace support
{
namespace tflite
{

class InputTensorSet final : public TensorSet
{
public:
  explicit InputTensorSet(::tflite::Interpreter &interp) : _interp(interp)
  {
    // DO NOTHING
  }

public:
  uint32_t size(void) const override { return _interp.inputs().size(); }

public:
  TfLiteTensor *at(uint32_t n) const override { return _interp.tensor(_interp.inputs().at(n)); }

private:
  ::tflite::Interpreter &_interp;
};

class OutputTensorSet final : public TensorSet
{
public:
  OutputTensorSet(::tflite::Interpreter &interp) : _interp(interp)
  {
    // DO NOTHING
  }

public:
  uint32_t size(void) const override { return _interp.outputs().size(); }

public:
  TfLiteTensor *at(uint32_t n) const override { return _interp.tensor(_interp.outputs().at(n)); }

private:
  ::tflite::Interpreter &_interp;
};

} // namespace tflite
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_TFLITE_TENSOR_SETS_H__
