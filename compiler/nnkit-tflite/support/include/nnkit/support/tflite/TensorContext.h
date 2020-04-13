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

#ifndef __NNKIT_SUPPORT_TFLITE_TENSOR_CONTEXT_H__
#define __NNKIT_SUPPORT_TFLITE_TENSOR_CONTEXT_H__

#include "nnkit/support/tflite/TensorSet.h"

#include <nnkit/TensorContext.h>

namespace nnkit
{
namespace support
{
namespace tflite
{

class TensorContext final : public nnkit::TensorContext
{
public:
  TensorContext(TensorSet &tensors) : _tensors(tensors)
  {
    // DO NOTHING
  }

public:
  uint32_t size(void) const override { return _tensors.size(); }

public:
  std::string name(uint32_t n) const override { return _tensors.at(n)->name; }

public:
  nncc::core::ADT::tensor::Shape shape(uint32_t n) const override;

public:
  // Float (fp32) tensor support
  bool isFloatTensor(uint32_t n) const override { return _tensors.at(n)->type == kTfLiteFloat32; }

  void getMutableFloatTensor(uint32_t n, const TensorContext::TypedAccessor<float> &f) override;
  void getConstFloatTensor(uint32_t n, const TensorContext::TypedReader<float> &f) const override;

private:
  TensorSet &_tensors;
};

} // namespace tflite
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_TFLITE_TENSOR_CONTEXT_H__
