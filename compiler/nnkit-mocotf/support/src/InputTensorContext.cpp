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

#include "InputTensorContext.h"
#include "nnkit/TensorContext.h"

#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Overlay.h>

namespace nnkit
{
namespace support
{
namespace moco
{
namespace tf
{

void InputTensorContext::getMutableFloatTensor(uint32_t n,
                                               const nnkit::TensorContext::TypedAccessor<float> &f)
{
  auto buf = _buffers.at(n).get();
  f(*this, n, *buf);
}

void InputTensorContext::getConstFloatTensor(
  uint32_t n, const nnkit::TensorContext::TypedReader<float> &f) const
{
  auto buf = _buffers.at(n).get();
  f(*this, n, *buf);
}

} // namespace tf
} // namespace moco
} // namespace support
} // namespace nnkit
