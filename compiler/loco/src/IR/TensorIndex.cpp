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

#include "loco/IR/TensorIndex.h"

#include <stdexcept>

namespace loco
{

TensorIndex::TensorIndex() = default;

uint32_t TensorIndex::rank(void) const { return _indices.size(); }

TensorIndex &TensorIndex::resize(uint32_t size)
{
  _indices.resize(size);
  return *this;
}

uint32_t &TensorIndex::at(uint32_t axis) { return _indices.at(axis); }

uint32_t TensorIndex::at(uint32_t axis) const { return _indices.at(axis); }

} // namespace loco
