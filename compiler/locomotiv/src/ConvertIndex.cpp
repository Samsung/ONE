/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <nncc/core/ADT/tensor/Index.h>

namespace locomotiv
{

nncc::core::ADT::tensor::Index as_nncc_index(const loco::TensorIndex &index)
{
  nncc::core::ADT::tensor::Index nncc_index;
  nncc_index.resize(index.rank());
  for (uint32_t axis = 0; axis < index.rank(); ++axis)
    nncc_index.at(axis) = index.at(axis);

  return nncc_index;
}

loco::TensorIndex as_loco_index(const nncc::core::ADT::tensor::Index &index)
{
  loco::TensorIndex loco_index;
  loco_index.resize(index.rank());
  for (uint32_t axis = 0; axis < index.rank(); ++axis)
    loco_index.at(axis) = index.at(axis);

  return loco_index;
}

} // namespace locomotiv
