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

// convert TensorIndex to nncc::core::ADT::tensor::Index
nncc::core::ADT::tensor::Index as_nncc_index(const loco::TensorIndex &index);

// convert nncc::core::ADT::tensor::Index to TensorIndex
loco::TensorIndex as_loco_index(const nncc::core::ADT::tensor::Index &index);

} // namespace locomotiv
