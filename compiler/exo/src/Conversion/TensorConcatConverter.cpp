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

#include "TensorConcatConverter.h"

#include "GraphBlock.h"
#include "Check.h"

#include "Dialect/IR/TFLNodes.h"

#include <loco/Service/ShapeInference.h>

namespace exo
{
/**
 * @brief Converts loco::TensorConcat to locoex::TFLConcatenate
 *
 * Before:
 *   input:0 ----- loco::TensorConcat ------- C
 *   input:1 ----/
 *
 * After:
 *   input:0 ----- locoex::TFLConcatenate --- C
 *   input:1 ----/
 *
 *   input:0 ----- loco::TensorConcat ---
 *   input:1 ----/
 *
 */
bool TensorConcatConverter::convert(loco::TensorConcat *origin)
{
  assert(loco::shape_get(origin).domain() == loco::Domain::Tensor);

  if (!loco::shape_known(origin))
  {
    return false;
  }

  auto tfl_concat = origin->graph()->nodes()->create<locoex::TFLConcatenation>(2);
  tfl_concat->values(0, origin->lhs());
  tfl_concat->values(1, origin->rhs());
  tfl_concat->axis(origin->axis());
  tfl_concat->fusedActivationFunction(locoex::FusedActFunc::NONE);

  loco::replace(origin).with(tfl_concat);

  origin->lhs(nullptr);
  origin->rhs(nullptr);

  return true;
}

} // namespace exo
