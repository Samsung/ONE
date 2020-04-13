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

#include "mir/ops/FullyConnectedOp.h"

namespace mir
{
namespace ops
{

void FullyConnectedOp::inferOutputTypes()
{
  auto &input_shape = getInputShape(0);
  auto &weights_shape = getInputShape(1);
  auto input_rank = input_shape.rank();
  auto weights_rank = weights_shape.rank();

  assert(weights_rank >= 2);
  assert(input_rank == weights_rank);
  assert(input_shape.dim(input_rank - 1) == weights_shape.dim(weights_rank - 2));
  (void)input_rank;
  for (int32_t i = 0; i < weights_rank - 2; ++i)
    assert(weights_shape.dim(i) == input_shape.dim(i));

  Shape output_shape = weights_shape;
  output_shape.dim(weights_rank - 1) = weights_shape.dim(weights_rank - 1);
  output_shape.dim(weights_rank - 2) = input_shape.dim(weights_rank - 2);

  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
