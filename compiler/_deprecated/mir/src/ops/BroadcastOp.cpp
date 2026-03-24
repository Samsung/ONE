/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/ops/BroadcastOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/Tensor.h"

namespace mir
{
namespace ops
{

void BroadcastOp::inferOutputTypes(const Shape &target_shape)
{
  const Shape &input_shape = getInputShape(0);
  Shape output_shape = broadcastShapes(input_shape, target_shape);

  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
