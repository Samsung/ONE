/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ReshapeLayer.h"

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

ReshapeLayer::ReshapeLayer() : cpu::ops::ReshapeLayer()
{
  // DO NOTHING
}

void ReshapeLayer::configure(const IPortableTensor *input, const IPortableTensor *shape,
                             IPortableTensor *output)
{
  cpu::ops::ReshapeLayer::configure(input, shape, output);
}

void ReshapeLayer::forward(bool training)
{
  if (training)
  {
    // TODO Implement details
  }
  else
  {
    cpu::ops::ReshapeLayer::run();
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
