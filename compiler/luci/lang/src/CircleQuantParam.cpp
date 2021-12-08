/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/CircleQuantParam.h"
#include "luci/IR/CircleNode.h"

#include <memory>

namespace luci
{

/**
 * @brief copy CircleQuantParam of src to dst
 */
void copy_quantparam(const luci::CircleNode *src, luci::CircleNode *dst)
{
  auto q = src->quantparam();
  if (q == nullptr)
    dst->quantparam(nullptr);
  else
  {
    auto qparam = std::make_unique<luci::CircleQuantParam>();
    qparam->scale = q->scale;
    qparam->zerop = q->zerop;
    qparam->min = q->min;
    qparam->max = q->max;
    qparam->quantized_dimension = q->quantized_dimension;

    dst->quantparam(std::move(qparam));
  }
}

// TODO remove this
void copy_QuantParam(const luci::CircleNode *src, luci::CircleNode *dst)
{
  copy_quantparam(src, dst);
}

} // namespace luci
