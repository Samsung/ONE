/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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
#ifndef __LUCI_CIRCLE_SHAPE_INFERENCE_HELPER_PADS_H__
#define __LUCI_CIRCLE_SHAPE_INFERENCE_HELPER_PADS_H__

#include "Check.h"

#include <loco/IR/TensorShape.h>
#include <luci/IR/CircleNodes.h>
#include <limits>

namespace luci
{
namespace sinf
{

template <class CIRCLENODE>
loco::TensorShape use_paddings(const CIRCLENODE *node, const luci::CircleConst *paddings)
{
  const loco::DataType S32 = loco::DataType::S32;
  const loco::DataType S64 = loco::DataType::S64;

  auto input_shape = luci::shape_get(node->input()).template as<loco::TensorShape>();
  // TODO support other data type
  LUCI_ASSERT(paddings->dtype() == S32 || paddings->dtype() == S64, "Support int 32/64 for now");
  LUCI_ASSERT(paddings->rank() == 2, "paddings should be rank 2");

  int32_t n = paddings->dim(0).value();
  int32_t v = paddings->dim(1).value();

  LUCI_ASSERT(v == 2, "paddings should be [n, 2]");
  LUCI_ASSERT(n == int32_t(input_shape.rank()),
              "paddings [n, 2] should have same value of input rank");

  loco::TensorShape output_shape;

  output_shape.rank(input_shape.rank());
  for (int32_t ni = 0; ni < n; ++ni)
  {
    if (not input_shape.dim(ni).known())
    {
      output_shape.dim(ni).unset();
      continue;
    }
    int32_t idx = ni * 2;
    int value = input_shape.dim(ni).value();
    if (paddings->dtype() == S32)
    {
      value += paddings->at<S32>(idx + 0); // left
      value += paddings->at<S32>(idx + 1); // right
    }
    else
    {
      auto pl = paddings->at<S64>(idx + 0);
      auto pr = paddings->at<S64>(idx + 1);
      auto max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
      auto low = static_cast<int64_t>(std::numeric_limits<int32_t>::lowest());
      LUCI_ASSERT(pl <= max, "paddings is over 32 bit limit");
      LUCI_ASSERT(pl >= low, "paddings is over 32 bit limit");
      LUCI_ASSERT(pr <= max, "paddings is over 32 bit limit");
      LUCI_ASSERT(pr >= low, "paddings is over 32 bit limit");
      value += static_cast<int32_t>(pl); // left
      value += static_cast<int32_t>(pr); // right
    }
    output_shape.dim(ni) = value;
  }

  return output_shape;
}
} // namespace sinf
} // namespace luci

#endif
