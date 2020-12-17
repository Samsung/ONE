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

#include "nnsuite/conv/RandomModel.h"

#include <nncc/core/ADT/kernel/NCHWLayout.h>

#include <random>

using namespace nncc::core::ADT;

namespace nnsuite
{
namespace conv
{

RandomModel::RandomModel(int32_t seed)
  : _ifm_shape{1, 8, 8}, _ifm_name{"ifm"}, _ofm_name{"ofm"}, _ofm_shape{2, 6, 6},
    _ker_buffer{kernel::Shape{2, 1, 3, 3}, kernel::NCHWLayout{}}
{
  std::default_random_engine gen{static_cast<uint32_t>(seed)};
  std::normal_distribution<float> dist{0.0f, 1.0f};

  const uint32_t N = _ker_buffer.shape().count();
  const uint32_t C = _ker_buffer.shape().depth();
  const uint32_t H = _ker_buffer.shape().height();
  const uint32_t W = _ker_buffer.shape().width();

  for (uint32_t n = 0; n < N; ++n)
  {
    for (uint32_t ch = 0; ch < C; ++ch)
    {
      for (uint32_t row = 0; row < H; ++row)
      {
        for (uint32_t col = 0; col < W; ++col)
        {
          _ker_buffer.at(n, ch, row, col) = dist(gen);
        }
      }
    }
  }
}

} // namespace conv
} // namespace nnsuite
