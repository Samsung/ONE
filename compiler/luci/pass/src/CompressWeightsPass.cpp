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

#include "luci/Pass/CompressWeightsPass.h"
#include "helpers/HuffmanEncoder.h"
#include "helpers/NodeFiller.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Service/Nodes/CircleConst.h>

#include <cmath>
#include <cassert>

namespace
{

template <loco::DataType T> class TypeSelector;

template <> class TypeSelector<loco::DataType::U8>
{
public:
  using Type = uint8_t;
};
template <> class TypeSelector<loco::DataType::S8>
{
public:
  using Type = int8_t;
};

template <loco::DataType DT> bool compress_weights_huffman(luci::CircleConv2D *conv2d)
{
  using T = typename TypeSelector<DT>::Type;
  assert(conv2d);

  auto weights = loco::must_cast<luci::CircleConst *>(conv2d->filter());
  if (weights->compression() != luci::CompressionType::NONE)
    return false;

  luci::huffman::HuffmanEncoder<T> encoder;
  auto new_weights = luci::clone(weights);

  std::vector<T> tmp_buf(weights->size<DT>());

  for (size_t i = 0; i < weights->size<DT>(); ++i)
  {
    tmp_buf[i] = weights->at<DT>(i);
  }

  std::vector<uint8_t> encoded = encoder.encode(tmp_buf);

  new_weights->dtype(DT);
  new_weights->size<DT>(encoded.size());
  new_weights->compression(luci::CompressionType::HUFFMAN);

  for (size_t i = 0; i < new_weights->size<DT>(); ++i)
  {
    new_weights->at<DT>(i) = encoded[i];
  }
  conv2d->filter(new_weights);

  return true;
}

} // namespace

namespace luci
{

bool CompressWeightsPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto conv2d = dynamic_cast<luci::CircleConv2D *>(node);
    if (not conv2d)
      continue;

    auto filter = loco::must_cast<luci::CircleConst *>(conv2d->filter());

    if (filter->dtype() == loco::DataType::S8)
    {
      if (compress_weights_huffman<loco::DataType::S8>(conv2d))
        changed = true;
    }
    else if (filter->dtype() == loco::DataType::U8)
    {
      if (compress_weights_huffman<loco::DataType::U8>(conv2d))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
