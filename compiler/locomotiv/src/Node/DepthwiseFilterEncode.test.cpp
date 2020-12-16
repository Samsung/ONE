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

#include "NodeExecution.h"

#include "locomotiv/NodeData.h"
#include "NodeDataImpl.h"
#include "NodeDomain.h"

#include <loco/IR/PermutingCodec.h>

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::IndexEnumerator;

TEST(NodeExecution_DepthwiseFilterEncode, f32)
{
  const uint32_t H = 2;
  const uint32_t W = 3;
  const uint32_t C = 4;
  const uint32_t M = 5;

  auto g = loco::make_graph();

  // Pull
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::FLOAT32);

  // Make and assign "MHWC" data to pull node
  auto pull_buf = make_buffer<float, LexicalLayout>(Shape{M, H, W, C});
  float f = 1;
  for (IndexEnumerator e{pull_buf.shape()}; e.valid(); e.advance())
  {
    pull_buf.at(e.current()) = f;
    f += 0.1f; // Doesn't matter what it is
  }
  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::annot_data(pull, std::move(pull_data));
  locomotiv::annot_domain(pull, loco::Domain::Tensor);

  // Encoder to correctly read input tensor as MHWC
  auto encoder = std::unique_ptr<loco::PermutingEncoder<loco::Domain::DepthwiseFilter>>(
    new loco::PermutingEncoder<loco::Domain::DepthwiseFilter>);
  encoder->perm()->axis(loco::DepthwiseFilterAxis::Multiplier) = 0;
  encoder->perm()->axis(loco::DepthwiseFilterAxis::Height) = 1;
  encoder->perm()->axis(loco::DepthwiseFilterAxis::Width) = 2;
  encoder->perm()->axis(loco::DepthwiseFilterAxis::Depth) = 3;

  // DepthwiseFilterEncode
  auto enc = g->nodes()->create<loco::DepthwiseFilterEncode>();
  enc->input(pull);
  enc->encoder(std::move(encoder));

  locomotiv::NodeExecution::get().run(enc);

  auto enc_data = locomotiv::annot_data(enc);
  ASSERT_NE(enc_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, enc_data->dtype());
  ASSERT_EQ((Shape{H, W, C, M}), *(enc_data->shape())); // locomotiv depthwise filter is HWCM
  auto enc_buf = enc_data->as_f32_bufptr();
  for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
      for (uint32_t c = 0; c < C; ++c)
        for (uint32_t m = 0; m < M; ++m)
          ASSERT_FLOAT_EQ(enc_buf->at(Index{h, w, c, m}), pull_buf.at(Index{m, h, w, c}));

  ASSERT_EQ(loco::Domain::DepthwiseFilter, locomotiv::annot_domain(enc));
}
