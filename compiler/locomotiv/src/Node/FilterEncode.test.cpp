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

TEST(NodeExecution_FilterEncode, s32)
{
  const uint32_t N = 2;
  const uint32_t H = 3;
  const uint32_t W = 4;
  const uint32_t C = 5;

  auto g = loco::make_graph();

  // Pull
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::S32);

  // Make and assign "NCHW" data to pull node
  auto pull_buf = make_buffer<int32_t, LexicalLayout>(Shape{N, C, H, W});
  int32_t i = 1;
  for (IndexEnumerator e{pull_buf.shape()}; e.valid(); e.advance())
  {
    pull_buf.at(e.current()) = i;
    ++i; // Doesn't matter what it is
  }
  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::annot_data(pull, std::move(pull_data));
  locomotiv::annot_domain(pull, loco::Domain::Tensor);

  // Encoder to correctly read input tensor as NCHW
  auto encoder = std::unique_ptr<loco::PermutingEncoder<loco::Domain::Filter>>(
    new loco::PermutingEncoder<loco::Domain::Filter>);
  encoder->perm()->axis(loco::FilterAxis::Count) = 0;
  encoder->perm()->axis(loco::FilterAxis::Depth) = 1;
  encoder->perm()->axis(loco::FilterAxis::Height) = 2;
  encoder->perm()->axis(loco::FilterAxis::Width) = 3;

  // FilterEncode
  auto enc = g->nodes()->create<loco::FilterEncode>();
  enc->input(pull);
  enc->encoder(std::move(encoder));

  locomotiv::NodeExecution::get().run(enc);

  auto enc_data = locomotiv::annot_data(enc);
  ASSERT_NE(enc_data, nullptr);
  ASSERT_EQ(loco::DataType::S32, enc_data->dtype());
  ASSERT_EQ((Shape{N, H, W, C}), *(enc_data->shape())); // locomotiv filter is NHWC
  auto enc_buf = enc_data->as_s32_bufptr();
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w)
        for (uint32_t c = 0; c < C; ++c)
          ASSERT_EQ(enc_buf->at(Index{n, h, w, c}), pull_buf.at(Index{n, c, h, w}));

  ASSERT_EQ(loco::Domain::Filter, locomotiv::annot_domain(enc));
}

TEST(NodeExecution_FilterEncode, f32)
{
  const uint32_t N = 2;
  const uint32_t H = 3;
  const uint32_t W = 4;
  const uint32_t C = 5;

  auto g = loco::make_graph();

  // Pull
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::FLOAT32);

  // Make and assign crazy "CHNW" data to pull node
  auto pull_buf = make_buffer<float, LexicalLayout>(Shape{C, H, N, W});
  float f = 1;
  for (IndexEnumerator e{pull_buf.shape()}; e.valid(); e.advance())
  {
    pull_buf.at(e.current()) = f;
    f += 0.1f; // Doesn't matter what it is
  }
  auto pull_data = locomotiv::make_data(pull_buf);
  locomotiv::annot_data(pull, std::move(pull_data));
  locomotiv::annot_domain(pull, loco::Domain::Tensor);

  // Encoder to correctly read input tensor as CHNW
  auto encoder = std::unique_ptr<loco::PermutingEncoder<loco::Domain::Filter>>(
    new loco::PermutingEncoder<loco::Domain::Filter>);
  encoder->perm()->axis(loco::FilterAxis::Depth) = 0;
  encoder->perm()->axis(loco::FilterAxis::Height) = 1;
  encoder->perm()->axis(loco::FilterAxis::Count) = 2;
  encoder->perm()->axis(loco::FilterAxis::Width) = 3;

  // FilterEncode
  auto enc = g->nodes()->create<loco::FilterEncode>();
  enc->input(pull);
  enc->encoder(std::move(encoder));

  locomotiv::NodeExecution::get().run(enc);

  auto enc_data = locomotiv::annot_data(enc);
  ASSERT_NE(enc_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, enc_data->dtype());
  ASSERT_EQ((Shape{N, H, W, C}), *(enc_data->shape())); // locomotiv filter is NHWC
  auto enc_buf = enc_data->as_f32_bufptr();
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w)
        for (uint32_t c = 0; c < C; ++c)
          ASSERT_FLOAT_EQ(enc_buf->at(Index{n, h, w, c}), pull_buf.at(Index{c, h, n, w}));

  ASSERT_EQ(loco::Domain::Filter, locomotiv::annot_domain(enc));
}
