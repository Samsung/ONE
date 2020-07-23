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

using nncc::core::ADT::tensor::Buffer;
using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

// This file is intended to test FeatureEncode and FeatureDecode at once
namespace
{

class NodeExecution_FeatureCodec : public ::testing::Test
{
private:
  loco::Graph g;

protected:
  /// @brief Make Pull node and set data by given buffer and data type
  template <typename DT> loco::Pull *pull_layer(Buffer<DT> &pull_buf, loco::DataType dtype)
  {
    auto pull = g.nodes()->create<loco::Pull>();
    pull->dtype(dtype);

    auto pull_data = locomotiv::make_data(pull_buf);
    locomotiv::annot_data(pull, std::move(pull_data));
    locomotiv::annot_domain(pull, loco::Domain::Tensor);

    return pull;
  }

  /// @brief Make FeatureEncode node with given input and encoding permutation
  loco::FeatureEncode *feature_encode_layer(loco::Node *input,
                                            const loco::Permutation<loco::Domain::Feature> &perm)
  {
    auto encoder = std::unique_ptr<loco::PermutingEncoder<loco::Domain::Feature>>(
        new loco::PermutingEncoder<loco::Domain::Feature>);

    encoder->perm(perm);

    auto enc = g.nodes()->create<loco::FeatureEncode>();
    enc->input(input);
    enc->encoder(std::move(encoder));

    return enc;
  }

  /// @brief Make FeatureDecode node with given input and decoding permutation
  loco::FeatureDecode *feature_decode_layer(loco::Node *input,
                                            const loco::Permutation<loco::Domain::Feature> &perm)
  {
    auto decoder = std::unique_ptr<loco::PermutingDecoder<loco::Domain::Feature>>(
        new loco::PermutingDecoder<loco::Domain::Feature>);

    decoder->perm(perm);

    auto dec = g.nodes()->create<loco::FeatureDecode>();
    dec->input(input);
    dec->decoder(std::move(decoder));

    return dec;
  }
};

} // namespace

TEST_F(NodeExecution_FeatureCodec, s32)
{
  const uint32_t N = 2;
  const uint32_t H = 3;
  const uint32_t W = 4;
  const uint32_t C = 5;

  // Make "NCHW" data for pull node
  auto pull_buf = make_buffer<int32_t, LexicalLayout>(Shape{N, C, H, W});
  int32_t i = 0;
  for (IndexEnumerator e{pull_buf.shape()}; e.valid(); e.advance())
  {
    pull_buf.at(e.current()) = i;
    ++i; // Doesn't matter what it is
  }

  // Make NCHW permutation for encoder and decoder
  loco::Permutation<loco::Domain::Feature> NCHW;

  NCHW.axis(loco::FeatureAxis::Count) = 0;
  NCHW.axis(loco::FeatureAxis::Depth) = 1;
  NCHW.axis(loco::FeatureAxis::Height) = 2;
  NCHW.axis(loco::FeatureAxis::Width) = 3;

  // Pull
  auto pull = pull_layer(pull_buf, loco::DataType::S32);

  // FeatureEncode
  auto enc = feature_encode_layer(pull, NCHW);
  locomotiv::NodeExecution::get().run(enc);

  // Test FeatureEncode
  auto enc_data = locomotiv::annot_data(enc);
  ASSERT_NE(enc_data, nullptr);
  ASSERT_EQ(loco::DataType::S32, enc_data->dtype());
  ASSERT_EQ((Shape{N, H, W, C}), *(enc_data->shape())); // locomotiv feature is NHWC
  auto enc_buf = enc_data->as_s32_bufptr();
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w)
        for (uint32_t c = 0; c < C; ++c)
          ASSERT_EQ(enc_buf->at(Index{n, h, w, c}), pull_buf.at(Index{n, c, h, w}));

  ASSERT_EQ(loco::Domain::Feature, locomotiv::annot_domain(enc));

  // FeatureDecode
  auto dec = feature_decode_layer(enc, NCHW);
  locomotiv::NodeExecution::get().run(dec);

  // Test FeatureDecode: Encode -> Decode == identity
  auto dec_data = locomotiv::annot_data(dec);
  ASSERT_NE(dec_data, nullptr);
  ASSERT_EQ(loco::DataType::S32, dec_data->dtype());
  ASSERT_EQ((Shape{N, C, H, W}), *(dec_data->shape()));
  auto dec_buf = dec_data->as_s32_bufptr();
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w)
        for (uint32_t c = 0; c < C; ++c)
          ASSERT_EQ(dec_buf->at(Index{n, c, h, w}), pull_buf.at(Index{n, c, h, w}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(dec));
}

TEST_F(NodeExecution_FeatureCodec, f32)
{
  const uint32_t N = 2;
  const uint32_t H = 3;
  const uint32_t W = 4;
  const uint32_t C = 5;

  // Make crazy "CHNW" data for pull node
  auto pull_buf = make_buffer<float, LexicalLayout>(Shape{C, H, N, W});
  float f = 0.0f;
  for (IndexEnumerator e{pull_buf.shape()}; e.valid(); e.advance())
  {
    pull_buf.at(e.current()) = f;
    f += 0.1f; // Doesn't matter what it is
  }

  // Make CHNW permutation for encoder and decoder
  loco::Permutation<loco::Domain::Feature> CHNW;

  CHNW.axis(loco::FeatureAxis::Depth) = 0;
  CHNW.axis(loco::FeatureAxis::Height) = 1;
  CHNW.axis(loco::FeatureAxis::Count) = 2;
  CHNW.axis(loco::FeatureAxis::Width) = 3;

  // Pull
  auto pull = pull_layer(pull_buf, loco::DataType::FLOAT32);

  // FeatureEncode
  auto enc = feature_encode_layer(pull, CHNW);
  locomotiv::NodeExecution::get().run(enc);

  // Test FeatureEncode
  auto enc_data = locomotiv::annot_data(enc);
  ASSERT_NE(enc_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, enc_data->dtype());
  ASSERT_EQ((Shape{N, H, W, C}), *(enc_data->shape())); // locomotiv feature is NHWC
  auto enc_buf = enc_data->as_f32_bufptr();
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w)
        for (uint32_t c = 0; c < C; ++c)
          ASSERT_FLOAT_EQ(enc_buf->at(Index{n, h, w, c}), pull_buf.at(Index{c, h, n, w}));

  ASSERT_EQ(loco::Domain::Feature, locomotiv::annot_domain(enc));

  // FeatureDecode
  auto dec = feature_decode_layer(enc, CHNW);
  locomotiv::NodeExecution::get().run(dec);

  // Test FeatureDecode: Encode -> Decode == identity
  auto dec_data = locomotiv::annot_data(dec);
  ASSERT_NE(dec_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, dec_data->dtype());
  ASSERT_EQ((Shape{C, H, N, W}), *(dec_data->shape()));
  auto dec_buf = dec_data->as_f32_bufptr();
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t h = 0; h < H; ++h)
      for (uint32_t w = 0; w < W; ++w)
        for (uint32_t c = 0; c < C; ++c)
          ASSERT_FLOAT_EQ(dec_buf->at(Index{c, h, n, w}), pull_buf.at(Index{c, h, n, w}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(dec));
}
