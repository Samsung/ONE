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

// This file is intended to test MatrixEncode and MatrixDecode at once
namespace
{

class NodeExecution_MatrixCodec : public ::testing::Test
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

  /// @brief Make MatrixEncode node with given input and encoding permutation
  loco::MatrixEncode *matrix_encode_layer(loco::Node *input,
                                          const loco::Permutation<loco::Domain::Matrix> &perm)
  {
    auto encoder = std::unique_ptr<loco::PermutingEncoder<loco::Domain::Matrix>>(
        new loco::PermutingEncoder<loco::Domain::Matrix>);

    encoder->perm(perm);

    auto enc = g.nodes()->create<loco::MatrixEncode>();
    enc->input(input);
    enc->encoder(std::move(encoder));

    return enc;
  }

  /// @brief Make MatrixDecode node with given input and decoding permutation
  loco::MatrixDecode *matrix_decode_layer(loco::Node *input,
                                          const loco::Permutation<loco::Domain::Matrix> &perm)
  {
    auto decoder = std::unique_ptr<loco::PermutingDecoder<loco::Domain::Matrix>>(
        new loco::PermutingDecoder<loco::Domain::Matrix>);

    decoder->perm(perm);

    auto dec = g.nodes()->create<loco::MatrixDecode>();
    dec->input(input);
    dec->decoder(std::move(decoder));

    return dec;
  }
};

} // namespace

TEST_F(NodeExecution_MatrixCodec, HW_s32)
{
  const uint32_t H = 3;
  const uint32_t W = 4;

  // Make HW data for pull node
  auto pull_buf = make_buffer<int32_t, LexicalLayout>(Shape{H, W});
  int32_t i = 0;
  for (IndexEnumerator e{pull_buf.shape()}; e.valid(); e.advance())
  {
    pull_buf.at(e.current()) = i;
    ++i; // Doesn't matter what it is
  }

  // Make HW permutation for encoder and decoder
  loco::Permutation<loco::Domain::Matrix> HW;

  HW.axis(loco::MatrixAxis::Height) = 0;
  HW.axis(loco::MatrixAxis::Width) = 1;

  // Pull
  auto pull = pull_layer(pull_buf, loco::DataType::S32);

  // MatrixEncode
  auto enc = matrix_encode_layer(pull, HW);
  locomotiv::NodeExecution::get().run(enc);

  // Test MatrixEncode
  auto enc_data = locomotiv::annot_data(enc);
  ASSERT_NE(enc_data, nullptr);
  ASSERT_EQ(loco::DataType::S32, enc_data->dtype());
  ASSERT_EQ((Shape{H, W}), *(enc_data->shape())); // locomotiv matrix is HW
  auto enc_buf = enc_data->as_s32_bufptr();
  for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
      ASSERT_EQ(enc_buf->at(Index{h, w}), pull_buf.at(Index{h, w}));

  ASSERT_EQ(loco::Domain::Matrix, locomotiv::annot_domain(enc));

  // MatrixDecode
  auto dec = matrix_decode_layer(enc, HW);
  locomotiv::NodeExecution::get().run(dec);

  // Test MatrixDecode: Encode -> Decode == identity
  auto dec_data = locomotiv::annot_data(dec);
  ASSERT_NE(dec_data, nullptr);
  ASSERT_EQ(loco::DataType::S32, dec_data->dtype());
  ASSERT_EQ((Shape{H, W}), *(dec_data->shape()));
  auto dec_buf = dec_data->as_s32_bufptr();
  for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
      ASSERT_EQ(dec_buf->at(Index{h, w}), pull_buf.at(Index{h, w}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(dec));
}

TEST_F(NodeExecution_MatrixCodec, WH_f32)
{
  const uint32_t W = 6;
  const uint32_t H = 5;

  // Make crazy WH data for pull node
  auto pull_buf = make_buffer<float, LexicalLayout>(Shape{W, H});
  float f = 0.0f;
  for (IndexEnumerator e{pull_buf.shape()}; e.valid(); e.advance())
  {
    pull_buf.at(e.current()) = f;
    f += 0.1f; // Doesn't matter what it is
  }

  // Make WH permutation for encoder and decoder
  loco::Permutation<loco::Domain::Matrix> WH;

  WH.axis(loco::MatrixAxis::Width) = 0;
  WH.axis(loco::MatrixAxis::Height) = 1;

  // Pull
  auto pull = pull_layer(pull_buf, loco::DataType::FLOAT32);

  // MatrixEncode
  auto enc = matrix_encode_layer(pull, WH);
  locomotiv::NodeExecution::get().run(enc);

  // Test MatrixEncode
  auto enc_data = locomotiv::annot_data(enc);
  ASSERT_NE(enc_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, enc_data->dtype());
  ASSERT_EQ((Shape{H, W}), *(enc_data->shape())); // locomotiv matrix is HW
  auto enc_buf = enc_data->as_f32_bufptr();
  for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
      ASSERT_FLOAT_EQ(enc_buf->at(Index{h, w}), pull_buf.at(Index{w, h}));

  ASSERT_EQ(loco::Domain::Matrix, locomotiv::annot_domain(enc));

  // MatrixDecode
  auto dec = matrix_decode_layer(enc, WH);
  locomotiv::NodeExecution::get().run(dec);

  // Test MatrixDecode: Encode -> Decode == identity
  auto dec_data = locomotiv::annot_data(dec);
  ASSERT_NE(dec_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, dec_data->dtype());
  ASSERT_EQ((Shape{W, H}), *(dec_data->shape()));
  auto dec_buf = dec_data->as_f32_bufptr();
  for (uint32_t h = 0; h < H; ++h)
    for (uint32_t w = 0; w < W; ++w)
      ASSERT_FLOAT_EQ(dec_buf->at(Index{w, h}), pull_buf.at(Index{w, h}));

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(dec));
}
