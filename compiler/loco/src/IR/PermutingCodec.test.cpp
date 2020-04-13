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

#include "loco/IR/PermutingCodec.h"

#include <gtest/gtest.h>

using namespace loco;

TEST(PemutationTest, feature)
{
  Permutation<Domain::Feature> perm;

  // All values are invalid at the beginning
  ASSERT_FALSE(perm.mapped(FeatureAxis::Count));
  ASSERT_FALSE(perm.mapped(FeatureAxis::Depth));
  ASSERT_FALSE(perm.mapped(FeatureAxis::Height));
  ASSERT_FALSE(perm.mapped(FeatureAxis::Width));

  // Update mapping
  perm[FeatureAxis::Count] = 5;
  perm[FeatureAxis::Depth] = 6;
  perm[FeatureAxis::Height] = 7;
  perm[FeatureAxis::Width] = 8;

  // Now perm has a mapping for all the axes
  ASSERT_TRUE(perm.mapped(FeatureAxis::Count));
  ASSERT_TRUE(perm.mapped(FeatureAxis::Depth));
  ASSERT_TRUE(perm.mapped(FeatureAxis::Height));
  ASSERT_TRUE(perm.mapped(FeatureAxis::Width));

  // Check the value
  ASSERT_EQ(perm[FeatureAxis::Count], 5);
  ASSERT_EQ(perm[FeatureAxis::Depth], 6);
  ASSERT_EQ(perm[FeatureAxis::Height], 7);
  ASSERT_EQ(perm[FeatureAxis::Width], 8);
}

TEST(PemutationTest, filter)
{
  Permutation<Domain::Filter> perm;

  // All values are invalid at the beginning
  ASSERT_FALSE(perm.mapped(FilterAxis::Count));
  ASSERT_FALSE(perm.mapped(FilterAxis::Depth));
  ASSERT_FALSE(perm.mapped(FilterAxis::Height));
  ASSERT_FALSE(perm.mapped(FilterAxis::Width));

  // Update mapping
  perm[FilterAxis::Count] = 5;
  perm[FilterAxis::Depth] = 6;
  perm[FilterAxis::Height] = 7;
  perm[FilterAxis::Width] = 8;

  // Now perm has a mapping for all the axes
  ASSERT_TRUE(perm.mapped(FilterAxis::Count));
  ASSERT_TRUE(perm.mapped(FilterAxis::Depth));
  ASSERT_TRUE(perm.mapped(FilterAxis::Height));
  ASSERT_TRUE(perm.mapped(FilterAxis::Width));

  // Check the value
  ASSERT_EQ(perm[FilterAxis::Count], 5);
  ASSERT_EQ(perm[FilterAxis::Depth], 6);
  ASSERT_EQ(perm[FilterAxis::Height], 7);
  ASSERT_EQ(perm[FilterAxis::Width], 8);
}

TEST(PemutationTest, depthwise_filter)
{
  Permutation<Domain::DepthwiseFilter> perm;

  // All values are invalid at the beginning
  ASSERT_FALSE(perm.mapped(DepthwiseFilterAxis::Depth));
  ASSERT_FALSE(perm.mapped(DepthwiseFilterAxis::Multiplier));
  ASSERT_FALSE(perm.mapped(DepthwiseFilterAxis::Height));
  ASSERT_FALSE(perm.mapped(DepthwiseFilterAxis::Width));

  // Update mapping
  perm[DepthwiseFilterAxis::Depth] = 5;
  perm[DepthwiseFilterAxis::Multiplier] = 6;
  perm[DepthwiseFilterAxis::Height] = 7;
  perm[DepthwiseFilterAxis::Width] = 8;

  // Now perm has a mapping for all the axes
  ASSERT_TRUE(perm.mapped(DepthwiseFilterAxis::Depth));
  ASSERT_TRUE(perm.mapped(DepthwiseFilterAxis::Multiplier));
  ASSERT_TRUE(perm.mapped(DepthwiseFilterAxis::Height));
  ASSERT_TRUE(perm.mapped(DepthwiseFilterAxis::Width));

  // Check the value
  ASSERT_EQ(perm[DepthwiseFilterAxis::Depth], 5);
  ASSERT_EQ(perm[DepthwiseFilterAxis::Multiplier], 6);
  ASSERT_EQ(perm[DepthwiseFilterAxis::Height], 7);
  ASSERT_EQ(perm[DepthwiseFilterAxis::Width], 8);
}

TEST(PermutingEncoderTest, feature)
{
  PermutingEncoder<Domain::Feature> enc;

  // Encoder is invalid at the beginning
  ASSERT_FALSE(enc.valid());

  // Set "invalid" mapping
  enc.perm()->axis(FeatureAxis::Count) = 0;
  enc.perm()->axis(FeatureAxis::Depth) = 6;
  enc.perm()->axis(FeatureAxis::Height) = 1;
  enc.perm()->axis(FeatureAxis::Width) = 2;

  // Encoder is still invalid
  ASSERT_FALSE(enc.valid());

  // Set another "invalid" mapping
  enc.perm()->axis(FeatureAxis::Depth) = 1;

  // Encoder is still invalid
  ASSERT_FALSE(enc.valid());

  // Set "valid" mapping
  enc.perm()->axis(FeatureAxis::Depth) = 3;

  // Encoder is now valid
  ASSERT_TRUE(enc.valid());

  // Let's test with a HD (1280x720) RGB image
  TensorShape tensor_shape;

  tensor_shape.rank(4);
  tensor_shape.dim(0) = 1;    // COUNT
  tensor_shape.dim(1) = 720;  // HEIGHT
  tensor_shape.dim(2) = 1280; // WIDTH
  tensor_shape.dim(3) = 3;    // DEPTH

  // Get the feature shape corresponding to a given image
  auto feature_shape = enc.shape(tensor_shape);

  ASSERT_EQ(feature_shape.count(), 1);
  ASSERT_EQ(feature_shape.depth(), 3);
  ASSERT_EQ(feature_shape.height(), 720);
  ASSERT_EQ(feature_shape.width(), 1280);

  // Let's find a source tensor index!
  FeatureIndex feature_index;

  feature_index.batch() = 0;
  feature_index.channel() = 1;
  feature_index.row() = 2;
  feature_index.column() = 3;

  auto tensor_index = enc.value(feature_index);

  ASSERT_EQ(tensor_index.at(0), 0); // BATCH(COUNT)
  ASSERT_EQ(tensor_index.at(1), 2); // ROW(HEIGHT)
  ASSERT_EQ(tensor_index.at(2), 3); // COLUMN(WIDTH)
  ASSERT_EQ(tensor_index.at(3), 1); // CHANNEL(DEPTH)
}

TEST(PermutingEncoderTest, feature_clone)
{
  PermutingEncoder<Domain::Feature> src_enc;

  auto src_perm = src_enc.perm();

  src_perm->axis(FeatureAxis::Count) = 0;
  src_perm->axis(FeatureAxis::Depth) = 3;
  src_perm->axis(FeatureAxis::Height) = 1;
  src_perm->axis(FeatureAxis::Width) = 2;

  auto dst_enc = src_enc.clone();
  auto dst_perm = dynamic_cast<PermutingEncoder<Domain::Feature> *>(dst_enc.get())->perm();

  EXPECT_EQ(dst_perm->axis(FeatureAxis::Count), src_perm->axis(FeatureAxis::Count));
  EXPECT_EQ(dst_perm->axis(FeatureAxis::Depth), src_perm->axis(FeatureAxis::Depth));
  EXPECT_EQ(dst_perm->axis(FeatureAxis::Height), src_perm->axis(FeatureAxis::Height));
  EXPECT_EQ(dst_perm->axis(FeatureAxis::Width), src_perm->axis(FeatureAxis::Width));

  // Update on cloned encoder SHOULD NOT affect the original encoder
  dst_perm->axis(FeatureAxis::Height) += 1;

  EXPECT_EQ(src_perm->axis(FeatureAxis::Height), 1);
  EXPECT_EQ(dst_perm->axis(FeatureAxis::Height), 2);
}

TEST(PermutingEncoderTest, filter)
{
  PermutingEncoder<Domain::Filter> enc;

  // Encoder is invalid at the beginning
  ASSERT_FALSE(enc.valid());

  // Set "invalid" mapping
  enc.perm()->axis(FilterAxis::Count) = 0;
  enc.perm()->axis(FilterAxis::Depth) = 6;
  enc.perm()->axis(FilterAxis::Height) = 1;
  enc.perm()->axis(FilterAxis::Width) = 2;

  // Encoder is still invalid
  ASSERT_FALSE(enc.valid());

  // Set another "invalid" mapping
  enc.perm()->axis(FilterAxis::Depth) = 1;

  // Encoder is still invalid
  ASSERT_FALSE(enc.valid());

  // Set "valid" mapping
  enc.perm()->axis(FilterAxis::Depth) = 3;

  // Encoder is now valid
  ASSERT_TRUE(enc.valid());

  TensorShape tensor_shape;

  tensor_shape.rank(4);
  tensor_shape.dim(0) = 8; // COUNT
  tensor_shape.dim(1) = 1; // HEIGHT
  tensor_shape.dim(2) = 7; // WIDTH
  tensor_shape.dim(3) = 4; // DEPTH

  // Get the corresponding filter shape
  auto filter_shape = enc.shape(tensor_shape);

  ASSERT_EQ(filter_shape.count(), 8);
  ASSERT_EQ(filter_shape.depth(), 4);
  ASSERT_EQ(filter_shape.height(), 1);
  ASSERT_EQ(filter_shape.width(), 7);

  // Let's find a source tensor index!
  FilterIndex filter_index;

  filter_index.nth() = 1;
  filter_index.channel() = 2;
  filter_index.row() = 0;
  filter_index.column() = 3;

  auto tensor_index = enc.value(filter_index);

  ASSERT_EQ(tensor_index.at(0), 1); // NTH(COUNT)
  ASSERT_EQ(tensor_index.at(1), 0); // ROW(HEIGHT)
  ASSERT_EQ(tensor_index.at(2), 3); // COLUMN(WIDTH)
  ASSERT_EQ(tensor_index.at(3), 2); // CHANNEL(DEPTH)
}

TEST(PermutingEncoderTest, depthwise_filter)
{
  PermutingEncoder<Domain::DepthwiseFilter> enc;

  // Encoder is invalid at the beginning
  ASSERT_FALSE(enc.valid());

  // Set "invalid" mapping
  enc.perm()->axis(DepthwiseFilterAxis::Depth) = 0;
  enc.perm()->axis(DepthwiseFilterAxis::Multiplier) = 6;
  enc.perm()->axis(DepthwiseFilterAxis::Height) = 1;
  enc.perm()->axis(DepthwiseFilterAxis::Width) = 2;

  // Encoder is still invalid
  ASSERT_FALSE(enc.valid());

  // Set another "invalid" mapping
  enc.perm()->axis(DepthwiseFilterAxis::Multiplier) = 1;

  // Encoder is still invalid
  ASSERT_FALSE(enc.valid());

  // Set "valid" mapping
  enc.perm()->axis(DepthwiseFilterAxis::Multiplier) = 3;

  // Encoder is now valid
  ASSERT_TRUE(enc.valid());

  TensorShape tensor_shape;

  tensor_shape.rank(4);
  tensor_shape.dim(0) = 8; // DEPTH
  tensor_shape.dim(1) = 1; // HEIGHT
  tensor_shape.dim(2) = 7; // WIDTH
  tensor_shape.dim(3) = 4; // MULTIPLIER

  // Get the corresponding depthwise filter shape
  auto filter_shape = enc.shape(tensor_shape);

  ASSERT_EQ(filter_shape.depth(), 8);
  ASSERT_EQ(filter_shape.multiplier(), 4);
  ASSERT_EQ(filter_shape.height(), 1);
  ASSERT_EQ(filter_shape.width(), 7);

  // Let's find a source tensor index!
  DepthwiseFilterIndex filter_index;

  filter_index.channel() = 1;
  filter_index.nth() = 2;
  filter_index.row() = 0;
  filter_index.column() = 3;

  auto tensor_index = enc.value(filter_index);

  ASSERT_EQ(tensor_index.at(0), 1); // CHANNEL(DEPTH)
  ASSERT_EQ(tensor_index.at(1), 0); // ROW(HEIGHT)
  ASSERT_EQ(tensor_index.at(2), 3); // COLUMN(WIDTH)
  ASSERT_EQ(tensor_index.at(3), 2); // NTH(MULTIPLIER)
}

TEST(PermutingEncoderTest, depthwisefilter_init)
{
  Permutation<Domain::DepthwiseFilter> src_perm;

  src_perm.axis(DepthwiseFilterAxis::Multiplier) = 0;
  src_perm.axis(DepthwiseFilterAxis::Depth) = 3;
  src_perm.axis(DepthwiseFilterAxis::Height) = 1;
  src_perm.axis(DepthwiseFilterAxis::Width) = 2;

  PermutingEncoder<Domain::DepthwiseFilter> dst_enc{src_perm};
  auto dst_perm = dst_enc.perm();

  EXPECT_EQ(dst_perm->axis(DepthwiseFilterAxis::Multiplier),
            src_perm.axis(DepthwiseFilterAxis::Multiplier));
  EXPECT_EQ(dst_perm->axis(DepthwiseFilterAxis::Depth), src_perm.axis(DepthwiseFilterAxis::Depth));
  EXPECT_EQ(dst_perm->axis(DepthwiseFilterAxis::Height),
            src_perm.axis(DepthwiseFilterAxis::Height));
  EXPECT_EQ(dst_perm->axis(DepthwiseFilterAxis::Width), src_perm.axis(DepthwiseFilterAxis::Width));

  // Update on dst perm SHOULD NOT affect the src perm
  dst_perm->axis(DepthwiseFilterAxis::Height) += 1;

  EXPECT_EQ(src_perm.axis(DepthwiseFilterAxis::Height), 1);
  EXPECT_EQ(dst_perm->axis(DepthwiseFilterAxis::Height), 2);
}

TEST(PermutingDecoderTest, feature)
{
  PermutingDecoder<Domain::Feature> dec;

  // Decoder is invalid at the beginning
  ASSERT_FALSE(dec.valid());

  // Set "invalid" mapping
  dec.perm()->axis(FeatureAxis::Count) = 0;
  dec.perm()->axis(FeatureAxis::Depth) = 6;
  dec.perm()->axis(FeatureAxis::Height) = 1;
  dec.perm()->axis(FeatureAxis::Width) = 2;

  // Decoder is still invalid
  ASSERT_FALSE(dec.valid());

  // Set another "invalid" mapping
  dec.perm()->axis(FeatureAxis::Depth) = 1;

  // Decoder is still invalid
  ASSERT_FALSE(dec.valid());

  // Set "valid" mapping
  dec.perm()->axis(FeatureAxis::Depth) = 3;

  // Decoder is now valid
  ASSERT_TRUE(dec.valid());

  // Let's test with a HD (1280x720) RGB image
  FeatureShape feature_shape;

  feature_shape.count() = 1;
  feature_shape.depth() = 3;
  feature_shape.height() = 720;
  feature_shape.width() = 1280;

  // Get the tensor shape corresponding to a given image
  auto tensor_shape = dec.shape(feature_shape);

  ASSERT_EQ(tensor_shape.rank(), 4);
  ASSERT_EQ(tensor_shape.dim(0), 1);    // COUNT
  ASSERT_EQ(tensor_shape.dim(1), 720);  // HEIGHT
  ASSERT_EQ(tensor_shape.dim(2), 1280); // WIDTH
  ASSERT_EQ(tensor_shape.dim(3), 3);    // DEPTH

  // Let's find a source feature index!
  TensorIndex tensor_index;

  tensor_index.resize(4);

  tensor_index.at(0) = 0; // BATCH(COUNT)
  tensor_index.at(3) = 1; // CHANNEL(DEPTH)
  tensor_index.at(1) = 2; // ROW(HEIGHT)
  tensor_index.at(2) = 3; // COLUMN(WIDTH)

  auto feature_index = dec.value(tensor_index);

  ASSERT_EQ(feature_index.batch(), 0);
  ASSERT_EQ(feature_index.channel(), 1);
  ASSERT_EQ(feature_index.row(), 2);
  ASSERT_EQ(feature_index.column(), 3);
}

TEST(PermutingDecoderTest, feature_clone)
{
  PermutingDecoder<Domain::Feature> src_enc;

  auto src_perm = src_enc.perm();

  src_perm->axis(FeatureAxis::Count) = 0;
  src_perm->axis(FeatureAxis::Depth) = 3;
  src_perm->axis(FeatureAxis::Height) = 1;
  src_perm->axis(FeatureAxis::Width) = 2;

  auto dst_enc = src_enc.clone();
  auto dst_perm = dynamic_cast<PermutingDecoder<Domain::Feature> *>(dst_enc.get())->perm();

  EXPECT_EQ(dst_perm->axis(FeatureAxis::Count), src_perm->axis(FeatureAxis::Count));
  EXPECT_EQ(dst_perm->axis(FeatureAxis::Depth), src_perm->axis(FeatureAxis::Depth));
  EXPECT_EQ(dst_perm->axis(FeatureAxis::Height), src_perm->axis(FeatureAxis::Height));
  EXPECT_EQ(dst_perm->axis(FeatureAxis::Width), src_perm->axis(FeatureAxis::Width));

  // Update on cloned decoder SHOULD NOT affect the original decoder
  dst_perm->axis(FeatureAxis::Height) += 1;

  EXPECT_EQ(src_perm->axis(FeatureAxis::Height), 1);
  EXPECT_EQ(dst_perm->axis(FeatureAxis::Height), 2);
}

TEST(PermutingDecoderTest, filter)
{
  PermutingDecoder<Domain::Filter> dec;

  // Decoder is invalid at the beginning
  ASSERT_FALSE(dec.valid());

  // Set "invalid" mapping
  dec.perm()->axis(FilterAxis::Count) = 0;
  dec.perm()->axis(FilterAxis::Depth) = 6;
  dec.perm()->axis(FilterAxis::Height) = 1;
  dec.perm()->axis(FilterAxis::Width) = 2;

  // Decoder is still invalid
  ASSERT_FALSE(dec.valid());

  // Set another "invalid" mapping
  dec.perm()->axis(FilterAxis::Depth) = 1;

  // Decoder is still invalid
  ASSERT_FALSE(dec.valid());

  // Set "valid" mapping
  dec.perm()->axis(FilterAxis::Depth) = 3;

  // Decoder is now valid
  ASSERT_TRUE(dec.valid());

  // Let's test with a small filter
  FilterShape filter_shape;

  filter_shape.count() = 10;
  filter_shape.depth() = 3;
  filter_shape.height() = 6;
  filter_shape.width() = 8;

  // Get the tensor shape corresponding to a given image
  auto tensor_shape = dec.shape(filter_shape);

  ASSERT_EQ(tensor_shape.rank(), 4);
  ASSERT_EQ(tensor_shape.dim(0), 10); // COUNT
  ASSERT_EQ(tensor_shape.dim(1), 6);  // HEIGHT
  ASSERT_EQ(tensor_shape.dim(2), 8);  // WIDTH
  ASSERT_EQ(tensor_shape.dim(3), 3);  // DEPTH

  // Let's find a source filter index!
  TensorIndex tensor_index;

  tensor_index.resize(4);

  tensor_index.at(0) = 0; // BATCH(COUNT)
  tensor_index.at(3) = 1; // CHANNEL(DEPTH)
  tensor_index.at(1) = 2; // ROW(HEIGHT)
  tensor_index.at(2) = 3; // COLUMN(WIDTH)

  auto filter_index = dec.value(tensor_index);

  ASSERT_EQ(filter_index.nth(), 0);
  ASSERT_EQ(filter_index.channel(), 1);
  ASSERT_EQ(filter_index.row(), 2);
  ASSERT_EQ(filter_index.column(), 3);
}

TEST(PermutingDecoderTest, depthwise_filter)
{
  PermutingDecoder<Domain::DepthwiseFilter> dec;

  // Decoder is invalid at the beginning
  ASSERT_FALSE(dec.valid());

  // Set "invalid" mapping
  dec.perm()->axis(DepthwiseFilterAxis::Depth) = 0;
  dec.perm()->axis(DepthwiseFilterAxis::Multiplier) = 6;
  dec.perm()->axis(DepthwiseFilterAxis::Height) = 1;
  dec.perm()->axis(DepthwiseFilterAxis::Width) = 2;

  // Decoder is still invalid
  ASSERT_FALSE(dec.valid());

  // Set another "invalid" mapping
  dec.perm()->axis(DepthwiseFilterAxis::Multiplier) = 1;

  // Decoder is still invalid
  ASSERT_FALSE(dec.valid());

  // Set "valid" mapping
  dec.perm()->axis(DepthwiseFilterAxis::Multiplier) = 3;

  // Decoder is now valid
  ASSERT_TRUE(dec.valid());

  DepthwiseFilterShape dw_filter_shape;

  dw_filter_shape.depth() = 8;
  dw_filter_shape.multiplier() = 1;
  dw_filter_shape.height() = 7;
  dw_filter_shape.width() = 4;

  // Get the corresponding depthwise filter shape
  auto tensor_shape = dec.shape(dw_filter_shape);

  ASSERT_EQ(tensor_shape.dim(0).value(), 8);
  ASSERT_EQ(tensor_shape.dim(1).value(), 7);
  ASSERT_EQ(tensor_shape.dim(2).value(), 4);
  ASSERT_EQ(tensor_shape.dim(3).value(), 1);

  // Let's find a source tensor index!
  TensorIndex tensor_index;
  tensor_index.resize(4);

  tensor_index.at(0) = 4;
  tensor_index.at(1) = 2;
  tensor_index.at(2) = 1;
  tensor_index.at(3) = 0;

  auto dw_filter_index = dec.value(tensor_index);

  ASSERT_EQ(dw_filter_index.channel(), 4);
  ASSERT_EQ(dw_filter_index.nth(), 0);
  ASSERT_EQ(dw_filter_index.row(), 2);
  ASSERT_EQ(dw_filter_index.column(), 1);
}
