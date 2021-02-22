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

#include "GraphBlock.h"

#include "Check.h"

#include <loco.h>
#include <memory>

namespace
{

template <exo::FeatureLayout T> loco::Permutation<loco::Domain::Feature> perm();

template <> loco::Permutation<loco::Domain::Feature> perm<exo::FeatureLayout::NHWC>()
{
  // Make NHWC permutation for encoder and decoder
  loco::Permutation<loco::Domain::Feature> NHWC;

  NHWC.axis(loco::FeatureAxis::Count) = 0;
  NHWC.axis(loco::FeatureAxis::Height) = 1;
  NHWC.axis(loco::FeatureAxis::Width) = 2;
  NHWC.axis(loco::FeatureAxis::Depth) = 3;

  return NHWC;
}

template <exo::FilterLayout T> loco::Permutation<loco::Domain::Filter> perm();

template <> loco::Permutation<loco::Domain::Filter> perm<exo::FilterLayout::HWIO>()
{
  loco::Permutation<loco::Domain::Filter> HWIO; // a.k.a., HWCN

  HWIO.axis(loco::FilterAxis::Height) = 0;
  HWIO.axis(loco::FilterAxis::Width) = 1;
  HWIO.axis(loco::FilterAxis::Depth) = 2;
  HWIO.axis(loco::FilterAxis::Count) = 3;

  return HWIO;
}

template <> loco::Permutation<loco::Domain::Filter> perm<exo::FilterLayout::OHWI>()
{

  // Make NHWC permutation for encoder and decoder
  loco::Permutation<loco::Domain::Filter> OHWI; // a.k.a., NHWC

  OHWI.axis(loco::FilterAxis::Count) = 0;
  OHWI.axis(loco::FilterAxis::Height) = 1;
  OHWI.axis(loco::FilterAxis::Width) = 2;
  OHWI.axis(loco::FilterAxis::Depth) = 3;

  return OHWI;
}

template <exo::DepthwiseFilterLayout T> loco::Permutation<loco::Domain::DepthwiseFilter> perm();

template <>
loco::Permutation<loco::Domain::DepthwiseFilter> perm<exo::DepthwiseFilterLayout::HWCM>()
{
  loco::Permutation<loco::Domain::DepthwiseFilter> HWCM;

  HWCM.axis(loco::DepthwiseFilterAxis::Height) = 0;
  HWCM.axis(loco::DepthwiseFilterAxis::Width) = 1;
  HWCM.axis(loco::DepthwiseFilterAxis::Depth) = 2;
  HWCM.axis(loco::DepthwiseFilterAxis::Multiplier) = 3;

  return HWCM;
}

template <exo::MatrixLayout T> loco::Permutation<loco::Domain::Matrix> perm();

template <> loco::Permutation<loco::Domain::Matrix> perm<exo::MatrixLayout::HW>()
{
  loco::Permutation<loco::Domain::Matrix> HW;

  HW.axis(loco::MatrixAxis::Height) = 0;
  HW.axis(loco::MatrixAxis::Width) = 1;

  return HW;
}

template <> loco::Permutation<loco::Domain::Matrix> perm<exo::MatrixLayout::WH>()
{
  loco::Permutation<loco::Domain::Matrix> WH;

  WH.axis(loco::MatrixAxis::Height) = 1;
  WH.axis(loco::MatrixAxis::Width) = 0;

  return WH;
}

} // namespace

namespace exo
{

template <FeatureLayout T> loco::FeatureEncode *make_feature_encode(loco::Node *input_for_encode)
{
  EXO_ASSERT(input_for_encode != nullptr, "input should not be nullptr");
  loco::Graph *g = input_for_encode->graph();

  auto encoder = std::make_unique<loco::PermutingEncoder<loco::Domain::Feature>>();

  encoder->perm(perm<T>());

  auto enc = g->nodes()->create<loco::FeatureEncode>();
  enc->input(input_for_encode);
  enc->encoder(std::move(encoder));

  return enc;
}

template <FeatureLayout T> loco::FeatureDecode *make_feature_decode(loco::Node *input_for_decode)
{
  EXO_ASSERT(input_for_decode != nullptr, "input should not be nullptr");
  loco::Graph *g = input_for_decode->graph();

  auto decoder = std::make_unique<loco::PermutingDecoder<loco::Domain::Feature>>();

  decoder->perm(perm<T>());

  auto dec = g->nodes()->create<loco::FeatureDecode>();
  dec->input(input_for_decode);
  dec->decoder(std::move(decoder));

  return dec;
}

template <FilterLayout T> loco::FilterEncode *make_filter_encode(loco::Node *input_for_encode)
{
  EXO_ASSERT(input_for_encode != nullptr, "filter should not be nullptr");
  loco::Graph *g = input_for_encode->graph();

  auto encoder = std::make_unique<loco::PermutingEncoder<loco::Domain::Filter>>();

  encoder->perm(perm<T>());

  auto enc = g->nodes()->create<loco::FilterEncode>();
  enc->input(input_for_encode);
  enc->encoder(std::move(encoder));

  return enc;
}

template <FilterLayout T> loco::FilterDecode *make_filter_decode(loco::Node *input_for_decode)
{
  EXO_ASSERT(input_for_decode != nullptr, "filter should not be nullptr");
  loco::Graph *g = input_for_decode->graph();

  auto decoder = std::make_unique<loco::PermutingDecoder<loco::Domain::Filter>>();

  decoder->perm(perm<T>());

  auto dec = g->nodes()->create<loco::FilterDecode>();
  dec->input(input_for_decode);
  dec->decoder(std::move(decoder));

  return dec;
}

template <DepthwiseFilterLayout T>
loco::DepthwiseFilterDecode *make_dw_filter_decode(loco::Node *input_for_decode)
{
  EXO_ASSERT(input_for_decode != nullptr, "filter should not be nullptr");
  loco::Graph *g = input_for_decode->graph();

  auto decoder = std::make_unique<loco::PermutingDecoder<loco::Domain::DepthwiseFilter>>();

  decoder->perm(perm<T>());

  auto dec = g->nodes()->create<loco::DepthwiseFilterDecode>();
  dec->input(input_for_decode);
  dec->decoder(std::move(decoder));

  return dec;
}

template <MatrixLayout T> loco::MatrixEncode *make_matrix_encode(loco::Node *input_for_encode)
{
  EXO_ASSERT(input_for_encode != nullptr, "input should not be nullptr");
  loco::Graph *g = input_for_encode->graph();

  auto encoder = std::make_unique<loco::PermutingEncoder<loco::Domain::Matrix>>();

  encoder->perm(perm<T>());

  auto enc = g->nodes()->create<loco::MatrixEncode>();
  enc->input(input_for_encode);
  enc->encoder(std::move(encoder));

  return enc;
}

template <MatrixLayout T> loco::MatrixDecode *make_matrix_decode(loco::Node *input_for_decode)
{
  EXO_ASSERT(input_for_decode != nullptr, "input should not be nullptr");
  loco::Graph *g = input_for_decode->graph();

  auto decoder = std::make_unique<loco::PermutingDecoder<loco::Domain::Matrix>>();

  decoder->perm(perm<T>());

  auto dec = g->nodes()->create<loco::MatrixDecode>();
  dec->input(input_for_decode);
  dec->decoder(std::move(decoder));

  return dec;
}

// template instantiation
template loco::FeatureEncode *
make_feature_encode<FeatureLayout::NHWC>(loco::Node *input_for_encode);

template loco::FeatureDecode *
make_feature_decode<FeatureLayout::NHWC>(loco::Node *input_for_encode);

template loco::FilterEncode *make_filter_encode<FilterLayout::HWIO>(loco::Node *input_for_encode);
template loco::FilterDecode *make_filter_decode<FilterLayout::OHWI>(loco::Node *input_for_decode);

template loco::DepthwiseFilterDecode *
make_dw_filter_decode<DepthwiseFilterLayout::HWCM>(loco::Node *input_for_decode);

template loco::MatrixEncode *make_matrix_encode<MatrixLayout::HW>(loco::Node *input_for_encode);
template loco::MatrixEncode *make_matrix_encode<MatrixLayout::WH>(loco::Node *input_for_encode);
template loco::MatrixDecode *make_matrix_decode<MatrixLayout::HW>(loco::Node *input_for_decode);
template loco::MatrixDecode *make_matrix_decode<MatrixLayout::WH>(loco::Node *input_for_decode);

} // namespace exo
