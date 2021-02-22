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

#ifndef __CODEC_HELPER_H__
#define __CODEC_HELPER_H__

#include <plier/tf/Convert.h>

#include <memory>

namespace
{

using plier::tf::DataLayout;

void set_feature_enc(loco::FeatureEncode *feature_enc, DataLayout data_layout)
{
  auto enc = std::make_unique<loco::PermutingEncoder<loco::Domain::Feature>>();

  if (data_layout == DataLayout::NHWC)
  {
    enc->perm()->axis(loco::FeatureAxis::Count) = 0;
    enc->perm()->axis(loco::FeatureAxis::Height) = 1;
    enc->perm()->axis(loco::FeatureAxis::Width) = 2;
    enc->perm()->axis(loco::FeatureAxis::Depth) = 3;
  }
  else if (data_layout == DataLayout::NCHW)
  {
    enc->perm()->axis(loco::FeatureAxis::Count) = 0;
    enc->perm()->axis(loco::FeatureAxis::Depth) = 1;
    enc->perm()->axis(loco::FeatureAxis::Height) = 2;
    enc->perm()->axis(loco::FeatureAxis::Width) = 3;
  }

  feature_enc->encoder(std::move(enc));
}

void set_feature_dec(loco::FeatureDecode *feature_dec, DataLayout data_layout)
{
  auto dec = std::make_unique<loco::PermutingDecoder<loco::Domain::Feature>>();

  if (data_layout == DataLayout::NHWC)
  {
    dec->perm()->axis(loco::FeatureAxis::Count) = 0;
    dec->perm()->axis(loco::FeatureAxis::Height) = 1;
    dec->perm()->axis(loco::FeatureAxis::Width) = 2;
    dec->perm()->axis(loco::FeatureAxis::Depth) = 3;
  }
  else if (data_layout == DataLayout::NCHW)
  {
    dec->perm()->axis(loco::FeatureAxis::Count) = 0;
    dec->perm()->axis(loco::FeatureAxis::Depth) = 1;
    dec->perm()->axis(loco::FeatureAxis::Height) = 2;
    dec->perm()->axis(loco::FeatureAxis::Width) = 3;
  }

  feature_dec->decoder(std::move(dec));
}

} // namespace

#endif // __CODEC_HELPER_H__
