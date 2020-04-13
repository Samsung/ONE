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

#ifndef __PADDING_UTILS_H__
#define __PADDING_UTILS_H__

#include "Padding.h"

#include <caffe/proto/caffe.pb.h>

/**
 * @brief Construct a raw padding from each Layer parameter
 *
 * @note This class is an auxiliary class for build_raw_padding function below
 */
class RawPaddingBuilder
{
public:
  friend RawPaddingBuilder build_raw_padding(void);

private:
  RawPaddingBuilder() = default;

public:
  RawPadding with(const ::caffe::ConvolutionParameter &) const;
  RawPadding with(const ::caffe::PoolingParameter &) const;
};

/**
 * RawPaddingBuilder is introduced to support the following code pattern:
 *
 *   auto raw_padding = build_raw_padding().with(conv_param);
 *   ...
 */
RawPaddingBuilder build_raw_padding(void);

/**
 * @brief Convert a raw padding to a spatial padding of a given spatial rank
 *
 * @note This class is an auxiliary class for build_raw_padding function below
 */
class SpatialPaddingBuilder
{
public:
  friend SpatialPaddingBuilder build_spatial_padding(uint32_t spatial_rank);

private:
  SpatialPaddingBuilder(uint32_t spatial_rank) : _spatial_rank{spatial_rank}
  {
    // DO NOTHING
  }

public:
  SpatialPadding with(const RawPadding &raw) const;

private:
  uint32_t _spatial_rank = 0;
};

/**
 * SpatialPaddingBuilder is introduced to support the following code pattern:
 *
 *   auto raw_padding = build_raw_padding().with(conv_param);
 *   auto spatial_padding = build_spatial_padding(4).with(raw_padding);
 */
SpatialPaddingBuilder build_spatial_padding(uint32_t spatial_rank);

#endif // __PADDING_UTILS_H__
