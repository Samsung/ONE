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

/**
 * @file  Padding.h
 * @brief This file declares padding-related data structures.
 */
#ifndef __PADDING_H__
#define __PADDING_H__

#include <cstdint>
#include <vector>

/**
 * @brief A PaddingBase encapsulates common implementation for derived Padding classes
 */
template <typename Derived> class PaddingBase
{
public:
  virtual ~PaddingBase() = default;

public:
  uint32_t count(void) const { return _values.size(); }

public:
  uint32_t &value(uint32_t n) { return _values.at(n); }
  const uint32_t &value(uint32_t n) const { return _values.at(n); }

public:
  void resize(uint32_t len) { return _values.resize(len); }

private:
  std::vector<uint32_t> _values;
};

/**
 * @brief A RawPadding denotes padding values stored in Caffe model
 *
 * @note There may be a mismatch between the number of values in RawPadding and spatial rank
 */
struct RawPadding final : public PaddingBase<RawPadding>
{
  // Empty
};

/**
 * @brief A SpatialPadding denotes padding values for each "spatial" dimension
 *
 * @note The number of values in SpatialPadding should be matched with spatial rank
 */
struct SpatialPadding final : public PaddingBase<SpatialPadding>
{
  // Empty
};

#endif // __PADDING_H__
