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

#include <memory>
#include <cassert>
#include <set>
#include <stdexcept>

/**
 * Feature Domain
 */
namespace
{

using loco::FeatureAxis;

inline bool valid(const FeatureAxis &axis)
{
  switch (axis)
  {
    case FeatureAxis::Count:
      return true;
    case FeatureAxis::Depth:
      return true;
    case FeatureAxis::Height:
      return true;
    case FeatureAxis::Width:
      return true;
    default:
      break;
  }

  return false;
}

inline bool valid(const loco::Permutation<loco::Domain::Feature> &perm)
{
  auto check = [&perm](FeatureAxis axis_f) {
    if (!perm.mapped(axis_f))
      return false;
    return perm.axis(axis_f) < 4;
  };

  if (!check(FeatureAxis::Count))
    return false;
  if (!check(FeatureAxis::Depth))
    return false;
  if (!check(FeatureAxis::Height))
    return false;
  if (!check(FeatureAxis::Width))
    return false;

  // Check whether tensor axes are all distinct
  std::set<loco::TensorAxis> values;

  values.insert(perm[FeatureAxis::Count]);
  values.insert(perm[FeatureAxis::Depth]);
  values.insert(perm[FeatureAxis::Height]);
  values.insert(perm[FeatureAxis::Width]);

  return values.size() == 4;
}

} // namespace

namespace loco
{

//
// Permutation
//
bool Permutation<Domain::Feature>::mapped(const FeatureAxis &axis_f) const
{
  assert(valid(axis_f) && "invalid feature axis");
  return _map.find(axis_f) != _map.end();
}

uint32_t Permutation<Domain::Feature>::axis(const FeatureAxis &axis_f) const
{
  assert(valid(axis_f) && "invalid feature axis");
  assert(mapped(axis_f) && "unmapped feature axis");
  return _map.at(axis_f);
}

uint32_t &Permutation<Domain::Feature>::axis(const FeatureAxis &axis_f)
{
  assert(valid(axis_f) && "invalid feature axis");
  return _map[axis_f];
}

//
// Permuting Encoder
//
FeatureShape PermutingEncoder<Domain::Feature>::shape(const TensorShape &in) const
{
  assert(valid() && "invalid permutation");

  FeatureShape out;

  out.count() = in.dim(_perm[FeatureAxis::Count]);
  out.depth() = in.dim(_perm[FeatureAxis::Depth]);
  out.height() = in.dim(_perm[FeatureAxis::Height]);
  out.width() = in.dim(_perm[FeatureAxis::Width]);

  return out;
}

TensorIndex PermutingEncoder<Domain::Feature>::value(const FeatureIndex &in) const
{
  assert(valid() && "invalid permutation");

  TensorIndex out;

  out.resize(4);

  out.at(_perm[FeatureAxis::Count]) = in.batch();
  out.at(_perm[FeatureAxis::Depth]) = in.channel();
  out.at(_perm[FeatureAxis::Height]) = in.row();
  out.at(_perm[FeatureAxis::Width]) = in.column();

  return out;
}

std::unique_ptr<FeatureEncoder> PermutingEncoder<Domain::Feature>::clone(void) const
{
  return std::make_unique<PermutingEncoder<Domain::Feature>>(_perm);
}

bool PermutingEncoder<Domain::Feature>::valid(void) const { return ::valid(_perm); }

//
// Permuting Decoder
//
TensorShape PermutingDecoder<Domain::Feature>::shape(const FeatureShape &in) const
{
  assert(valid() && "invalid permuation");

  TensorShape out;

  out.rank(4);

  out.dim(_perm[FeatureAxis::Count]) = in.count();
  out.dim(_perm[FeatureAxis::Depth]) = in.depth();
  out.dim(_perm[FeatureAxis::Height]) = in.height();
  out.dim(_perm[FeatureAxis::Width]) = in.width();

  return out;
}

FeatureIndex PermutingDecoder<Domain::Feature>::value(const TensorIndex &in) const
{
  assert(valid() && "invalid permutation");

  FeatureIndex out;

  out.batch() = in.at(_perm[FeatureAxis::Count]);
  out.channel() = in.at(_perm[FeatureAxis::Depth]);
  out.row() = in.at(_perm[FeatureAxis::Height]);
  out.column() = in.at(_perm[FeatureAxis::Width]);

  return out;
}

std::unique_ptr<FeatureDecoder> PermutingDecoder<Domain::Feature>::clone(void) const
{
  return std::make_unique<PermutingDecoder<Domain::Feature>>(_perm);
}

bool PermutingDecoder<Domain::Feature>::valid(void) const { return ::valid(_perm); }

} // namespace loco

/**
 * Filter Domain
 */
namespace
{

using loco::FilterAxis;

inline bool valid(const FilterAxis &axis)
{
  switch (axis)
  {
    case FilterAxis::Count:
      return true;
    case FilterAxis::Depth:
      return true;
    case FilterAxis::Height:
      return true;
    case FilterAxis::Width:
      return true;
    default:
      break;
  }

  return false;
}

inline bool valid(const loco::Permutation<loco::Domain::Filter> &perm)
{
  auto check = [&perm](FilterAxis axis_f) {
    if (!perm.mapped(axis_f))
      return false;
    return perm.axis(axis_f) < 4;
  };

  if (!check(FilterAxis::Count))
    return false;
  if (!check(FilterAxis::Depth))
    return false;
  if (!check(FilterAxis::Height))
    return false;
  if (!check(FilterAxis::Width))
    return false;

  // Check whether tensor axes are all distinct
  std::set<loco::TensorAxis> values;

  values.insert(perm[FilterAxis::Count]);
  values.insert(perm[FilterAxis::Depth]);
  values.insert(perm[FilterAxis::Height]);
  values.insert(perm[FilterAxis::Width]);

  return values.size() == 4;
}

} // namespace

namespace loco
{

//
// Permutation
//
bool Permutation<Domain::Filter>::mapped(const FilterAxis &axis_f) const
{
  assert(valid(axis_f) && "invalid filter axis");
  return _map.find(axis_f) != _map.end();
}

const uint32_t &Permutation<Domain::Filter>::axis(const FilterAxis &axis_f) const
{
  assert(valid(axis_f) && "invalid filter axis");
  assert(mapped(axis_f) && "unmapped filter axis");
  return _map.at(axis_f);
}

uint32_t &Permutation<Domain::Filter>::axis(const FilterAxis &axis_f)
{
  assert(valid(axis_f) && "invalid filter axis");
  return _map[axis_f];
}

//
// Permuting Encoder
//
FilterShape PermutingEncoder<Domain::Filter>::shape(const TensorShape &in) const
{
  assert(valid() && "invalid permutation");

  FilterShape out;

  out.count() = in.dim(_perm[FilterAxis::Count]);
  out.depth() = in.dim(_perm[FilterAxis::Depth]);
  out.height() = in.dim(_perm[FilterAxis::Height]);
  out.width() = in.dim(_perm[FilterAxis::Width]);

  return out;
}

TensorIndex PermutingEncoder<Domain::Filter>::value(const FilterIndex &in) const
{
  assert(valid() && "invalid permutation");

  TensorIndex out;

  out.resize(4);

  out.at(_perm[FilterAxis::Count]) = in.nth();
  out.at(_perm[FilterAxis::Depth]) = in.channel();
  out.at(_perm[FilterAxis::Height]) = in.row();
  out.at(_perm[FilterAxis::Width]) = in.column();

  return out;
}

bool PermutingEncoder<Domain::Filter>::valid(void) const { return ::valid(_perm); }

//
// Permuting Decoder
//
TensorShape PermutingDecoder<Domain::Filter>::shape(const FilterShape &in) const
{
  assert(valid() && "invalid permutation");

  TensorShape out;

  out.rank(4);
  out.dim(_perm[FilterAxis::Count]) = in.count();
  out.dim(_perm[FilterAxis::Depth]) = in.depth();
  out.dim(_perm[FilterAxis::Height]) = in.height();
  out.dim(_perm[FilterAxis::Width]) = in.width();

  return out;
}

FilterIndex PermutingDecoder<Domain::Filter>::value(const TensorIndex &in) const
{
  assert(valid() && "invalid permutation");

  FilterIndex out;

  out.nth() = in.at(_perm[FilterAxis::Count]);
  out.channel() = in.at(_perm[FilterAxis::Depth]);
  out.row() = in.at(_perm[FilterAxis::Height]);
  out.column() = in.at(_perm[FilterAxis::Width]);

  return out;
}

bool PermutingDecoder<Domain::Filter>::valid(void) const { return ::valid(_perm); }

} // namespace loco

/**
 * DepthwiseFilter Domain
 */
namespace
{

using loco::DepthwiseFilterAxis;

inline bool valid(const DepthwiseFilterAxis &axis)
{
  switch (axis)
  {
    case DepthwiseFilterAxis::Depth:
      return true;
    case DepthwiseFilterAxis::Multiplier:
      return true;
    case DepthwiseFilterAxis::Height:
      return true;
    case DepthwiseFilterAxis::Width:
      return true;
    default:
      break;
  }

  return false;
}

inline bool valid(const loco::Permutation<loco::Domain::DepthwiseFilter> &perm)
{
  auto check = [&perm](DepthwiseFilterAxis axis_f) {
    if (!perm.mapped(axis_f))
      return false;
    return perm.axis(axis_f) < 4;
  };

  if (!check(DepthwiseFilterAxis::Depth))
    return false;
  if (!check(DepthwiseFilterAxis::Multiplier))
    return false;
  if (!check(DepthwiseFilterAxis::Height))
    return false;
  if (!check(DepthwiseFilterAxis::Width))
    return false;

  // Check whether tensor axes are all distinct
  std::set<loco::TensorAxis> values;

  values.insert(perm[DepthwiseFilterAxis::Depth]);
  values.insert(perm[DepthwiseFilterAxis::Multiplier]);
  values.insert(perm[DepthwiseFilterAxis::Height]);
  values.insert(perm[DepthwiseFilterAxis::Width]);

  return values.size() == 4;
}

} // namespace

namespace loco
{

//
// Permutation
//
bool Permutation<Domain::DepthwiseFilter>::mapped(const DepthwiseFilterAxis &axis_f) const
{
  assert(valid(axis_f) && "invalid depthwise filter axis");
  return _map.find(axis_f) != _map.end();
}

const uint32_t &Permutation<Domain::DepthwiseFilter>::axis(const DepthwiseFilterAxis &axis_f) const
{
  assert(valid(axis_f) && "invalid depthwise filter axis");
  assert(mapped(axis_f) && "unmapped depthwise filter axis");
  return _map.at(axis_f);
}

uint32_t &Permutation<Domain::DepthwiseFilter>::axis(const DepthwiseFilterAxis &axis_f)
{
  assert(valid(axis_f) && "invalid depthwise filter axis");
  return _map[axis_f];
}

//
// Permuting Encoder
//
DepthwiseFilterShape PermutingEncoder<Domain::DepthwiseFilter>::shape(const TensorShape &in) const
{
  assert(valid() && "invalid permutation");

  DepthwiseFilterShape out;

  out.depth() = in.dim(_perm[DepthwiseFilterAxis::Depth]);
  out.multiplier() = in.dim(_perm[DepthwiseFilterAxis::Multiplier]);
  out.height() = in.dim(_perm[DepthwiseFilterAxis::Height]);
  out.width() = in.dim(_perm[DepthwiseFilterAxis::Width]);

  return out;
}

TensorIndex PermutingEncoder<Domain::DepthwiseFilter>::value(const DepthwiseFilterIndex &in) const
{
  assert(valid() && "invalid permutation");

  TensorIndex out;

  out.resize(4);

  out.at(_perm[DepthwiseFilterAxis::Depth]) = in.channel();
  out.at(_perm[DepthwiseFilterAxis::Multiplier]) = in.nth();
  out.at(_perm[DepthwiseFilterAxis::Height]) = in.row();
  out.at(_perm[DepthwiseFilterAxis::Width]) = in.column();

  return out;
}

bool PermutingEncoder<Domain::DepthwiseFilter>::valid(void) const { return ::valid(_perm); }

//
// Permuting Decoder
//
TensorShape PermutingDecoder<Domain::DepthwiseFilter>::shape(const DepthwiseFilterShape &in) const
{
  assert(valid() && "invalid permutation");

  TensorShape out;
  out.rank(4);

  out.dim(_perm[DepthwiseFilterAxis::Depth]) = in.depth();
  out.dim(_perm[DepthwiseFilterAxis::Multiplier]) = in.multiplier();
  out.dim(_perm[DepthwiseFilterAxis::Height]) = in.height();
  out.dim(_perm[DepthwiseFilterAxis::Width]) = in.width();

  return out;
}

DepthwiseFilterIndex PermutingDecoder<Domain::DepthwiseFilter>::value(const TensorIndex &in) const
{
  assert(valid() && "invalid permutation");
  assert(in.rank() == 4);

  DepthwiseFilterIndex out;

  out.channel() = in.at(_perm[DepthwiseFilterAxis::Depth]);
  out.nth() = in.at(_perm[DepthwiseFilterAxis::Multiplier]);
  out.row() = in.at(_perm[DepthwiseFilterAxis::Height]);
  out.column() = in.at(_perm[DepthwiseFilterAxis::Width]);

  return out;
}

bool PermutingDecoder<Domain::DepthwiseFilter>::valid(void) const { return ::valid(_perm); }

} // namespace loco

/**
 * Matrix Domain
 */
namespace
{

using loco::MatrixAxis;

inline bool valid(const MatrixAxis &axis)
{
  switch (axis)
  {
    case MatrixAxis::Height:
      return true;
    case MatrixAxis::Width:
      return true;
    default:
      break;
  }

  return false;
}

inline bool valid(const loco::Permutation<loco::Domain::Matrix> &perm)
{
  auto check = [&perm](MatrixAxis axis_f) {
    if (!perm.mapped(axis_f))
      return false;
    return perm.axis(axis_f) < 2;
  };

  if (!check(MatrixAxis::Height))
    return false;
  if (!check(MatrixAxis::Width))
    return false;

  // Check whether tensor axes are all distinct
  std::set<loco::TensorAxis> values;

  values.insert(perm[MatrixAxis::Height]);
  values.insert(perm[MatrixAxis::Width]);

  return values.size() == 2;
}

} // namespace

namespace loco
{

//
// Permutation
//
bool Permutation<Domain::Matrix>::mapped(const MatrixAxis &axis_f) const
{
  assert(valid(axis_f) && "invalid matrix axis");
  return _map.find(axis_f) != _map.end();
}

uint32_t Permutation<Domain::Matrix>::axis(const MatrixAxis &axis_f) const
{
  assert(valid(axis_f) && "invalid matrix axis");
  assert(mapped(axis_f) && "unmapped matrix axis");
  return _map.at(axis_f);
}

uint32_t &Permutation<Domain::Matrix>::axis(const MatrixAxis &axis_f)
{
  assert(valid(axis_f) && "invalid matrix axis");
  return _map[axis_f];
}

//
// Permuting Encoder
//
MatrixShape PermutingEncoder<Domain::Matrix>::shape(const TensorShape &in) const
{
  assert(valid() && "invalid permutation");

  MatrixShape out;

  out.height() = in.dim(_perm[MatrixAxis::Height]);
  out.width() = in.dim(_perm[MatrixAxis::Width]);

  return out;
}

TensorIndex PermutingEncoder<Domain::Matrix>::value(const MatrixIndex &in) const
{
  assert(valid() && "invalid permutation");

  TensorIndex out;

  out.resize(2);

  out.at(_perm[MatrixAxis::Height]) = in.row();
  out.at(_perm[MatrixAxis::Width]) = in.column();

  return out;
}

bool PermutingEncoder<Domain::Matrix>::valid(void) const { return ::valid(_perm); }

//
// Permuting Decoder
//
TensorShape PermutingDecoder<Domain::Matrix>::shape(const MatrixShape &in) const
{
  assert(valid() && "invalid permuation");

  TensorShape out;

  out.rank(2);

  out.dim(_perm[MatrixAxis::Height]) = in.height();
  out.dim(_perm[MatrixAxis::Width]) = in.width();

  return out;
}

MatrixIndex PermutingDecoder<Domain::Matrix>::value(const TensorIndex &in) const
{
  assert(valid() && "invalid permutation");

  MatrixIndex out;

  out.row() = in.at(_perm[MatrixAxis::Height]);
  out.column() = in.at(_perm[MatrixAxis::Width]);

  return out;
}

bool PermutingDecoder<Domain::Matrix>::valid(void) const { return ::valid(_perm); }

} // namespace loco
