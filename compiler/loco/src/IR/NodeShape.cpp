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

#include "loco/IR/NodeShape.h"

#include <cassert>
#include <stdexcept>

//
// BiasShape Support
//
namespace loco
{

void NodeShape::set(const BiasShape &shape)
{
  _domain = Domain::Bias;

  _dims.resize(1);
  _dims.at(0) = shape.length();
}

template <> BiasShape NodeShape::as(void) const
{
  assert(_domain == Domain::Bias);

  BiasShape res;

  res.length() = _dims.at(0);

  return res;
}

} // namespace loco

//
// DepthwiseFilterShape Support
//
namespace loco
{

void NodeShape::set(const DepthwiseFilterShape &shape)
{
  _domain = Domain::DepthwiseFilter;

  _dims.resize(4);
  _dims.at(0) = shape.multiplier();
  _dims.at(1) = shape.depth();
  _dims.at(2) = shape.height();
  _dims.at(3) = shape.width();
}

template <> DepthwiseFilterShape NodeShape::as(void) const
{
  assert(_domain == Domain::DepthwiseFilter);

  DepthwiseFilterShape res;

  res.multiplier() = _dims.at(0);
  res.depth() = _dims.at(1);
  res.height() = _dims.at(2);
  res.width() = _dims.at(3);

  return res;
}

} // namespace loco

//
// FeatureShape Support
//
namespace loco
{

void NodeShape::set(const FeatureShape &shape)
{
  _domain = Domain::Feature;

  _dims.resize(4);
  _dims.at(0) = shape.count();
  _dims.at(1) = shape.depth();
  _dims.at(2) = shape.height();
  _dims.at(3) = shape.width();
}

template <> FeatureShape NodeShape::as(void) const
{
  assert(_domain == Domain::Feature);

  FeatureShape res;

  res.count() = _dims.at(0);
  res.depth() = _dims.at(1);
  res.height() = _dims.at(2);
  res.width() = _dims.at(3);

  return res;
}

} // namespace loco

//
// FilterShape Support
//
namespace loco
{

void NodeShape::set(const FilterShape &shape)
{
  _domain = Domain::Filter;

  _dims.resize(4);
  _dims.at(0) = shape.count();
  _dims.at(1) = shape.depth();
  _dims.at(2) = shape.height();
  _dims.at(3) = shape.width();
}

template <> FilterShape NodeShape::as(void) const
{
  assert(_domain == Domain::Filter);

  FilterShape res;

  res.count() = _dims.at(0);
  res.depth() = _dims.at(1);
  res.height() = _dims.at(2);
  res.width() = _dims.at(3);

  return res;
}

} // namespace loco

//
// MatrixShape Support
//
namespace loco
{

void NodeShape::set(const MatrixShape &shape)
{
  _domain = Domain::Matrix;

  _dims.resize(2);
  _dims.at(0) = shape.height();
  _dims.at(1) = shape.width();
}

template <> MatrixShape NodeShape::as(void) const
{
  assert(_domain == Domain::Matrix);

  MatrixShape res;

  res.height() = _dims.at(0);
  res.width() = _dims.at(1);

  return res;
}

} // namespace loco

//
// TensorShape Support
//
namespace loco
{

void NodeShape::set(const TensorShape &shape)
{
  _domain = Domain::Tensor;

  _dims.resize(shape.rank());
  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    _dims.at(axis) = shape.dim(axis);
  }
}

template <> TensorShape NodeShape::as(void) const
{
  assert(_domain == Domain::Tensor);

  TensorShape res;

  res.rank(_dims.size());
  for (uint32_t axis = 0; axis < _dims.size(); ++axis)
  {
    res.dim(axis) = _dims.at(axis);
  }

  return res;
}

} // namespace loco

namespace loco
{

bool operator==(const NodeShape &lhs, const NodeShape &rhs)
{
  if (lhs.domain() != rhs.domain())
    return false;

  switch (lhs.domain())
  {
    case loco::Domain::Tensor:
    {
      auto lhs_t = lhs.as<TensorShape>();
      auto rhs_t = rhs.as<TensorShape>();
      if (lhs_t.rank() != rhs_t.rank())
        return false;
      for (uint32_t axis = 0; axis < lhs_t.rank(); ++axis)
      {
        if (!(lhs_t.dim(axis) == rhs_t.dim(axis)))
          return false;
      }
      return true;
    }

    case loco::Domain::Feature:
    {
      auto lhs_f = lhs.as<FeatureShape>();
      auto rhs_f = rhs.as<FeatureShape>();

      return (lhs_f.count() == rhs_f.count() && lhs_f.depth() == rhs_f.depth() &&
              lhs_f.height() == rhs_f.height() && lhs_f.width() == rhs_f.width());
    }

    case loco::Domain::Filter:
    {
      auto lhs_f = lhs.as<FilterShape>();
      auto rhs_f = rhs.as<FilterShape>();

      return (lhs_f.count() == rhs_f.count() && lhs_f.depth() == rhs_f.depth() &&
              lhs_f.height() == rhs_f.height() && lhs_f.width() == rhs_f.width());
    }

    case loco::Domain::DepthwiseFilter:
    {
      auto lhs_f = lhs.as<DepthwiseFilterShape>();
      auto rhs_f = rhs.as<DepthwiseFilterShape>();

      return (lhs_f.multiplier() == rhs_f.multiplier() && lhs_f.depth() == rhs_f.depth() &&
              lhs_f.height() == rhs_f.height() && lhs_f.width() == rhs_f.width());
    }

    case loco::Domain::Bias:
    {
      auto lhs_f = lhs.as<BiasShape>();
      auto rhs_f = rhs.as<BiasShape>();

      return (lhs_f.length() == rhs_f.length());
    }

    case loco::Domain::Matrix:
    {
      auto lhs_f = lhs.as<MatrixShape>();
      auto rhs_f = rhs.as<MatrixShape>();

      return (lhs_f.height() == rhs_f.height() && lhs_f.width() == rhs_f.width());
    }

    default:
      throw std::runtime_error("Not supported domain for NodeShape equality");
  }
  return false;
}

} // namespace loco
