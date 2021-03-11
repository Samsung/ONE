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

#include "loco/IR/Nodes.h"
#include "loco/IR/Graph.h"

#include <cassert>
#include <limits>

// This file validates "Nodes.h". Please DO NOT remove this file.
namespace
{

/**
 * @note  This function is currently only used in assert. Compiler will
 *        warn/error this function as unused in Release build.
 *        Making inline will make compiler happy.
 */
// Is it possible to update lhs as rhs?
inline bool dtype_assignable(loco::DataType lhs, loco::DataType rhs)
{
  if (lhs == loco::DataType::Unknown)
  {
    return true;
  }

  // lhs is already known, and thus rhs should be matched
  return lhs == rhs;
}

} // namespace

/**
 * Push
 */
namespace loco
{

void Push::index(const GraphOutputIndex &index)
{
  // Push internally stores "GraphOutputIndex" as int64_t
  _index = static_cast<int64_t>(index);
}

GraphOutputIndex Push::index(void) const
{
  assert(_index >= std::numeric_limits<GraphOutputIndex>::min());
  assert(_index <= std::numeric_limits<GraphOutputIndex>::max());
  return static_cast<GraphOutputIndex>(_index);
}

void link(GraphOutput *output, Push *push) { push->index(output->index()); }

Push *push_node(Graph *g, const GraphOutputIndex &index)
{
  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    if (auto push = dynamic_cast<Push *>(g->nodes()->at(n)))
    {
      if (push->indexed() && push->index() == index)
      {
        return push;
      }
    }
  }
  return nullptr;
}

} // namespace loco

/**
 * Pull
 */
namespace loco
{

void Pull::index(const GraphInputIndex &index)
{
  // ASSUMPTION
  //
  // It is possible to update index multiple times, but only with the same value!
  assert(!indexed() or _index == index);

  if (indexed())
  {
    assert(_index == index);
    return;
  }

  // Push internally stores "GraphInputIndex" as int64_t
  _index = static_cast<int64_t>(index);

  // ASSUMPTION: The return value of graph() never changes!
  if (graph() != nullptr && _dtype != loco::DataType::Unknown)
  {
    // Update Graph-level input only if it is not yet specified
    if (graph()->inputs()->at(_index)->dtype() == DataType::Unknown)
    {
      graph()->inputs()->at(_index)->dtype(_dtype);
    }
    assert(graph()->inputs()->at(_index)->dtype() == _dtype);
    graph()->inputs()->at(_index)->dtype(_dtype);

    // Reset the locally cached data
    _dtype = DataType::Unknown;
  }
}

GraphInputIndex Pull::index(void) const
{
  assert(_index >= std::numeric_limits<GraphInputIndex>::min());
  assert(_index <= std::numeric_limits<GraphInputIndex>::max());
  return static_cast<GraphInputIndex>(_index);
}

void Pull::dtype(const DataType &dt)
{
  // ASSUMPTION: "dtype" is never invalidated!
  assert(dt != loco::DataType::Unknown);
  // ASSUMPTION
  //
  // It is possible to update index multiple times, but only with the same value!
  if (indexed())
  {
    assert(dtype_assignable(graph()->inputs()->at(_index)->dtype(), dt));
    graph()->inputs()->at(_index)->dtype(dt);
    return;
  }

  // Use local cache
  _dtype = dt;
}

DataType Pull::dtype(void) const
{
  if (graph() != nullptr and _index >= 0)
  {
    assert(_dtype == DataType::Unknown);
    return graph()->inputs()->at(_index)->dtype();
  }
  else
  {
    return _dtype;
  }
}

void link(GraphInput *input, Pull *pull) { pull->index(input->index()); }

Pull *pull_node(Graph *g, const GraphInputIndex &index)
{
  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    if (auto pull = dynamic_cast<Pull *>(g->nodes()->at(n)))
    {
      if (pull->indexed() && pull->index() == index)
      {
        return pull;
      }
    }
  }
  return nullptr;
}

} // namespace loco

/**
 * ConstGen
 */
namespace loco
{

template <DataType DT> uint32_t ConstGen::size(void) const
{
  assert(dtype() == DT);
  assert(_data.size() % sizeof(typename DataTypeImpl<DT>::Type) == 0);
  return _data.size() / sizeof(typename DataTypeImpl<DT>::Type);
}

template <DataType DT> void ConstGen::size(uint32_t l)
{
  assert(dtype() == DT);
  _data.resize(l * sizeof(typename DataTypeImpl<DT>::Type));
}

template <DataType DT> const typename DataTypeImpl<DT>::Type &ConstGen::at(uint32_t n) const
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<const typename DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

template <DataType DT> typename DataTypeImpl<DT>::Type &ConstGen::at(uint32_t n)
{
  assert(dtype() == DT);
  assert(n < size<DT>());
  return *(reinterpret_cast<typename DataTypeImpl<DT>::Type *>(_data.data()) + n);
}

#define INSTANTIATE(DT)                                                             \
  template uint32_t ConstGen::size<DT>(void) const;                                 \
  template void ConstGen::size<DT>(uint32_t);                                       \
  template const typename DataTypeImpl<DT>::Type &ConstGen::at<DT>(uint32_t) const; \
  template typename DataTypeImpl<DT>::Type &ConstGen::at<DT>(uint32_t);

INSTANTIATE(DataType::S32);
INSTANTIATE(DataType::FLOAT32);

#undef INSTANTIATE

} // namespace loco

/**
 * TensorBroadcast
 */
namespace loco
{

bool TensorBroadcast::Mapping::defined(const TensorAxis &axis) const
{
  return _content.find(axis) != _content.end();
}

const Dimension &TensorBroadcast::Mapping::dim(const TensorAxis &axis) const
{
  return _content.at(axis);
}

Dimension &TensorBroadcast::Mapping::dim(const TensorAxis &axis) { return _content[axis]; }

} // namespace loco
