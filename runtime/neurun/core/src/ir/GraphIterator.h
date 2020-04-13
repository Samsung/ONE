/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NEURUN_IR_GRAPH_ITERATOR_H__
#define __NEURUN_IR_GRAPH_ITERATOR_H__

#include <type_traits>

#include "ir/Index.h"

namespace neurun
{
namespace ir
{

class Graph;
class Operation;

template <bool is_const> class Iterator
{
public:
  using GraphRef = typename std::conditional<is_const, const Graph &, Graph &>::type;
  using IndexRef = const OperationIndex &;
  using NodeRef = typename std::conditional<is_const, const Operation &, Operation &>::type;
  using IterFn = std::function<void(IndexRef, NodeRef)>;

public:
  virtual ~Iterator() = default;
  virtual void iterate(GraphRef graph, const IterFn &fn) const = 0;
};

template <bool is_const = false> class DefaultIterator final : public Iterator<is_const>
{
public:
  using GraphRef = typename Iterator<is_const>::GraphRef;
  using IndexRef = typename Iterator<is_const>::IndexRef;
  using NodeRef = typename Iterator<is_const>::NodeRef;
  using IterFn = typename Iterator<is_const>::IterFn;

public:
  void iterate(GraphRef graph, const IterFn &fn) const;
};
using DefaultConstIterator = DefaultIterator<true>;

template <bool is_const = false> class PostDfsIterator final : public Iterator<is_const>
{
public:
  using GraphRef = typename Iterator<is_const>::GraphRef;
  using IndexRef = typename Iterator<is_const>::IndexRef;
  using NodeRef = typename Iterator<is_const>::NodeRef;
  using IterFn = typename Iterator<is_const>::IterFn;

public:
  void iterate(GraphRef graph, const IterFn &fn) const;
};
using PostDfsConstIterator = PostDfsIterator<true>;

} // namespace ir
} // namespace neurun

#endif // __NEURUN_IR_GRAPH_ITERATOR_H__
