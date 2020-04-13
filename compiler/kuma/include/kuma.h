/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __KUMA_H__
#define __KUMA_H__

#include <cstdint>
#include <set>

namespace kuma
{

// Supported algorithms
enum Algorithm
{
  // No reuse
  Greedy,
  LinearScanFirstFit,
};

/**
 * Each algorithm defines its own context. The context describes its in and out.
 */
template <Algorithm Alg> class Context;

using ItemID = uint32_t;
using ItemSize = uint32_t;

using MemoryOffset = uint32_t;
using MemorySize = uint32_t;

//
// Greedy Algorithm
//
template <> class Context<Algorithm::Greedy>
{
public:
  virtual ~Context() = default;

public: // Inputs
  // count() returns the number of items to be allocated
  virtual uint32_t item_count(void) const = 0;

  // size(N) returns the size of the N-th item
  virtual ItemSize item_size(const ItemID &) const = 0;

public: // Outputs
  virtual void mem_offset(const ItemID &, const MemoryOffset &) = 0;
  virtual void mem_total(const MemorySize &) = 0;
};

void solve(Context<Greedy> *);

//
// Linear Scan First-Fit Algorithm
//
template <> class Context<Algorithm::LinearScanFirstFit>
{
public:
  virtual ~Context() = default;

public: // Inputs
  // count() returns the number of items to be allocated
  virtual uint32_t item_count(void) const = 0;

  // size(N) returns the size of the N-th item
  virtual ItemSize item_size(const ItemID &) const = 0;

  // conflict_with(N) returns all the items that are in conflict with item N
  // - An item N is said to be in conflict with item M if item M and N cannot have overlap
  //
  // NOTE
  // - conflict_with(N) SHOULD NOT include N itself
  virtual std::set<ItemID> conflict_with(const ItemID &) const = 0;

public: // Outputs
  virtual void mem_offset(const ItemID &, const MemoryOffset &) = 0;
  virtual void mem_total(const MemorySize &) = 0;
};

void solve(Context<Algorithm::LinearScanFirstFit> *);

} // namespace kuma

#endif // __KUMA_H__
