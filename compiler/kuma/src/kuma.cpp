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

#include "kuma.h"

//
// Greedy Allocation Algorithm
//
namespace kuma
{

void solve(Context<Algorithm::Greedy> *ctx)
{
  uint32_t next = 0;

  for (uint32_t n = 0; n < ctx->item_count(); ++n)
  {
    ctx->mem_offset(n, next);
    next += ctx->item_size(n);
  }

  ctx->mem_total(next);
};

} // namespace kuma

//
// Linear Scan First Fit Algorithm
//
#include "IntervalSet.h"

namespace kuma
{

void solve(Context<Algorithm::LinearScanFirstFit> *ctx)
{
  using namespace kuma::details;

  uint32_t upper_bound = 0;
  std::map<ItemID, std::pair<uint32_t /* BEGIN */, uint32_t /* END */>> committed_items;

  // Allocate items in linear order (from item 0, item 1, ...)
  //
  // The implementor of Context is responsible for item ordering.
  for (uint32_t n = 0; n < ctx->item_count(); ++n)
  {
    IntervalSet intervals;

    for (auto item_in_conflict : ctx->conflict_with(n))
    {
      auto it = committed_items.find(item_in_conflict);

      // Skip if item_in_conflict is not committed yet
      if (it == committed_items.end())
      {
        continue;
      }

      auto const alloc_s = it->second.first;
      auto const alloc_e = it->second.second;
      intervals.insert(mask(alloc_s, alloc_e));
    }

    uint32_t const item_size = ctx->item_size(n);
    uint32_t const item_alloc_s = intervals.firstfit(item_size);
    uint32_t const item_alloc_e = item_alloc_s + item_size;

    // Notify "mem_offset"
    ctx->mem_offset(n, item_alloc_s);

    // Update "upper bound" and commit allocation
    upper_bound = std::max(upper_bound, item_alloc_e);
    committed_items[n] = std::make_pair(item_alloc_s, item_alloc_e);
  }

  // Notify "mem_total"
  ctx->mem_total(upper_bound);
}

} // namespace kuma
