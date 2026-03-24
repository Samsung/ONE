/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef __MEMORY_TRACKER_H__
#define __MEMORY_TRACKER_H__

#include "Memory.h"

#include <vector>
#include <unordered_map>

// A utility class to accumulate mulitple Memory objects and assign each
// a distinct index number, starting with 0.
//
// The user of this class is responsible for avoiding concurrent calls
// to this class from multiple threads.
class MemoryTracker
{
public:
  // Adds the memory, if it does not already exists.  Returns its index.
  // The memories should survive the tracker.
  uint32_t add(const Memory *memory);
  // Returns the number of memories contained.
  uint32_t size() const { return static_cast<uint32_t>(mKnown.size()); }
  // Returns the ith memory.
  const Memory *operator[](size_t i) const { return mMemories[i]; }

private:
  // The vector of Memory pointers we are building.
  std::vector<const Memory *> mMemories;
  // A faster way to see if we already have a memory than doing find().
  std::unordered_map<const Memory *, uint32_t> mKnown;
};

#endif // __MEMORY_TRACKER_H__
