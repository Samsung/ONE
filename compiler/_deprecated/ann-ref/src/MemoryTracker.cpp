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

#define LOG_TAG "Memory"

#include "NeuralNetworks.h" // For ANEURALNETWORKS_...
#include "MemoryTracker.h"

#include "Logging.h"

#include <unistd.h> // It's for 'close' and 'dup'
                    // TODO-NNRT : Remove this if included another header including this.

uint32_t MemoryTracker::add(const Memory *memory)
{
  VLOG(MODEL) << __func__ << " for " << memory;
  // See if we already have this memory. If so,
  // return its index.
  auto i = mKnown.find(memory);
  if (i != mKnown.end())
  {
    return i->second;
  }
  VLOG(MODEL) << "It's new";
  // It's a new one.  Save it an assign an index to it.
  size_t next = mKnown.size();
  if (next > 0xFFFFFFFF)
  {
    LOG(ERROR) << "ANeuralNetworks more than 2^32 memories.";
    return ANEURALNETWORKS_BAD_DATA;
  }
  uint32_t idx = static_cast<uint32_t>(next);
  mKnown[memory] = idx;
  mMemories.push_back(memory);
  return idx;
}
