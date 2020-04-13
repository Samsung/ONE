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

#ifndef __ANN_CONTEXT_H__
#define __ANN_CONTEXT_H__

#include "ANN/Binder.h"

#include <map>
#include <vector>

#include <memory>

struct ANNContext
{
public:
  ANNBinder *create(coco::Block *blk);

public:
  uint32_t count(void) const { return _binders.size(); }

public:
  ANNBinder *nth(uint32_t n) { return _binders.at(n).get(); }
  const ANNBinder *nth(uint32_t n) const { return _binders.at(n).get(); }

public:
  ANNBinder *find(const coco::Block *blk) const
  {
    auto it = _map.find(blk);

    if (it == _map.end())
    {
      return nullptr;
    }

    return it->second;
  }

private:
  std::vector<std::unique_ptr<ANNBinder>> _binders;
  std::map<const coco::Block *, ANNBinder *> _map;
};

#endif // __ANN_CONTEXT_H__
