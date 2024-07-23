/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __FME_DETECT_EQUALIZE_PATTERN_FINDER_H__
#define __FME_DETECT_EQUALIZE_PATTERN_FINDER_H__

#include "EqualizePattern.h"

#include <loco.h>

#include <vector>

namespace fme_detect
{

// Class to find EqualizePattrn in the Circle graph
class EqualizePatternFinder final
{
public:
  struct Context
  {
    bool _allow_dup_op = true;
  };

public:
  EqualizePatternFinder(const Context &ctx) : _ctx(ctx) {}

public:
  std::vector<EqualizePattern> find(loco::Graph *) const;

private:
  const Context &_ctx;
};

} // namespace fme_detect

#endif // __FME_DETECT_EQUALIZE_PATTERN_FINDER_H__
