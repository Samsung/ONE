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

#ifndef __FME_APPLY_INSERT_SCALE_SHIFT_H__
#define __FME_APPLY_INSERT_SCALE_SHIFT_H__

#include <loco.h>

#include "EqualizePattern.h"

namespace fme_apply
{

/**
 * @brief Class to insert scale/shift virtual Ops to loco::Graph
 */
class InsertScaleShift
{
public:
  InsertScaleShift(std::vector<EqualizePattern> &patterns) : _patterns{patterns}
  {
    // DO NOTHING
  }

public:
  void run(loco::Graph *graph);

private:
  std::vector<EqualizePattern> &_patterns;
};

} // namespace fme_apply

#endif //__FME_APPLY_INSERT_SCALE_SHIFT_H__
