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

#ifndef __FME_APPLY_EQUALIZE_PATTERN_CHECK_H__
#define __FME_APPLY_EQUALIZE_PATTERN_CHECK_H__

#include "EqualizePattern.h"

#include <loco.h>

#include <vector>

namespace fme_apply
{

// Check the patterns are valid on the graph
void check_patterns_valid(loco::Graph *g, const std::vector<EqualizePattern> &patterns);

} // namespace fme_apply

#endif //__FME_APPLY_EQUALIZE_PATTERN_CHECK_H__
