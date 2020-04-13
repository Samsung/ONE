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

#ifndef __LOGO_RESOLVE_REDUNDANT_RESHAPE_PASS_H__
#define __LOGO_RESOLVE_REDUNDANT_RESHAPE_PASS_H__

#include <logo/Pass.h>

#include <loco.h>

namespace logo
{

/**
 * @brief  Remove redundant canonical FixedReshape
 *
 * @note  To effectively run this transform, canonical shape inference should be
 *        done ahead
 */
class ResolveRedundantReshapePass final : public Pass
{
public:
  const char *name(void) const final { return "ResolveRedundantReshapePass"; }

public:
  bool run(loco::Graph *graph) override;
};

} // namespace logo

#endif // __LOGO_RESOLVE_REDUNDANT_RESHAPE_PASS_H__
