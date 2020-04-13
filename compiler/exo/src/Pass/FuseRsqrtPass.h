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

#ifndef __FUSE_RSQRT_PASS_H__
#define __FUSE_RSQRT_PASS_H__

#include <logo/Pass.h>

namespace exo
{

/**
 * @brief Class to fuse TFLSqrt that is divided(TFLDiv) by 1, into TFLRsqrt
 *
 * <BEFORE>
 *
 *    TFLConst(1) ------
 *                      \
 *    A --- TFLSqrt --- TFLDiv --- B
 *
 * <AFTER>
 *
 *    A --- TFLRsqrt --- B
 */
struct FuseRsqrtPass final : public logo::Pass
{
  const char *name(void) const final { return "exo::FuseRsqrtPass"; }

  bool run(loco::Graph *g) final;
};

} // namespace exo

#endif // __FUSE_RSQRT_PASS_H__
