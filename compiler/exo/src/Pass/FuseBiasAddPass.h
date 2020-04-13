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

#ifndef __PASS_FUSE_BIASADD_PASS_H__
#define __PASS_FUSE_BIASADD_PASS_H__

#include <logo/Pass.h>

namespace exo
{

/**
 * @brief Class to fuse TFLAdd or TFLSub into Bias input of the following ops:
 *    - TFLConv2D, TFLDepthwiseConv2D
 *    - TODO Consider to add FullyConnected, etc.
 *
 * Case 1. Conv2D and TFLAdd
 *
 * BEFORE:
 *
 *                   TFLConst A (a scalar or a tensor of shape [1] or [depth of TFLConv2D])
 *                        |
 *   Foo -- TFLConv2D -- TFLAdd (or TFLSub) -- Bar
 *                |
 *   TFLConst B --+ (bias)
 *
 * AFTER:
 *   Foo ----- TFLConv2D ----- Bar
 *                 |
 *   TFLConst A' --+ (bias)
 *
 *   TFLConst B (dead node)
 *
 *   TFLAdd (or TFLSub) (dead node)
 *
 * @note TFLSub, of which x() == TFLConv2D and y() == TFLConst, will be fused.
 *       If x() == TFLConst and y() == TFLConv2D, it won't be fused.
 */
struct FuseBiasAddPass final : public logo::Pass
{
  const char *name(void) const final { return "exo::FuseBiasAddPass"; }

  bool run(loco::Graph *g) final;
};

} // namespace exo

#endif // __PASS_FUSE_BIASADD_PASS_H__
