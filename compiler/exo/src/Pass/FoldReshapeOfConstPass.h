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

#ifndef __PASS_FOLD_RESHAPE_OF_CONST_PASS_H__
#define __PASS_FOLD_RESHAPE_OF_CONST_PASS_H__

#include <logo/Pass.h>

namespace exo
{

/**
 * @brief Class to fuse TFLReshape + TFLConst into one equivalent TFLConst
 *
 * <before>
 *      TFLConst --- TFLReshape --- Out
 *
 * <after>
 *      TFLConst --- TFLReshape ---
 *      TFLConst (new) ------------ Out
 *
 * TODO This pass is for temporary. Deprecate this pass.
 */
struct FoldReshapeOfConstPass final : public logo::Pass
{
  const char *name(void) const final { return "exo::FoldReshapeOfConstPass"; }

  bool run(loco::Graph *g) final;
};

} // namespace exo

#endif // __PASS_FOLD_RESHAPE_OF_CONST_PASS_H__
