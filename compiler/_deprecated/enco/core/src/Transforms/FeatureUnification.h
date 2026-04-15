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

#ifndef __ENCO_TRANSFORM_FEATURE_UNIFICATION_H__
#define __ENCO_TRANSFORM_FEATURE_UNIFICATION_H__

#include "Code.h"
#include "Pass.h"

namespace enco
{

/**
 * @brief Remove duplicated feature objects inside each bag
 *
 * >>> BEFORE <<<
 * %b = Bag(...)
 *
 * %feature_0 = Feature(...) at %b
 * %feature_1 = Feature(...) at %b
 *
 * ...
 * Use(%feature_0)
 * ...
 * Use(%feature_1)
 * ...
 *
 * >>> AFTER <<<
 * %b = Bag(...)
 *
 * %feature_0 = Feature(...) at %b
 * ~~%feature_1 = Feature(...) at %b~~ <- REMOVED
 *
 * ...
 * Use(%feature_0)
 * ...
 * Use(%feature_0)
 * ...
 *
 * Note that all the occurrences of "%feature_1" are replaced with "%feature_0"
 */
void unify_feature(enco::Code *code);

struct FeatureUnificationPass final : public Pass
{
  PASS_CTOR(FeatureUnificationPass)
  {
    // DO NOTHING
  }
  void run(const SessionID &sess) const override { unify_feature(code(sess)); }
};

} // namespace enco

#endif // __ENCO_TRANSFORM_FEATURE_UNIFICATION_H__
