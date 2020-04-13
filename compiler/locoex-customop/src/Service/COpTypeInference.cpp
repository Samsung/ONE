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

#include "locoex/Service/COpTypeInference.h"

#include "locoex/COpDialect.h"
#include "locoex/COpCall.h"

#include <cassert>

namespace locoex
{

bool COpTypeInferenceRule::recognize(const loco::Dialect *d) const
{
  // This rule recognizes only "COpDialect" dialect!
  return COpDialect::get() == d;
}

bool COpTypeInferenceRule::infer(const loco::Node *node, loco::DataType &dtype) const
{
  assert(node->dialect() == COpDialect::get());

  auto customop = dynamic_cast<const COpCall *>(node);

  assert(customop != nullptr);
  assert(customop->dtype() != loco::DataType::Unknown);

  dtype = customop->dtype();

  return true;
}

} // namespace locoex
