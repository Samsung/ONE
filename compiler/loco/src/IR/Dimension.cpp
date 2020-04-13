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

#include "loco/IR/Dimension.h"

namespace loco
{

bool operator==(const Dimension &lhs, const Dimension &rhs)
{
  return lhs.known() && rhs.known() && lhs.value() == rhs.value();
}

bool operator==(const Dimension &lhs, uint32_t rhs) { return lhs.known() && lhs.value() == rhs; }
bool operator==(uint32_t lhs, const Dimension &rhs) { return rhs.known() && lhs == rhs.value(); }

Dimension make_dimension(void) { return Dimension{}; }

} // namespace loco
