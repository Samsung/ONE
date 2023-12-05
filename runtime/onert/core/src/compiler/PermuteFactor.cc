/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "compiler/PermuteFactor.h"

#include <assert.h>
#include <ostream>

#include "backend/Backend.h"

std::ostream &operator<<(std::ostream &os, const onert::compiler::PermuteFactor &obj)
{
  assert(obj.backend() && obj.backend()->config());
  return os << "(" << obj.backend()->config()->id() << ")";
}
