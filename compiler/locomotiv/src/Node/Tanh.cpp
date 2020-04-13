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

#include "NodeExecution.h"

#include <cmath>

namespace
{

struct Func final : public locomotiv::UnaryFunc
{
  float apply(float v) const final { return std::tanh(v); }
};

} // namespace

namespace locomotiv
{

void NodeExecution::execute(loco::Tanh *tanh)
{
  Func f;

  eltwise_unary(tanh, f);
}

} // namespace locomotiv
