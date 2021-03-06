/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_INTERP_REGISTRATION_H__
#define __ONERT_INTERP_REGISTRATION_H__

#include "ExecEnv.h"

#include "ir/Operation.h"

namespace onert
{
namespace interp
{

struct OpKernel
{
  std::function<void(ExecEnv *, const ir::Operation &)> prepare;
  std::function<void(const ExecEnv *, const ir::Operation &)> invoke;
};

// Defined in operations/ directory
#define INTERP_OP(InternalName) OpKernel *get##InternalName();
#include "InterpOps.lst"
#undef INTERP_OP

} // namespace interp
} // namespace onert

#endif // __ONERT_INTERP_REGISTRATION_H__
