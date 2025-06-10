/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_COMPILER_OPERAND_LOWER_INFO_H__
#define __ONERT_COMPILER_OPERAND_LOWER_INFO_H__

#include "backend/Backend.h"
#include "util/Set.h"

namespace onert::compiler
{

using BackendSet = util::Set<const backend::Backend *>;

class OperandLowerInfo
{
public:
  OperandLowerInfo()
  {
    // DO NOTHING
  }

public:
  const BackendSet &def_backends(void) const { return _def_backends; }
  const BackendSet &use_backends(void) const { return _use_backends; }

public:
  void addDefBackend(const backend::Backend *backend) { _def_backends.add(backend); }
  void addUseBackend(const backend::Backend *backend) { _use_backends.add(backend); }
  void removeDefBackend(const backend::Backend *backend) { _def_backends.remove(backend); }
  void removeUseBackend(const backend::Backend *backend) { _use_backends.remove(backend); }

private:
  BackendSet _def_backends;
  BackendSet _use_backends;
};

} // namespace onert::compiler

#endif // __ONERT_COMPILER_OPERAND_LOWER_INFO_H__
