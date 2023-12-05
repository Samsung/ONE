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

#ifndef __ONERT_COMPILER_OP_SEQUENCE_LOWER_INFO_H__
#define __ONERT_COMPILER_OP_SEQUENCE_LOWER_INFO_H__

#include <string>

#include <compiler/PermuteFactor.h>
#include <ir/Layout.h>

namespace onert
{
namespace backend
{
class Backend;
} // namespace backend
} // namespace onert

namespace onert
{
namespace compiler
{

class OperationLowerInfo
{
public:
  OperationLowerInfo(const backend::Backend *backend);
  const backend::Backend *backend() const { return _permute_factor.backend(); }

private:
  PermuteFactor _permute_factor;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_OP_SEQUENCE_LOWER_INFO_H__
