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

#ifndef __NEURUN_IR_OPERATION_LOWER_INFO_H__
#define __NEURUN_IR_OPERATION_LOWER_INFO_H__

#include <string>

#include <ir/operand/PermuteFactor.h>

namespace neurun
{
namespace backend
{
class Backend;
} // namespace backend
} // namespace neurun

namespace neurun
{
namespace ir
{
namespace operation
{

class LowerInfo
{
public:
  LowerInfo(const backend::Backend *backend, Layout layout);
  const backend::Backend *backend() const { return _permute_factor.backend(); }
  Layout layout() const { return _permute_factor.layout(); }

private:
  operand::PermuteFactor _permute_factor;
};

} // namespace operation
} // namespace ir
} // namespace neurun

#endif // __NEURUN_IR_OPERATION_LOWER_INFO_H__
