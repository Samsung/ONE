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

#ifndef __NEURUN_IR_OPERAND_INDEX_H__
#define __NEURUN_IR_OPERAND_INDEX_H__

#include "util/Index.h"

namespace neurun
{
namespace ir
{

struct OperationIndexTag;
using OperationIndex = ::neurun::util::Index<uint32_t, OperationIndexTag>;

struct OperandIndexTag;
using OperandIndex = ::neurun::util::Index<uint32_t, OperandIndexTag>;

struct IOIndexTag;
using IOIndex = ::neurun::util::Index<uint32_t, IOIndexTag>;

struct OpSequenceIndexTag;
using OpSequenceIndex = ::neurun::util::Index<uint32_t, OpSequenceIndexTag>;

} // namespace ir
} // namespace neurun

#endif // __NEURUN_IR_OPERAND_INDEX_H__
