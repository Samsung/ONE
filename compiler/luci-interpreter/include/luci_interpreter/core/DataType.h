/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_CORE_DATATYPE_H
#define LUCI_INTERPRETER_CORE_DATATYPE_H

#include <loco/IR/DataType.h>
#include <loco/IR/DataTypeTraits.h>
#include <luci/IR/DataTypeHelper.h>

#include <cstddef>

namespace luci_interpreter
{

using DataType = loco::DataType;

template <DataType DT> using DataTypeImpl = loco::DataTypeImpl<DT>;

inline size_t getDataTypeSize(DataType data_type) { return luci::size(data_type); }

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_DATATYPE_H
