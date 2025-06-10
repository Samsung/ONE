/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_SHARED_MEMORY_OPERANDS_H__
#define __ONERT_BACKEND_CPU_SHARED_MEMORY_OPERANDS_H__

#include "ir/IGraph.h"
#include "ir/OperandIndexMap.h"

namespace onert::backend::cpu
{
/*
 * Find indexed of operands assigned to tensors which can share memory (indicate the same buffer).
 * Note that it's applicable for operations that do NOT change data but only shape like Reshape.
 */
ir::OperandIndexMap<ir::OperandIndex> findSharedMemoryOperandIndexes(const ir::IGraph &graph);

} // namespace onert::backend::cpu

#endif // __ONERT_BACKEND_CPU_SHARED_MEMORY_OPERANDS_H__
