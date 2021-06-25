/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_PARENT_INFO_H__
#define __ONERT_BACKEND_PARENT_INFO_H__

#include <ir/Index.h>
#include <ir/Coordinates.h>

namespace onert
{
namespace backend
{
namespace gpu_cl
{

/**
 * @brief	Struct to represent parent operand in child operand
 */
struct ParentInfo
{
  ir::OperandIndex parent;
  ir::Layout frontend_layout;
  ir::Coordinates coordinates;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_PARENT_INFO_H__
