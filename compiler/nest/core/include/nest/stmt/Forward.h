/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NEST_STMT_FORWARD_H__
#define __NEST_STMT_FORWARD_H__

#include "nest/stmt/Macro.h"

namespace nest
{
namespace stmt
{

#define STMT(Tag) class NEST_STMT_CLASS_NAME(Tag);
#include "nest/stmt/Node.def"
#undef STMT

} // namespace stmt
} // namespace nest

#endif // __NEST_STMT_FORWARD_H__
