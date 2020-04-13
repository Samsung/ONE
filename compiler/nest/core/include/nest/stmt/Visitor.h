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

#ifndef __NEST_STMT_VISITOR_H__
#define __NEST_STMT_VISITOR_H__

#include "nest/stmt/Macro.h"
#include "nest/stmt/Forward.h"

namespace nest
{
namespace stmt
{

template <typename T> struct Visitor
{
  virtual ~Visitor() = default;

#define STMT(Tag) virtual T visit(const NEST_STMT_CLASS_NAME(Tag) *) = 0;
#include "nest/stmt/Node.def"
#undef STMT
};

} // namespace stmt
} // namespace nest

#endif // __NEST_STMT_VISITOR_H__
