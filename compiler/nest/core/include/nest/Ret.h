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

#ifndef __NEST_RET_H__
#define __NEST_RET_H__

#include "nest/DomainID.h"
#include "nest/Expr.h"

namespace nest
{

// WARNING Ret SHOULD BE immutable
//
// The copy/move constructor of Module class simply copies the shared pointer under the assumption
// that Ret is immutable.
class Ret
{
public:
  Ret(const DomainID &id, const expr::Subscript &sub) : _id{id}, _sub{sub}
  {
    // DO NOTHING
  }

public:
  const DomainID &id(void) const { return _id; }
  const expr::Subscript &sub(void) const { return _sub; }

private:
  DomainID _id;
  expr::Subscript _sub;
};

} // namespace nest

#endif // __NEST_RET_H__
