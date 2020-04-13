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

#ifndef __NEST_MODULE_H__
#define __NEST_MODULE_H__

#include "nest/VarContext.h"
#include "nest/DomainContext.h"
#include "nest/Ret.h"
#include "nest/Block.h"

namespace nest
{

class Module
{
public:
  Module() = default;

private:
  VarContext _var_ctx;

public:
  VarContext &var(void) { return _var_ctx; }
  const VarContext &var(void) const { return _var_ctx; }

private:
  DomainContext _domain_ctx;

public:
  DomainContext &domain(void) { return _domain_ctx; }
  const DomainContext &domain(void) const { return _domain_ctx; }

private:
  Block _block;

public:
  const Block &block(void) const { return _block; }

public:
  void push(const Expr &expr);

private:
  std::shared_ptr<Ret> _ret;

public:
  // NOTE Do NOT invoke ret() before ret(expr) call
  const Ret &ret(void) const;

public:
  // NOTE Only one ret(expr) call is allowed for each module
  void ret(const Closure &closure);
};

} // namespace nest

#endif // __NEST_MODULE_H__
