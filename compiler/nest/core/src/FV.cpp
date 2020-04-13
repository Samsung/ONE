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

#include "nest/FV.h"

namespace
{

using namespace nest;
using namespace nest::expr;

class Collector final : public Visitor<void>
{
public:
  Collector(std::set<VarID> &out) : _out(out)
  {
    // DO NOTHING
  }

public:
  void visit(const VarNode *e) override { _out.insert(e->id()); }

  void visit(const DerefNode *e) override
  {
    for (uint32_t n = 0; n < e->sub().rank(); ++n)
    {
      e->sub().at(n)->accept(this);
    }
  }

  void visit(const AddNode *e) override
  {
    e->lhs()->accept(this);
    e->rhs()->accept(this);
  }

  void visit(const MulNode *e) override
  {
    e->lhs()->accept(this);
    e->rhs()->accept(this);
  }

private:
  std::set<nest::VarID> &_out;
};

} // namespace

namespace nest
{

std::set<VarID> FV::in(const Expr &expr)
{
  std::set<VarID> res;

  Collector collector{res};
  expr->accept(collector);

  return res;
}

} // namespace nest
