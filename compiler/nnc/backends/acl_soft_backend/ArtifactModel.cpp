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

#include "ArtifactModel.h"

namespace nnc
{

using namespace std;

// ArtifactFunctionCall implementation
ArtifactFunctionCall::ArtifactFunctionCall(string func_name,
                                           list<shared_ptr<ArtifactExpr>> param_list,
                                           shared_ptr<ArtifactExpr> on, ArtifactCallType call_type)
  : _funcName(std::move(func_name)), _callType(call_type), _on(std::move(on)),
    _paramList(std::move(param_list))
{
}

// ArtifactBlock implementation.
shared_ptr<ArtifactVariable>
ArtifactBlock::var(const string &type_name, const string &var_name,
                   const list<shared_ptr<ArtifactExpr>> &dimensions,
                   const list<std::shared_ptr<ArtifactExpr>> &initializers)
{
  auto var = make_shared<ArtifactVariable>(type_name, var_name, dimensions, initializers);
  _statements.push_back(var);
  return var;
}

shared_ptr<ArtifactFunctionCall>
ArtifactBlock::call(const string &func_name, const list<shared_ptr<ArtifactExpr>> &param_list,
                    shared_ptr<ArtifactExpr> call_on, ArtifactCallType call_type)
{
  auto func_call = make_shared<ArtifactFunctionCall>(func_name, param_list, call_on, call_type);
  _statements.push_back(func_call);
  return func_call;
}

shared_ptr<ArtifactRet> ArtifactBlock::ret(shared_ptr<ArtifactExpr> expr)
{
  auto ret = make_shared<ArtifactRet>(expr);
  _statements.push_back(ret);
  return ret;
}

shared_ptr<ArtifactBreak> ArtifactBlock::brk()
{
  auto brk = make_shared<ArtifactBreak>();
  _statements.push_back(brk);
  return brk;
}

shared_ptr<ArtifactCont> ArtifactBlock::cont()
{
  auto cont = make_shared<ArtifactCont>();
  _statements.push_back(cont);
  return cont;
}

shared_ptr<ArtifactForLoop> ArtifactBlock::forLoop(shared_ptr<ArtifactVariable> init,
                                                   shared_ptr<ArtifactExpr> cond,
                                                   shared_ptr<ArtifactExpr> iter)
{
  auto loop = make_shared<ArtifactForLoop>(init, cond, iter);
  _statements.push_back(loop);
  return loop;
}

shared_ptr<ArtifactIf> ArtifactBlock::ifCond(shared_ptr<ArtifactExpr> cond)
{
  auto ifb = make_shared<ArtifactIf>(cond);
  _statements.push_back(ifb);
  return ifb;
}

shared_ptr<ArtifactUnaryExpr> ArtifactBlock::un(ArtifactUnOp op, shared_ptr<ArtifactExpr> expr)
{
  auto un = make_shared<ArtifactUnaryExpr>(op, expr);
  _statements.push_back(un);
  return un;
}

shared_ptr<ArtifactBinaryExpr> ArtifactBlock::bin(ArtifactBinOp op, shared_ptr<ArtifactExpr> left,
                                                  shared_ptr<ArtifactExpr> right)
{
  auto bin = make_shared<ArtifactBinaryExpr>(op, left, right);
  _statements.push_back(bin);
  return bin;
}

shared_ptr<ArtifactUnaryExpr> ArtifactBlock::heapNew(shared_ptr<ArtifactExpr> expr)
{
  auto heap_new = make_shared<ArtifactUnaryExpr>(ArtifactUnOp::heapNew, expr);
  _statements.push_back(heap_new);
  return heap_new;
}

shared_ptr<ArtifactUnaryExpr> ArtifactBlock::heapFree(shared_ptr<ArtifactExpr> expr)
{
  auto heap_del = make_shared<ArtifactUnaryExpr>(ArtifactUnOp::heapFree, expr);
  _statements.push_back(heap_del);
  return heap_del;
}

} // namespace nnc
