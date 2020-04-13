/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef NNCC_UTIL_H
#define NNCC_UTIL_H

#include "mir/ops/AddOp.h"
#include "mir/ops/AvgPool2DOp.h"
#include "mir/ops/ConcatOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/MaxPool2DOp.h"
#include "mir/ops/MulOp.h"
#include "mir/ops/OutputOp.h"
#include "mir/ops/ReluOp.h"
#include "mir/ops/TanhOp.h"
#include "mir/ops/TransposeOp.h"
#include "mir/Visitor.h"

namespace nnc
{

class DumpVisitor : public mir::Visitor
{
public:
  explicit DumpVisitor(std::ostream &s) : _s(s) {}

  void visit(mir::ops::InputOp &op) override { _s << "i_" << std::to_string(op.getId()) << "."; };

  void visit(mir::ops::TanhOp &op) override { _s << "th_" << std::to_string(op.getId()) << "."; }

  void visit(mir::ops::MulOp &op) override { _s << "s_" << std::to_string(op.getId()) << "."; }

  void visit(mir::ops::AddOp &op) override { _s << "b_" << std::to_string(op.getId()) << "."; }

  void visit(mir::ops::ReluOp &op) override { _s << "r_" << std::to_string(op.getId()) << "."; }

  void visit(mir::ops::AvgPool2DOp &op) override
  {
    _s << "p_" << std::to_string(op.getId()) << ".";
  }

  void visit(mir::ops::MaxPool2DOp &op) override
  {
    _s << "p_" << std::to_string(op.getId()) << ".";
  }

  void visit(mir::ops::TransposeOp &op) override
  {
    _s << "t_" << std::to_string(op.getId()) << ".";
  }

  void visit(mir::ops::Conv2DOp &op) override
  {
    _s << "conv_" << std::to_string(op.getId()) << ".";
  }

  void visit(mir::ops::ConstantOp &op) override
  {
    _s << "const_" << std::to_string(op.getId()) << ".";
  }

  std::ostream &_s;
};

} // namespace nnc
#endif // NNCC_UTIL_H
