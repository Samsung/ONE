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

#ifndef _MIR_VISITOR_H_
#define _MIR_VISITOR_H_

namespace mir
{

// Forward declare operations as we don't need anything but references
namespace ops
{
#define HANDLE_OP(OpType, OpClass) class OpClass;
#include "mir/Operations.inc"
#undef HANDLE_OP
} // namespace ops

class Operation;

/**
 * @brief Interface for visitors
 * Use in MIR component if you want to enforce to implement visits for all operations
 */
class IVisitor
{
public:
#define HANDLE_OP(OpType, OpClass) virtual void visit(ops::OpClass &) = 0;
#include "Operations.inc"
#undef HANDLE_OP

  virtual ~IVisitor() = default;
};

/**
 * @brief Base visitor with empty fallback function
 */
class Visitor : public IVisitor
{
public:
#define HANDLE_OP(OpType, OpClass) void visit(ops::OpClass &) override;
#include "Operations.inc"
#undef HANDLE_OP

protected:
  virtual void visit_fallback(Operation &) {}
};

} // namespace mir

#endif //_MIR_VISITOR_H_
