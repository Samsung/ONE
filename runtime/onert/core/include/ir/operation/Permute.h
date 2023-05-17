/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_OPERATION_PERMUTE_H__
#define __ONERT_IR_OPERATION_PERMUTE_H__

#include "ir/Operation.h"

namespace onert
{
namespace backend
{
class BackendContext;
} // namespace backend
} // namespace onert

namespace onert
{
namespace ir
{
namespace operation
{

class Permute : public Operation
{
public:
  enum class Type
  {
    NHWC_TO_NCHW,
    NCHW_TO_NHWC,
    COPY
  };

public:
  void accept(OperationVisitor &v) const override;
  void accept(MutableOperationVisitor &v) override;
  OpCode opcode() const final { return OpCode::Permute; }

public:
  Permute(const OperandIndex &input, const OperandIndex &output, Type type);

public:
  Type getPermuteType() const { return _type; }

private:
  Type _type;
};

} // namespace operation
} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_PERMUTE_H__
