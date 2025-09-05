/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_OPERATION_TRANSPOSE_CONV_H__
#define __ONERT_IR_OPERATION_TRANSPOSE_CONV_H__

#include "ir/InternalType.h"
#include "ir/Operation.h"
#include "ir/Padding.h"

#include <memory>

namespace onert::ir::operation
{

class TransposeConv : public Operation
{
public:
  enum Input
  {
    OUTPUT_SHAPE = 0,
    KERNEL,
    INPUT
  };

  struct Param
  {
    Padding padding;
    Stride stride;
  };

public:
  TransposeConv(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
                const Param &param);

public:
  void accept(OperationVisitor &v) const override;
  OpCode opcode() const final { return OpCode::TransposeConv; }

public:
  const Param &param() const { return _param; }

private:
  Param _param;
};

} // namespace onert::ir::operation

#endif // __ONERT_IR_OPERATION_TRANSPOSE_CONV_H__
