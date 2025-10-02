/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_OPERATION_ATTENTION_H__
#define __ONERT_IR_OPERATION_ATTENTION_H__

#include "ir/Operation.h"
#include "ir/InternalType.h"

namespace onert::ir::operation
{

class Attention : public Operation
{
public:
  enum Input
  {
    INPUT = 0,
    WQ = 1,
    WK = 2,
    WV = 3,
    WO = 4,
    COS = 5,
    SIN = 6,
    MASK = 7,
    K_CACHE = 8,
    V_CACHE = 9,
    POS = 10,
  };

  struct Param
  {
    int layer_idx;
  };

public:
  Attention(const OperandIndexSequence &inputs, const OperandIndexSequence &outputs,
            const Param &param);

public:
  void accept(OperationVisitor &v) const override;
  OpCode opcode() const final { return OpCode::Attention; }

public:
  const Param &param() const { return _param; }

private:
  Param _param;
};

} // namespace onert::ir::operation

#endif // __ONERT_IR_OPERATION_ATTENTION_H__
